//! Conversion of one completed binary-output evaluation into a static PRIM
//! dataset.

use std::sync::Arc;

use graphcal_model_protocol::{
    InputColumn, OutputSlot, PortName, SessionInput, SessionSchema, Value, VariantName, WireType,
};
use thiserror::Error;

use crate::DataError;
use crate::data::{Dataset, Feature};
use crate::evaluation::{
    ChunkResult, EvalRequest, EvaluationId, OutputRowError, RowFailure, RowOutcome, SessionMismatch,
};

/// Category labels of a Boolean input, indexed by its value.
const BOOL_CATEGORIES: [&str; 2] = ["false", "true"];

/// Errors raised when converting one completed binary-output request to PRIM
/// input data.
#[derive(Debug, Error, PartialEq)]
pub enum EvaluationDatasetError {
    /// The result belongs to a different request.
    #[error("request ID {request_id} does not match result ID {result_id}")]
    MismatchedId {
        /// Identifier on the retained request.
        request_id: EvaluationId,
        /// Identifier echoed by the result.
        result_id: EvaluationId,
    },

    /// The request was built for a different session schema.
    #[error(transparent)]
    InvalidInputs(#[from] SessionMismatch),

    /// The chosen output does not exist.
    #[error("output slot {slot} is out of bounds for a session with {output_count} outputs")]
    OutputSlotOutOfBounds {
        /// Requested output slot.
        slot: usize,
        /// Number of session outputs.
        output_count: usize,
    },

    /// The chosen output is not Boolean, so it cannot classify cases of
    /// interest.
    #[error("output '{output}' is not a Boolean output")]
    NonBooleanOutput {
        /// The chosen output.
        output: PortName,
        /// Its wire type.
        wire: WireType,
    },

    /// The result has a different number of rows from the request.
    #[error("result has {actual} rows, but request {id} has {expected} input rows")]
    RowCountMismatch {
        /// Evaluation request identifier.
        id: EvaluationId,
        /// Number of input rows.
        expected: usize,
        /// Number of result rows.
        actual: usize,
    },

    /// One result row records a model or validation failure.
    #[error("evaluation row {row} failed: {failure}")]
    FailedRow {
        /// Zero-based row position.
        row: usize,
        /// Explicit failure data.
        failure: RowFailure,
    },

    /// A supposedly successful row does not match the session's outputs.
    #[error("evaluation row {row} has invalid outputs: {error}")]
    InvalidOutputRow {
        /// Zero-based row position.
        row: usize,
        /// Output schema violation.
        error: OutputRowError,
    },

    /// The resulting static dataset violates a dataset invariant.
    #[error(transparent)]
    InvalidDataset(#[from] DataError),
}

/// Converts one successful evaluation with a Boolean output into a static PRIM
/// dataset.
///
/// Each session input becomes one PRIM feature with the input's name:
/// quantities become continuous features (in the input's wire unit), integers
/// become integer features, keys become categorical features labeled by
/// variant name, and Booleans become categorical features labeled `"false"`
/// and `"true"`. The Boolean `output` classifies cases of interest.
///
/// This function deliberately rejects the first failed row rather than silently
/// dropping it. A driver that wants a different missing-evaluation policy must
/// make that policy explicit before constructing a [`Dataset`].
///
/// # Errors
///
/// Returns [`EvaluationDatasetError`] when request/result identity or row count
/// differs, the request belongs to another session, `output` is not a Boolean
/// output of `schema`, an output row violates `schema`, or any row failed.
pub fn evaluation_to_dataset(
    schema: &Arc<SessionSchema>,
    request: EvalRequest,
    result: ChunkResult,
    output: OutputSlot,
) -> Result<Dataset, EvaluationDatasetError> {
    if request.id() != result.id() {
        return Err(EvaluationDatasetError::MismatchedId {
            request_id: request.id(),
            result_id: result.id(),
        });
    }
    request.validate_against(schema)?;
    let selected =
        schema
            .output(output)
            .ok_or_else(|| EvaluationDatasetError::OutputSlotOutOfBounds {
                slot: output.get(),
                output_count: schema.outputs().len(),
            })?;
    if selected.wire() != &WireType::Bool {
        return Err(EvaluationDatasetError::NonBooleanOutput {
            output: selected.name().clone(),
            wire: selected.wire().clone(),
        });
    }
    if request.inputs().rows() != result.rows().len() {
        return Err(EvaluationDatasetError::RowCountMismatch {
            id: request.id(),
            expected: request.inputs().rows(),
            actual: result.rows().len(),
        });
    }

    let cases_of_interest = result
        .into_rows()
        .into_iter()
        .enumerate()
        .map(|(row, outcome)| match outcome {
            RowOutcome::Success(outputs) => {
                outputs
                    .validate_against(schema)
                    .map_err(|error| EvaluationDatasetError::InvalidOutputRow { row, error })?;
                match outputs.value(output) {
                    Some(Value::Bool(value)) => Ok(*value),
                    // Validation above proved the slot holds a Boolean.
                    _ => Err(EvaluationDatasetError::InvalidOutputRow {
                        row,
                        error: OutputRowError::KindMismatch {
                            output: selected.name().clone(),
                        },
                    }),
                }
            }
            RowOutcome::Failure(failure) => Err(EvaluationDatasetError::FailedRow { row, failure }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let features = schema
        .inputs()
        .iter()
        .zip(request.into_inputs().columns())
        .map(|(input, column)| input_feature(input, column))
        .collect::<Result<Vec<_>, _>>()?;
    Dataset::new(features, cases_of_interest).map_err(Into::into)
}

fn input_feature(input: &SessionInput, column: &InputColumn) -> Result<Feature, DataError> {
    let name = input.name().as_str();
    match (column, input.wire()) {
        (InputColumn::Quantity(values), _) => {
            Feature::continuous(name, values.iter().map(|value| value.get()).collect())
        }
        (InputColumn::Int(values), _) => Feature::integer(name, values.clone()),
        (InputColumn::Bool(values), _) => Feature::categorical_from_dictionary(
            name,
            values.iter().map(|value| i32::from(*value)).collect(),
            BOOL_CATEGORIES.map(str::to_owned).to_vec(),
        ),
        (InputColumn::Key(positions), wire) => {
            // A validated request batch pairs every key column with a key
            // input; any other pairing yields no dictionary and is rejected by
            // the dictionary encoder below.
            let variants: &[VariantName] = match wire {
                WireType::Key(key) => key.variants(),
                _ => &[],
            };
            let codes = positions
                .iter()
                .map(|position| {
                    i32::try_from(position.get()).map_err(|_| DataError::TooManyCategories {
                        name: name.to_owned(),
                        count: variants.len(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Feature::categorical_from_dictionary(
                name,
                codes,
                variants
                    .iter()
                    .map(|variant| variant.as_str().to_owned())
                    .collect(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{FiniteF64, RequestBatch, RowFailure as ProtocolRowFailure};

    use super::*;
    use crate::{
        Evaluator, FeatureView, InProcessEvaluator, InputRow, ModelSchemaBuilder, RowContext,
    };

    fn schema() -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .integer("n", 0, 3)
            .unwrap()
            .categorical("mode", "Mode", ["safe", "risky"])
            .unwrap()
            .boolean("enabled")
            .unwrap()
            .integer_output("count")
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap()
    }

    fn request(schema: &Arc<SessionSchema>) -> EvalRequest {
        let WireType::Key(mode) = schema.inputs()[2].wire() else {
            panic!("mode must be a key input");
        };
        let safe = mode.position("safe").unwrap();
        let risky = mode.position("risky").unwrap();
        let inputs = RequestBatch::new(
            Arc::clone(schema),
            vec![
                InputColumn::Quantity(vec![
                    FiniteF64::new(0.25).unwrap(),
                    FiniteF64::new(0.75).unwrap(),
                    FiniteF64::new(0.5).unwrap(),
                ]),
                InputColumn::Int(vec![0, 1, 2]),
                InputColumn::Key(vec![risky, risky, safe]),
                InputColumn::Bool(vec![true, false, true]),
            ],
        )
        .unwrap();
        EvalRequest::new(EvaluationId::new(1), 0, inputs)
    }

    fn evaluate(schema: &Arc<SessionSchema>, request: &EvalRequest) -> ChunkResult {
        let n = schema.input_slot("n").unwrap();
        let evaluator = InProcessEvaluator::new(
            Arc::clone(schema),
            move |row: InputRow<'_>, _: RowContext| {
                let n = row.int(n)?;
                if n == 2 {
                    return Err(ProtocolRowFailure::EvaluationError {
                        message: "diverged".to_owned(),
                    });
                }
                Ok(vec![Value::Int(n), Value::Bool(n == 1)])
            },
        );
        evaluator
            .evaluate(vec![request.clone()])
            .next()
            .unwrap()
            .unwrap()
    }

    #[test]
    fn converts_every_input_kind_to_a_prim_feature() {
        let schema = schema();
        let request = request(&schema);
        let n = schema.input_slot("n").unwrap();
        let evaluator = InProcessEvaluator::new(
            Arc::clone(&schema),
            move |row: InputRow<'_>, _: RowContext| {
                let n = row.int(n)?;
                Ok(vec![Value::Int(n), Value::Bool(n == 1)])
            },
        );
        let result = evaluator
            .evaluate(vec![request.clone()])
            .next()
            .unwrap()
            .unwrap();
        let failure = schema.output_slot("failure").unwrap();
        let dataset = evaluation_to_dataset(&schema, request, result, failure).unwrap();

        assert_eq!(dataset.cases_of_interest(), [false, true, false]);
        let views = dataset.feature_views().collect::<Vec<_>>();
        assert_eq!(views[0], FeatureView::Continuous(&[0.25, 0.75, 0.5]));
        assert_eq!(views[1], FeatureView::Integer(&[0, 1, 2]));
        let FeatureView::Categorical(mode) = views[2] else {
            panic!("keys become categorical features");
        };
        let modes = mode
            .codes()
            .iter()
            .map(|code| mode.category(*code).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(modes, ["risky", "risky", "safe"]);
        let FeatureView::Categorical(enabled) = views[3] else {
            panic!("Booleans become categorical features");
        };
        let enabled = enabled
            .codes()
            .iter()
            .map(|code| enabled.category(*code).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(enabled, ["true", "false", "true"]);
    }

    #[test]
    fn conversion_rejects_failed_rows_instead_of_dropping_them() {
        let schema = schema();
        let request = request(&schema);
        let result = evaluate(&schema, &request);
        let failure = schema.output_slot("failure").unwrap();

        assert!(matches!(
            evaluation_to_dataset(&schema, request, result, failure),
            Err(EvaluationDatasetError::FailedRow {
                row: 2,
                failure: RowFailure::Model(ProtocolRowFailure::EvaluationError { .. })
            })
        ));
    }

    #[test]
    fn conversion_requires_a_boolean_output() {
        let schema = schema();
        let request = request(&schema);
        let result = evaluate(&schema, &request);
        let count = schema.output_slot("count").unwrap();

        assert!(matches!(
            evaluation_to_dataset(&schema, request, result, count),
            Err(EvaluationDatasetError::NonBooleanOutput { output, wire: WireType::Int })
                if output.as_str() == "count"
        ));
    }

    #[test]
    fn conversion_rejects_mismatched_ids() {
        let schema = schema();
        let request = request(&schema);
        let result = evaluate(&schema, &request);
        let other = EvalRequest::new(EvaluationId::new(2), 0, request.into_inputs());
        let failure = schema.output_slot("failure").unwrap();

        assert!(matches!(
            evaluation_to_dataset(&schema, other, result, failure),
            Err(EvaluationDatasetError::MismatchedId { .. })
        ));
    }
}
