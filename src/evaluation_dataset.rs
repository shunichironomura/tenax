//! Conversion of completed evaluations into static PRIM datasets.

use graphcal_model_protocol::{
    InputColumn, KeyRef, KeyType, OutputSlot, PortName, SessionInput, SessionOutput, Value,
    WireType,
};
use thiserror::Error;

use crate::DataError;
use crate::case_rule::CaseRule;
use crate::data::{Dataset, Feature};
use crate::evaluation::{EvaluationId, RowFailure, RowOutcome};
use crate::evaluation_table::EvaluationTable;

/// Category labels of a Boolean input, indexed by its value.
const BOOL_CATEGORIES: [&str; 2] = ["false", "true"];

/// Errors raised when converting completed evaluations to PRIM input data.
#[derive(Debug, Error, PartialEq)]
pub enum EvaluationDatasetError {
    /// The chosen output does not exist.
    #[error("output slot {slot} is out of bounds for a session with {output_count} outputs")]
    OutputSlotOutOfBounds {
        /// Requested output slot.
        slot: usize,
        /// Number of session outputs.
        output_count: usize,
    },

    /// The case rule cannot classify values of the chosen output.
    #[error("the case rule does not apply to output '{output}'")]
    InapplicableRule {
        /// The chosen output.
        output: PortName,
        /// Its wire type.
        wire: WireType,
    },

    /// One row records a model or validation failure.
    #[error("row {row} of evaluation {id} failed: {failure}")]
    FailedRow {
        /// Evaluation request identifier.
        id: EvaluationId,
        /// Zero-based row position within the request.
        row: usize,
        /// Explicit failure data.
        failure: RowFailure,
    },

    /// The resulting static dataset violates a dataset invariant.
    #[error(transparent)]
    InvalidDataset(#[from] DataError),
}

impl EvaluationTable {
    /// Converts every session input into one PRIM feature, in session order,
    /// with one observation per table row.
    ///
    /// Each feature has the input's name: quantities and datetimes become
    /// continuous features (in the input's wire unit and seconds since J2000,
    /// respectively), integers become integer features, keys become
    /// categorical features labeled by entry (see below), and Booleans become
    /// categorical features labeled `"false"` and `"true"`. Key entries are
    /// labeled by variant name, `Fin` position, or coordinate in its SI unit.
    ///
    /// Inputs of failed rows are included, so the features describe every
    /// evaluated row.
    ///
    /// # Errors
    ///
    /// Returns [`DataError`] when the table is empty, or when an input is
    /// complex, indexed, or algebraic, which PRIM cannot use yet.
    pub fn features(&self) -> Result<Vec<Feature>, DataError> {
        self.schema()
            .inputs()
            .iter()
            .enumerate()
            .map(|(position, input)| {
                let columns = self
                    .evaluations()
                    .iter()
                    .filter_map(|evaluation| evaluation.request().inputs().columns().get(position))
                    .collect::<Vec<_>>();
                input_feature(input, &concatenate(&columns))
            })
            .collect()
    }

    /// Reads the Boolean `output` of every row as its case-of-interest
    /// classification, in table order.
    ///
    /// This is [`EvaluationTable::cases_where`] with [`CaseRule::IsTrue`].
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationDatasetError`] when `output` is out of range or not
    /// Boolean, or when any row failed.
    pub fn cases(&self, output: OutputSlot) -> Result<Vec<bool>, EvaluationDatasetError> {
        self.cases_where(output, &CaseRule::IsTrue)
    }

    /// Classifies every row by applying `rule` to its value of `output`, in
    /// table order.
    ///
    /// Failed rows are rejected rather than silently dropped; a caller that
    /// wants another missing-evaluation policy must apply it explicitly
    /// before classifying.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationDatasetError`] when `output` is out of range, the
    /// rule does not apply to the output's wire type (including a key that is
    /// not an entry of the output's axis), or any row failed.
    pub fn cases_where(
        &self,
        output: OutputSlot,
        rule: &CaseRule,
    ) -> Result<Vec<bool>, EvaluationDatasetError> {
        let selected = self.selected_output(output)?;
        let inapplicable = || EvaluationDatasetError::InapplicableRule {
            output: selected.name().clone(),
            wire: selected.wire().clone(),
        };
        if !rule.applies_to(selected.wire()) {
            return Err(inapplicable());
        }
        // The table validated every successful row against the session, so
        // each value has the output's wire type, which the rule applies to.
        self.classify(output, |value| {
            rule.classify(value).ok_or_else(inapplicable)
        })
    }

    /// Classifies every row by calling `classify` on its value of `output`,
    /// in table order.
    ///
    /// Use this for a classification that [`CaseRule`] cannot express, such
    /// as one over an indexed or algebraic output. Values are in the output's
    /// wire unit.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationDatasetError`] when `output` is out of range or any
    /// row failed.
    pub fn cases_by(
        &self,
        output: OutputSlot,
        mut classify: impl FnMut(&Value) -> bool,
    ) -> Result<Vec<bool>, EvaluationDatasetError> {
        self.selected_output(output)?;
        self.classify(output, |value| Ok(classify(value)))
    }

    fn selected_output(
        &self,
        output: OutputSlot,
    ) -> Result<&SessionOutput, EvaluationDatasetError> {
        self.schema()
            .output(output)
            .ok_or_else(|| EvaluationDatasetError::OutputSlotOutOfBounds {
                slot: output.get(),
                output_count: self.schema().outputs().len(),
            })
    }

    /// Classifies the value of `output` in every row; `output` is in range.
    fn classify(
        &self,
        output: OutputSlot,
        mut classify: impl FnMut(&Value) -> Result<bool, EvaluationDatasetError>,
    ) -> Result<Vec<bool>, EvaluationDatasetError> {
        self.evaluations()
            .iter()
            .flat_map(|evaluation| {
                evaluation
                    .rows()
                    .iter()
                    .enumerate()
                    .map(move |(row, outcome)| (evaluation.id(), row, outcome))
            })
            .map(|(id, row, outcome)| match outcome {
                RowOutcome::Success(outputs) => outputs.value(output).map_or_else(
                    || {
                        Err(EvaluationDatasetError::OutputSlotOutOfBounds {
                            slot: output.get(),
                            output_count: outputs.values().len(),
                        })
                    },
                    &mut classify,
                ),
                RowOutcome::Failure(failure) => Err(EvaluationDatasetError::FailedRow {
                    id,
                    row,
                    failure: failure.clone(),
                }),
            })
            .collect()
    }

    /// Builds a PRIM dataset from every input (see
    /// [`EvaluationTable::features`]) and the Boolean `output` (see
    /// [`EvaluationTable::cases`]).
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationDatasetError`] when the cases cannot be read, an
    /// input cannot become a feature, or the dataset is invalid.
    pub fn dataset(&self, output: OutputSlot) -> Result<Dataset, EvaluationDatasetError> {
        self.dataset_where(output, &CaseRule::IsTrue)
    }

    /// Builds a PRIM dataset from every input (see
    /// [`EvaluationTable::features`]) and `rule` applied to `output` (see
    /// [`EvaluationTable::cases_where`]).
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationDatasetError`] when the cases cannot be classified,
    /// an input cannot become a feature, or the dataset is invalid.
    pub fn dataset_where(
        &self,
        output: OutputSlot,
        rule: &CaseRule,
    ) -> Result<Dataset, EvaluationDatasetError> {
        let cases = self.cases_where(output, rule)?;
        Ok(Dataset::new(self.features()?, cases)?)
    }
}

/// Joins the columns of one input across evaluations, in order.
///
/// Every column was validated against the same input, so all share the
/// variant of the first; an empty list yields an empty quantity column.
pub fn concatenate(columns: &[&InputColumn]) -> InputColumn {
    fn join<T: Clone>(
        columns: &[&InputColumn],
        values: impl Fn(&InputColumn) -> Option<&[T]>,
    ) -> Vec<T> {
        columns
            .iter()
            .filter_map(|column| values(column))
            .flatten()
            .cloned()
            .collect()
    }
    match columns.first() {
        None | Some(InputColumn::Quantity(_)) => {
            InputColumn::Quantity(join(columns, |column| match column {
                InputColumn::Quantity(values) => Some(values),
                _ => None,
            }))
        }
        Some(InputColumn::Datetime(_)) => {
            InputColumn::Datetime(join(columns, |column| match column {
                InputColumn::Datetime(values) => Some(values),
                _ => None,
            }))
        }
        Some(InputColumn::Int(_)) => InputColumn::Int(join(columns, |column| match column {
            InputColumn::Int(values) => Some(values),
            _ => None,
        })),
        Some(InputColumn::Bool(_)) => InputColumn::Bool(join(columns, |column| match column {
            InputColumn::Bool(values) => Some(values),
            _ => None,
        })),
        Some(InputColumn::Key(_)) => InputColumn::Key(join(columns, |column| match column {
            InputColumn::Key(values) => Some(values),
            _ => None,
        })),
        Some(InputColumn::Complex(_)) => {
            InputColumn::Complex(join(columns, |column| match column {
                InputColumn::Complex(values) => Some(values),
                _ => None,
            }))
        }
        Some(InputColumn::Structured(_)) => {
            InputColumn::Structured(join(columns, |column| match column {
                InputColumn::Structured(values) => Some(values),
                _ => None,
            }))
        }
    }
}

fn input_feature(input: &SessionInput, column: &InputColumn) -> Result<Feature, DataError> {
    let name = input.name().as_str();
    match (column, input.wire()) {
        (InputColumn::Quantity(values) | InputColumn::Datetime(values), _) => {
            Feature::continuous(name, values.iter().map(|value| value.get()).collect())
        }
        (InputColumn::Complex(_), _) => Err(DataError::UnsupportedFeature {
            name: name.to_owned(),
            kind: "complex",
        }),
        (InputColumn::Structured(_), wire) => Err(DataError::UnsupportedFeature {
            name: name.to_owned(),
            kind: match wire {
                WireType::Indexed(_) => "indexed",
                _ => "algebraic",
            },
        }),
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
            let labels = match wire {
                WireType::Key(key) => key_labels(key),
                _ => Vec::new(),
            };
            let codes = positions
                .iter()
                .map(|position| {
                    i32::try_from(position.get()).map_err(|_| DataError::TooManyCategories {
                        name: name.to_owned(),
                        count: labels.len(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Feature::categorical_from_dictionary(name, codes, labels)
        }
    }
}

/// The category label of every entry of `key`, in axis order: the variant
/// name, the `Fin` position, or the coordinate in its SI unit.
fn key_labels(key: &KeyType) -> Vec<String> {
    key.positions()
        .filter_map(|position| key.key(position))
        .map(|entry| match entry {
            KeyRef::Variant(variant) => variant.as_str().to_owned(),
            KeyRef::Position(position) => position.to_string(),
            KeyRef::Coordinate(coordinate) => coordinate.get().to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graphcal_model_protocol::{
        FiniteF64, RequestBatch, RowFailure as ProtocolRowFailure, SessionSchema,
    };

    use super::*;
    use crate::{
        ChunkResult, EvalRequest, Evaluator, FeatureView, InProcessEvaluator, InputRow,
        ModelSchemaBuilder, RowContext,
    };

    fn table(
        schema: &Arc<SessionSchema>,
        evaluations: impl IntoIterator<Item = (EvalRequest, ChunkResult)>,
    ) -> EvaluationTable {
        let mut table = EvaluationTable::new(Arc::clone(schema));
        for (request, result) in evaluations {
            table.push(request, result).unwrap();
        }
        table
    }

    #[test]
    fn key_labels_name_fin_positions_and_coordinates() {
        use graphcal_model_protocol::{Axis, AxisKind, Dimension, IndexName};

        let key = |index: &str, kind| {
            KeyType::new(Axis::new(IndexName::new(index).unwrap(), kind).unwrap())
        };
        assert_eq!(
            key_labels(&key("Fin(3)", AxisKind::Finite { size: 3 })),
            ["0", "1", "2"]
        );
        let coordinates = [0.0, 0.5]
            .map(|coordinate| FiniteF64::new(coordinate).unwrap())
            .to_vec();
        assert_eq!(
            key_labels(&key(
                "Step",
                AxisKind::Coordinate {
                    dimension: Dimension::dimensionless(),
                    coordinates,
                }
            )),
            ["0", "0.5"]
        );
    }

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
        request_with_id(schema, 1)
    }

    fn request_with_id(schema: &Arc<SessionSchema>, id: u128) -> EvalRequest {
        let WireType::Key(mode) = schema.inputs()[2].wire() else {
            panic!("mode must be a key input");
        };
        let variant = |name: &str| {
            let variant = graphcal_model_protocol::VariantName::new(name).unwrap();
            mode.position(KeyRef::Variant(&variant)).unwrap()
        };
        let (safe, risky) = (variant("safe"), variant("risky"));
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
        EvalRequest::new(EvaluationId::new(id), 0, inputs)
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
        let dataset = table(&schema, [(request, result)])
            .dataset(failure)
            .unwrap();

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

        let table = table(&schema, [(request, result)]);
        assert!(matches!(
            table.dataset(failure),
            Err(EvaluationDatasetError::FailedRow {
                row: 2,
                failure: RowFailure::Model(ProtocolRowFailure::EvaluationError { .. }),
                ..
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
            table(&schema, [(request, result)]).cases(count),
            Err(EvaluationDatasetError::InapplicableRule { output, wire: WireType::Int })
                if output.as_str() == "count"
        ));
    }

    #[test]
    fn features_and_cases_span_every_evaluation_in_order() {
        let schema = schema();
        let n = schema.input_slot("n").unwrap();
        let evaluator = InProcessEvaluator::new(
            Arc::clone(&schema),
            move |row: InputRow<'_>, _: RowContext| {
                let n = row.int(n)?;
                Ok(vec![Value::Int(n), Value::Bool(n == 1)])
            },
        );
        let requests = [request_with_id(&schema, 1), request_with_id(&schema, 2)];
        let results = evaluator
            .evaluate(requests.to_vec())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let table = table(&schema, requests.into_iter().zip(results));

        let features = table.features().unwrap();
        assert_eq!(features[1].integer_values(), Some(&[0, 1, 2, 0, 1, 2][..]));
        let failure = schema.output_slot("failure").unwrap();
        assert_eq!(
            table.cases(failure).unwrap(),
            [false, true, false, false, true, false]
        );
        let wider = ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .boolean_output("a")
            .unwrap()
            .boolean_output("b")
            .unwrap()
            .boolean_output("c")
            .unwrap()
            .build()
            .unwrap();
        assert!(matches!(
            table.cases(wider.output_slot("c").unwrap()),
            Err(EvaluationDatasetError::OutputSlotOutOfBounds { slot: 2, .. })
        ));
    }

    #[test]
    fn rules_classify_non_boolean_outputs() {
        let schema = schema();
        let n = schema.input_slot("n").unwrap();
        let evaluator = InProcessEvaluator::new(
            Arc::clone(&schema),
            move |row: InputRow<'_>, _: RowContext| {
                let n = row.int(n)?;
                Ok(vec![Value::Int(n), Value::Bool(n == 1)])
            },
        );
        let request = request(&schema);
        let result = evaluator
            .evaluate(vec![request.clone()])
            .next()
            .unwrap()
            .unwrap();
        let table = table(&schema, [(request, result)]);
        let count = schema.output_slot("count").unwrap();

        let at_least_one = CaseRule::Integer {
            comparison: crate::Comparison::AtLeast,
            threshold: 1,
        };
        assert_eq!(
            table.cases_where(count, &at_least_one).unwrap(),
            [false, true, true]
        );
        assert_eq!(
            table
                .cases_by(count, |value| matches!(value, Value::Int(2)))
                .unwrap(),
            [false, false, true]
        );
        let dataset = table.dataset_where(count, &at_least_one).unwrap();
        assert_eq!(dataset.cases_of_interest(), [false, true, true]);

        let quantity = CaseRule::Quantity {
            comparison: crate::Comparison::Below,
            threshold: FiniteF64::new(1.0).unwrap(),
        };
        assert!(matches!(
            table.cases_where(count, &quantity),
            Err(EvaluationDatasetError::InapplicableRule { .. })
        ));
    }

    #[test]
    fn key_rules_must_name_entries_of_the_output_axis() {
        use graphcal_model_protocol::{KeyValue, VariantName};

        let schema = ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .key_output("verdict", "Verdict", ["pass", "fail"])
            .unwrap()
            .build()
            .unwrap();
        let x = schema.input_slot("x").unwrap();
        let evaluator = InProcessEvaluator::new(
            Arc::clone(&schema),
            move |row: InputRow<'_>, _: RowContext| {
                let verdict = if row.quantity(x)? > 0.5 {
                    "fail"
                } else {
                    "pass"
                };
                Ok(vec![Value::Key(KeyValue::Variant(
                    VariantName::new(verdict).unwrap(),
                ))])
            },
        );
        let request = crate::sample_uniform(&schema, 50, 3, 0).unwrap();
        let result = evaluator
            .evaluate(vec![request.clone()])
            .next()
            .unwrap()
            .unwrap();
        let table = table(&schema, [(request, result)]);
        let verdict = schema.output_slot("verdict").unwrap();
        let variant = |name| KeyValue::Variant(VariantName::new(name).unwrap());

        let failed = table
            .cases_where(
                verdict,
                &CaseRule::KeyIn {
                    keys: vec![variant("fail")],
                },
            )
            .unwrap();
        let x_values = table.features().unwrap()[0]
            .continuous_values()
            .unwrap()
            .to_vec();
        assert!(
            failed
                .iter()
                .zip(&x_values)
                .all(|(failed, x)| *failed == (*x > 0.5))
        );
        assert!(matches!(
            table.cases_where(
                verdict,
                &CaseRule::KeyIn {
                    keys: vec![variant("unknown")],
                },
            ),
            Err(EvaluationDatasetError::InapplicableRule { .. })
        ));
    }

    #[test]
    fn empty_tables_have_no_dataset() {
        let schema = schema();
        let failure = schema.output_slot("failure").unwrap();
        assert!(EvaluationTable::new(schema).dataset(failure).is_err());
    }

    #[test]
    fn datetime_inputs_are_continuous_features_and_complex_inputs_are_rejected() {
        let finite = |value| FiniteF64::new(value).unwrap();
        let datetime = ModelSchemaBuilder::new()
            .datetime(
                "launch",
                graphcal_model_protocol::DatetimeType::new(graphcal_model_protocol::TimeScale::Utc),
                graphcal_model_protocol::Bounds::new(Some(finite(0.0)), Some(finite(10.0)))
                    .unwrap(),
            )
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        assert!(
            input_feature(
                &datetime.inputs()[0],
                &InputColumn::Datetime(vec![finite(1.0), finite(2.0)])
            )
            .is_ok()
        );
        assert!(matches!(
            input_feature(
                &datetime.inputs()[0],
                &InputColumn::Complex(vec![graphcal_model_protocol::ComplexValue {
                    re: finite(1.0),
                    im: finite(0.0),
                }])
            ),
            Err(DataError::UnsupportedFeature {
                kind: "complex",
                ..
            })
        ));
    }
}
