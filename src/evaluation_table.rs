//! Completed evaluations of one session, retained with every input and output.

use std::sync::Arc;

use graphcal_model_protocol::SessionSchema;
use thiserror::Error;

use crate::evaluation::{
    ChunkResult, EvalRequest, EvaluationId, OutputRowError, RowOutcome, SessionMismatch,
};

/// Errors raised when retaining a completed evaluation.
#[derive(Debug, Error, PartialEq)]
pub enum EvaluationTableError {
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

    /// A successful row does not match the session's outputs.
    #[error("row {row} of evaluation {id} has invalid outputs: {error}")]
    InvalidOutputRow {
        /// Evaluation request identifier.
        id: EvaluationId,
        /// Zero-based row position within the request.
        row: usize,
        /// Output schema violation.
        error: OutputRowError,
    },

    /// The table already holds an evaluation with this identifier.
    #[error("evaluation {id} is already in the table")]
    DuplicateId {
        /// The repeated identifier.
        id: EvaluationId,
    },
}

/// One evaluation request paired with the outcome of each of its rows.
///
/// Construction checks that the result answers the request: identifiers and
/// row counts agree, the request belongs to the session, and every successful
/// row conforms to the session's outputs. Failed rows are retained with their
/// failures.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedRequest {
    request: EvalRequest,
    rows: Vec<RowOutcome>,
}

impl EvaluatedRequest {
    /// Pairs `request` with its `result` under `schema`.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationTableError`] when the identifiers or row counts
    /// differ, the request belongs to another session, or a successful row
    /// violates `schema`'s outputs.
    pub fn new(
        schema: &Arc<SessionSchema>,
        request: EvalRequest,
        result: ChunkResult,
    ) -> Result<Self, EvaluationTableError> {
        if request.id() != result.id() {
            return Err(EvaluationTableError::MismatchedId {
                request_id: request.id(),
                result_id: result.id(),
            });
        }
        request.validate_against(schema)?;
        let id = request.id();
        let rows = result.into_rows();
        if request.inputs().rows() != rows.len() {
            return Err(EvaluationTableError::RowCountMismatch {
                id,
                expected: request.inputs().rows(),
                actual: rows.len(),
            });
        }
        for (row, outcome) in rows.iter().enumerate() {
            if let RowOutcome::Success(outputs) = outcome {
                outputs
                    .validate_against(schema)
                    .map_err(|error| EvaluationTableError::InvalidOutputRow { id, row, error })?;
            }
        }
        Ok(Self { request, rows })
    }

    /// Returns the evaluation identifier.
    #[must_use]
    pub const fn id(&self) -> EvaluationId {
        self.request.id()
    }

    /// Returns the evaluated request.
    #[must_use]
    pub const fn request(&self) -> &EvalRequest {
        &self.request
    }

    /// Returns the outcome of each request row, in request order.
    #[must_use]
    pub fn rows(&self) -> &[RowOutcome] {
        &self.rows
    }
}

/// Every completed evaluation of one session, in the order they were added.
///
/// Rows are ordered by evaluation, then by row within the request. The table
/// keeps failed rows; conversions that cannot use them, such as
/// [`EvaluationTable::dataset`], reject them explicitly.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationTable {
    schema: Arc<SessionSchema>,
    evaluations: Vec<EvaluatedRequest>,
}

impl EvaluationTable {
    /// Starts an empty table for `schema`'s evaluations.
    #[must_use]
    pub const fn new(schema: Arc<SessionSchema>) -> Self {
        Self {
            schema,
            evaluations: Vec::new(),
        }
    }

    /// Builds a table from evaluations, in order.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationTableError`] for the first evaluation that
    /// [`EvaluationTable::push_evaluated`] rejects.
    pub fn from_evaluations(
        schema: Arc<SessionSchema>,
        evaluations: impl IntoIterator<Item = EvaluatedRequest>,
    ) -> Result<Self, EvaluationTableError> {
        let mut table = Self::new(schema);
        for evaluation in evaluations {
            table.push_evaluated(evaluation)?;
        }
        Ok(table)
    }

    /// Validates `result` against `request` and appends the pair.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationTableError`] when [`EvaluatedRequest::new`] rejects
    /// the pair or the table already holds its identifier. The table is
    /// unchanged on error.
    pub fn push(
        &mut self,
        request: EvalRequest,
        result: ChunkResult,
    ) -> Result<(), EvaluationTableError> {
        let evaluation = EvaluatedRequest::new(&self.schema, request, result)?;
        self.push_evaluated(evaluation)
    }

    /// Appends an evaluation that was paired under this table's session.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationTableError`] when the evaluation belongs to another
    /// session, a successful row violates the session's outputs, or the table
    /// already holds its identifier. The table is unchanged on error.
    pub fn push_evaluated(
        &mut self,
        evaluation: EvaluatedRequest,
    ) -> Result<(), EvaluationTableError> {
        // Revalidate: the evaluation may have been paired under another
        // session schema.
        let EvaluatedRequest { request, rows } = evaluation;
        let id = request.id();
        let evaluation =
            EvaluatedRequest::new(&self.schema, request, ChunkResult::from_rows(id, rows))?;
        if self.evaluations.iter().any(|existing| existing.id() == id) {
            return Err(EvaluationTableError::DuplicateId { id });
        }
        self.evaluations.push(evaluation);
        Ok(())
    }

    /// Returns the session schema of every evaluation.
    #[must_use]
    pub const fn schema(&self) -> &Arc<SessionSchema> {
        &self.schema
    }

    /// Returns the evaluations in the order they were added.
    #[must_use]
    pub fn evaluations(&self) -> &[EvaluatedRequest] {
        &self.evaluations
    }

    /// Returns the number of rows across all evaluations.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.evaluations
            .iter()
            .map(|evaluation| evaluation.rows.len())
            .sum()
    }

    /// Returns whether the table holds no evaluation.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }

    /// Returns every row outcome in table order.
    pub fn outcomes(&self) -> impl Iterator<Item = &RowOutcome> {
        self.evaluations
            .iter()
            .flat_map(|evaluation| evaluation.rows.iter())
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{FiniteF64, InputColumn, RequestBatch, Value};

    use super::*;
    use crate::{ModelSchemaBuilder, OutputRow, RowFailure};

    fn schema() -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap()
    }

    fn request(schema: &Arc<SessionSchema>, id: u128, xs: &[f64]) -> EvalRequest {
        let inputs = RequestBatch::new(
            Arc::clone(schema),
            vec![InputColumn::Quantity(
                xs.iter().map(|x| FiniteF64::new(*x).unwrap()).collect(),
            )],
        )
        .unwrap();
        EvalRequest::new(EvaluationId::new(id), 7, inputs)
    }

    fn success(schema: &SessionSchema, value: bool) -> RowOutcome {
        RowOutcome::Success(OutputRow::new(schema, vec![Value::Bool(value)]).unwrap())
    }

    #[test]
    fn tables_keep_evaluations_and_failed_rows_in_order() {
        let schema = schema();
        let mut table = EvaluationTable::new(Arc::clone(&schema));
        assert!(table.is_empty());
        table
            .push(
                request(&schema, 1, &[0.1, 0.2]),
                ChunkResult::new(
                    EvaluationId::new(1),
                    vec![
                        success(&schema, true),
                        RowOutcome::Failure(RowFailure::InvalidInput(SessionMismatch)),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        table
            .push(
                request(&schema, 2, &[0.3]),
                ChunkResult::new(EvaluationId::new(2), vec![success(&schema, false)]).unwrap(),
            )
            .unwrap();
        assert_eq!(table.row_count(), 3);
        assert_eq!(
            table
                .evaluations()
                .iter()
                .map(EvaluatedRequest::id)
                .collect::<Vec<_>>(),
            [EvaluationId::new(1), EvaluationId::new(2)]
        );
        assert!(matches!(
            table.outcomes().nth(1),
            Some(RowOutcome::Failure(RowFailure::InvalidInput(_)))
        ));
    }

    #[test]
    fn tables_reject_mismatched_and_repeated_evaluations() {
        let schema = schema();
        let mut table = EvaluationTable::new(Arc::clone(&schema));
        let result = |id| ChunkResult::new(EvaluationId::new(id), vec![success(&schema, true)]);

        assert!(matches!(
            table.push(request(&schema, 1, &[0.1]), result(2).unwrap()),
            Err(EvaluationTableError::MismatchedId { .. })
        ));
        assert!(matches!(
            table.push(request(&schema, 1, &[0.1, 0.2]), result(1).unwrap()),
            Err(EvaluationTableError::RowCountMismatch {
                expected: 2,
                actual: 1,
                ..
            })
        ));
        table
            .push(request(&schema, 1, &[0.1]), result(1).unwrap())
            .unwrap();
        assert!(matches!(
            table.push(request(&schema, 1, &[0.1]), result(1).unwrap()),
            Err(EvaluationTableError::DuplicateId { .. })
        ));
        assert_eq!(table.row_count(), 1);

        let other = ModelSchemaBuilder::new()
            .continuous("x", 0.0, 2.0)
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        assert!(matches!(
            table.push(request(&other, 3, &[0.1]), result(3).unwrap()),
            Err(EvaluationTableError::InvalidInputs(_))
        ));
    }

    #[test]
    fn tables_revalidate_outputs_from_other_sessions() {
        let schema = schema();
        let other = ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .integer_output("count")
            .unwrap()
            .build()
            .unwrap();
        let foreign = RowOutcome::Success(OutputRow::new(&other, vec![Value::Int(1)]).unwrap());
        assert!(matches!(
            EvaluatedRequest::new(
                &schema,
                request(&schema, 1, &[0.1]),
                ChunkResult::new(EvaluationId::new(1), vec![foreign]).unwrap(),
            ),
            Err(EvaluationTableError::InvalidOutputRow { row: 0, .. })
        ));
    }
}
