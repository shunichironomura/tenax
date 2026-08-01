//! Transport-independent evaluation requests, streamed chunk results, and the
//! single-threaded in-process reference evaluator.

use std::any::Any;
use std::convert::Infallible;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};

use thiserror::Error;

use crate::DataError;
use crate::data::Dataset;
use crate::input::{InputChunk, InputChunkError, InputRow};
use crate::schema::{ModelSchema, OutputKind, OutputPosition};
use crate::seed::derive_seed;

const ROW_SEED_DOMAIN: u64 = 0x7465_6e61_785f_726f;

/// A stable identifier for one evaluation request.
///
/// IDs created with [`EvaluationId::from_run_seed`] combine the complete run
/// seed and request sequence without hashing, so they are collision-free within
/// and across distinct seeded runs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvaluationId(u128);

impl EvaluationId {
    /// Constructs an identifier from an externally assigned value.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Deterministically identifies `sequence` within a seeded run.
    #[must_use]
    pub fn from_run_seed(run_seed: u64, sequence: u64) -> Self {
        Self((u128::from(run_seed) << 64) | u128::from(sequence))
    }

    /// Returns the underlying boundary representation.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }
}

impl fmt::Display for EvaluationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:032x}", self.0)
    }
}

/// One requested input chunk with explicit reproducibility metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalRequest {
    id: EvaluationId,
    seed: u64,
    inputs: InputChunk,
}

impl EvalRequest {
    /// Constructs an evaluation request.
    #[must_use]
    pub const fn new(id: EvaluationId, seed: u64, inputs: InputChunk) -> Self {
        Self { id, seed, inputs }
    }

    /// Returns the stable request identifier.
    #[must_use]
    pub const fn id(&self) -> EvaluationId {
        self.id
    }

    /// Returns the explicit model seed for this request chunk.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the validated input chunk.
    #[must_use]
    pub const fn inputs(&self) -> &InputChunk {
        &self.inputs
    }

    /// Consumes the request and returns its input chunk.
    #[must_use]
    pub fn into_inputs(self) -> InputChunk {
        self.inputs
    }
}

/// Reproducibility metadata supplied to an in-process model row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowContext {
    evaluation_id: EvaluationId,
    request_seed: u64,
    row_index: usize,
}

impl RowContext {
    pub(super) const fn new(
        evaluation_id: EvaluationId,
        request_seed: u64,
        row_index: usize,
    ) -> Self {
        Self {
            evaluation_id,
            request_seed,
            row_index,
        }
    }

    /// Returns the enclosing request identifier.
    #[must_use]
    pub const fn evaluation_id(&self) -> EvaluationId {
        self.evaluation_id
    }

    /// Returns the seed attached to the complete request chunk.
    #[must_use]
    pub const fn request_seed(&self) -> u64 {
        self.request_seed
    }

    /// Returns the row's zero-based position within the request chunk.
    #[must_use]
    pub const fn row_index(&self) -> usize {
        self.row_index
    }

    /// Derives a deterministic seed unique to this row position.
    ///
    /// A row-at-a-time stochastic model should use this seed rather than ambient
    /// randomness. Vectorized evaluator implementations may instead consume the
    /// request seed directly with an explicitly documented stream policy.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "Rust's supported targets never have a `usize` wider than `u64`"
    )]
    pub fn seed(&self) -> u64 {
        let row_index = u64::try_from(self.row_index)
            .expect("supported Rust targets represent every usize as u64");
        derive_seed(self.request_seed, row_index, ROW_SEED_DOMAIN)
    }
}

/// A typed model output value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutputValue {
    /// A binary classification output.
    Boolean(bool),
}

impl OutputValue {
    /// Returns this value's schema kind.
    #[must_use]
    pub const fn kind(&self) -> OutputKind {
        match self {
            Self::Boolean(_) => OutputKind::Boolean,
        }
    }
}

/// Errors raised when a successful model row disagrees with its output schema.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum OutputRowError {
    /// The number of values differs from the number of declared outputs.
    #[error("model returned {actual} outputs, but its schema declares {expected}")]
    CountMismatch {
        /// Number of declared outputs.
        expected: usize,
        /// Number of returned values.
        actual: usize,
    },

    /// One output value has the wrong type.
    #[error("model output {position} has kind {actual:?}, but its schema requires {expected:?}")]
    KindMismatch {
        /// Zero-based output position.
        position: usize,
        /// Declared output type.
        expected: OutputKind,
        /// Returned output type.
        actual: OutputKind,
    },
}

/// A schema-validated set of values returned for one input row.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputRow {
    values: Vec<OutputValue>,
}

impl OutputRow {
    /// Validates one successful output row.
    ///
    /// # Errors
    ///
    /// Returns [`OutputRowError`] when value count or kinds differ from
    /// `schema`.
    pub fn new(schema: &ModelSchema, values: Vec<OutputValue>) -> Result<Self, OutputRowError> {
        validate_output_values(schema, &values)?;
        Ok(Self { values })
    }

    /// Returns values in model-schema order.
    #[must_use]
    pub fn values(&self) -> &[OutputValue] {
        &self.values
    }

    pub(crate) fn validate_against(&self, schema: &ModelSchema) -> Result<(), OutputRowError> {
        validate_output_values(schema, &self.values)
    }
}

/// A model-supplied explanation for one failed row evaluation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct ModelError {
    message: String,
}

impl ModelError {
    /// Constructs a diagnostic model failure.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the model-supplied diagnostic.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<String> for ModelError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for ModelError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

/// A panic raised while invoking an in-process model row.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ModelPanic {
    /// The panic carried a string diagnostic.
    #[error("{0}")]
    Message(String),

    /// The panic carried an application-specific non-string payload.
    #[error("non-string panic payload")]
    NonStringPayload,
}

impl ModelPanic {
    /// Returns the panic diagnostic when its payload was a string.
    #[must_use]
    pub fn message(&self) -> Option<&str> {
        match self {
            Self::Message(message) => Some(message),
            Self::NonStringPayload => None,
        }
    }

    fn from_payload(payload: Box<dyn Any + Send>) -> Self {
        match payload.downcast::<String>() {
            Ok(message) => Self::Message(*message),
            Err(payload) => payload
                .downcast::<&'static str>()
                .map_or(Self::NonStringPayload, |message| {
                    Self::Message((*message).to_owned())
                }),
        }
    }
}

/// Broad category of a row failure reported across an evaluator boundary.
///
/// Local evaluators retain their concrete [`InputChunkError`] and
/// [`OutputRowError`] values. A process or network peer cannot reconstruct
/// those Rust implementation types from a stable wire diagnostic, so it uses
/// this closed boundary category instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluatorFailureKind {
    /// The peer rejected the row's inputs.
    InvalidInput,
    /// The peer produced values that did not satisfy its declared outputs.
    InvalidOutput,
}

impl fmt::Display for EvaluatorFailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput => formatter.write_str("invalid input"),
            Self::InvalidOutput => formatter.write_str("invalid output"),
        }
    }
}

/// A structured row failure reported by an evaluator adapter.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("evaluator reported {kind}: {message}")]
pub struct EvaluatorFailure {
    kind: EvaluatorFailureKind,
    message: String,
}

impl EvaluatorFailure {
    /// Constructs a boundary-reported row failure.
    #[must_use]
    pub fn new(kind: EvaluatorFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Returns the transport-independent failure category.
    #[must_use]
    pub const fn kind(&self) -> EvaluatorFailureKind {
        self.kind
    }

    /// Returns the peer-supplied diagnostic.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Why one input row did not produce schema-valid outputs.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RowFailure {
    /// The model rejected or returned an error for this row.
    #[error("model evaluation failed: {0}")]
    Model(#[source] ModelError),

    /// The in-process model panicked while evaluating this row.
    #[error("model evaluation panicked: {0}")]
    Panic(#[source] ModelPanic),

    /// The request chunk does not satisfy this evaluator's schema.
    #[error("invalid model input: {0}")]
    InvalidInput(#[source] InputChunkError),

    /// The model returned outputs that do not satisfy its schema.
    #[error("invalid model output: {0}")]
    InvalidOutput(#[source] OutputRowError),

    /// A process or network adapter reported a typed boundary failure.
    #[error(transparent)]
    Evaluator(#[from] EvaluatorFailure),
}

/// The result of evaluating one input row.
#[derive(Clone, Debug, PartialEq)]
pub enum RowOutcome {
    /// All declared model outputs were produced successfully.
    Success(OutputRow),
    /// This row failed without invalidating other rows in its chunk.
    Failure(RowFailure),
}

/// Errors raised while constructing a result chunk.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ChunkResultError {
    /// A result chunk has no row outcomes.
    #[error("an evaluation result chunk must contain at least one row outcome")]
    NoRows,
}

/// Completed per-row outcomes for one evaluation request.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkResult {
    id: EvaluationId,
    rows: Vec<RowOutcome>,
}

impl ChunkResult {
    pub(super) fn from_rows(id: EvaluationId, rows: Vec<RowOutcome>) -> Self {
        debug_assert!(!rows.is_empty());
        Self { id, rows }
    }

    /// Constructs a non-empty result chunk for custom evaluator implementations.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkResultError::NoRows`] when `rows` is empty.
    pub fn new(id: EvaluationId, rows: Vec<RowOutcome>) -> Result<Self, ChunkResultError> {
        if rows.is_empty() {
            Err(ChunkResultError::NoRows)
        } else {
            Ok(Self { id, rows })
        }
    }

    /// Returns the request identifier echoed by the evaluator.
    #[must_use]
    pub const fn id(&self) -> EvaluationId {
        self.id
    }

    /// Returns per-row outcomes in original input-row order.
    #[must_use]
    pub fn rows(&self) -> &[RowOutcome] {
        &self.rows
    }
}

/// A synchronous, transport-independent batch evaluator.
///
/// A call accepts multiple request chunks and yields each completed chunk via
/// an iterator. Implementations may return chunks out of request order. The
/// [`InProcessEvaluator`] preserves request order, while
/// [`crate::ParallelInProcessEvaluator`] uses a channel to stream completion
/// order without adding an async runtime to the core.
///
/// Per-row model failures remain inside [`ChunkResult`]. The iterator's
/// [`Result`] is reserved for failures of the evaluator as a whole, such as a
/// broken process or malformed wire message. Keeping those failure scopes
/// distinct lets future drivers retry a stable [`EvaluationId`] without
/// mistaking a transport outage for model behavior.
pub trait Evaluator {
    /// Failure raised by the evaluator shell rather than an individual row.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the model's validated discovery schema.
    fn schema(&self) -> &ModelSchema;

    /// Starts evaluating a batch and streams completed chunks or evaluator
    /// failures.
    fn evaluate(
        &self,
        requests: Vec<EvalRequest>,
    ) -> Box<dyn Iterator<Item = Result<ChunkResult, Self::Error>> + '_>;
}

/// A single-threaded evaluator backed by a row-at-a-time Rust closure.
///
/// The closure receives a zero-copy [`InputRow`] and deterministic
/// [`RowContext`], and returns either all output values or a [`ModelError`]. An
/// unwinding model panic is retained as [`RowFailure::Panic`] and does not stop
/// later rows. Panics compiled with `panic = "abort"` cannot be caught.
pub struct InProcessEvaluator<F> {
    schema: ModelSchema,
    model: F,
}

impl<F> InProcessEvaluator<F> {
    /// Constructs an in-process evaluator around a validated schema and model.
    #[must_use]
    pub const fn new(schema: ModelSchema, model: F) -> Self {
        Self { schema, model }
    }
}

impl<F> InProcessEvaluator<F>
where
    F: for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<OutputValue>, ModelError>,
{
    fn evaluate_request(&self, request: EvalRequest) -> ChunkResult {
        let id = request.id;
        let seed = request.seed;
        let inputs = request.inputs;
        let rows = match inputs.validate_against(&self.schema) {
            Ok(()) => inputs
                .rows()
                .map(|row| {
                    let context = RowContext::new(id, seed, row.index());
                    evaluate_model_row(&self.schema, &self.model, row, context)
                })
                .collect(),
            Err(error) => (0..inputs.row_count())
                .map(|_| RowOutcome::Failure(RowFailure::InvalidInput(error.clone())))
                .collect(),
        };
        ChunkResult::from_rows(id, rows)
    }
}

pub fn evaluate_model_row<F>(
    schema: &ModelSchema,
    model: &F,
    row: InputRow<'_>,
    context: RowContext,
) -> RowOutcome
where
    F: for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<OutputValue>, ModelError>,
{
    match catch_unwind(AssertUnwindSafe(|| model(row, context))) {
        Ok(Ok(values)) => match OutputRow::new(schema, values) {
            Ok(outputs) => RowOutcome::Success(outputs),
            Err(error) => RowOutcome::Failure(RowFailure::InvalidOutput(error)),
        },
        Ok(Err(error)) => RowOutcome::Failure(RowFailure::Model(error)),
        Err(payload) => RowOutcome::Failure(RowFailure::Panic(ModelPanic::from_payload(payload))),
    }
}

impl<F> Evaluator for InProcessEvaluator<F>
where
    F: for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<OutputValue>, ModelError>,
{
    type Error = Infallible;

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn evaluate(
        &self,
        requests: Vec<EvalRequest>,
    ) -> Box<dyn Iterator<Item = Result<ChunkResult, Self::Error>> + '_> {
        Box::new(
            requests
                .into_iter()
                .map(|request| Ok(self.evaluate_request(request))),
        )
    }
}

/// Errors raised when converting one completed binary-output chunk to PRIM
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

    /// The chosen output does not exist.
    #[error("output index {index} is out of bounds for a schema with {output_count} outputs")]
    OutputIndexOutOfBounds {
        /// Requested output position.
        index: usize,
        /// Number of schema outputs.
        output_count: usize,
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

    /// A supposedly successful row does not match this model schema.
    #[error("evaluation row {row} has invalid outputs: {error}")]
    InvalidOutputRow {
        /// Zero-based row position.
        row: usize,
        /// Output schema violation.
        error: OutputRowError,
    },

    /// Request inputs do not satisfy the supplied model schema.
    #[error(transparent)]
    InvalidInputs(#[from] InputChunkError),

    /// The resulting static dataset violates a dataset invariant.
    #[error(transparent)]
    InvalidDataset(#[from] DataError),
}

/// Converts one successful binary-output evaluation chunk into a static PRIM
/// dataset.
///
/// This function deliberately rejects the first failed row rather than silently
/// dropping it. A driver that wants a different missing-evaluation policy must
/// make that policy explicit before constructing a [`Dataset`].
///
/// # Errors
///
/// Returns [`EvaluationDatasetError`] when request/result identity or row count
/// differs, inputs or outputs violate `schema`, `output_position` is invalid,
/// or any row failed.
pub fn evaluation_to_dataset(
    schema: &ModelSchema,
    request: EvalRequest,
    result: ChunkResult,
    output_position: OutputPosition,
) -> Result<Dataset, EvaluationDatasetError> {
    if request.id != result.id {
        return Err(EvaluationDatasetError::MismatchedId {
            request_id: request.id,
            result_id: result.id,
        });
    }
    request.inputs.validate_against(schema)?;
    if output_position.index() >= schema.outputs().len() {
        return Err(EvaluationDatasetError::OutputIndexOutOfBounds {
            index: output_position.index(),
            output_count: schema.outputs().len(),
        });
    }
    if request.inputs.row_count() != result.rows.len() {
        return Err(EvaluationDatasetError::RowCountMismatch {
            id: request.id,
            expected: request.inputs.row_count(),
            actual: result.rows.len(),
        });
    }

    let cases_of_interest = result
        .rows
        .into_iter()
        .enumerate()
        .map(|(row, outcome)| match outcome {
            RowOutcome::Success(outputs) => {
                outputs
                    .validate_against(schema)
                    .map_err(|error| EvaluationDatasetError::InvalidOutputRow { row, error })?;
                match outputs.values[output_position.index()] {
                    OutputValue::Boolean(value) => Ok(value),
                }
            }
            RowOutcome::Failure(failure) => Err(EvaluationDatasetError::FailedRow { row, failure }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    Dataset::new(request.inputs.into_features(), cases_of_interest).map_err(Into::into)
}

fn validate_output_values(
    schema: &ModelSchema,
    values: &[OutputValue],
) -> Result<(), OutputRowError> {
    if values.len() != schema.outputs().len() {
        return Err(OutputRowError::CountMismatch {
            expected: schema.outputs().len(),
            actual: values.len(),
        });
    }
    match schema
        .outputs()
        .iter()
        .zip(values)
        .enumerate()
        .find(|(_, (output, value))| output.kind() != value.kind())
    {
        Some((position, (output, value))) => Err(OutputRowError::KindMismatch {
            position,
            expected: output.kind(),
            actual: value.kind(),
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Feature, InputSchema, InputValue, OutputSchema};

    fn schema() -> ModelSchema {
        ModelSchema::new(
            vec![InputSchema::integer("x", 0, 2).unwrap()],
            vec![OutputSchema::boolean("is_even").unwrap()],
        )
        .unwrap()
    }

    fn request(id: u128, values: Vec<i64>) -> EvalRequest {
        let schema = schema();
        let inputs =
            InputChunk::new(&schema, vec![Feature::integer("x", values).unwrap()]).unwrap();
        EvalRequest::new(EvaluationId::new(id), 17, inputs)
    }

    #[test]
    fn in_process_evaluator_preserves_chunk_and_row_order() {
        let schema = schema();
        let x = schema.input_position("x").unwrap();
        let evaluator = InProcessEvaluator::new(schema, move |row: InputRow<'_>, _| {
            let InputValue::Integer(value) = row
                .value(x)
                .map_err(|error| ModelError::new(error.to_string()))?
            else {
                return Err(ModelError::new("x must be an integer"));
            };
            Ok(vec![OutputValue::Boolean(value % 2 == 0)])
        });
        let results = evaluator
            .evaluate(vec![request(10, vec![0, 1]), request(11, vec![2])])
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results[0].id(), EvaluationId::new(10));
        assert_eq!(results[1].id(), EvaluationId::new(11));
        assert_eq!(
            results[0].rows(),
            [
                RowOutcome::Success(
                    OutputRow::new(evaluator.schema(), vec![OutputValue::Boolean(true)]).unwrap()
                ),
                RowOutcome::Success(
                    OutputRow::new(evaluator.schema(), vec![OutputValue::Boolean(false)]).unwrap()
                )
            ]
        );
    }

    #[test]
    fn model_failures_are_per_row_data() {
        let schema = schema();
        let x = schema.input_position("x").unwrap();
        let evaluator =
            InProcessEvaluator::new(schema, move |row: InputRow<'_>, _| match row.value(x) {
                Ok(InputValue::Integer(1)) => Err(ModelError::new("singular region")),
                Ok(InputValue::Integer(value)) => Ok(vec![OutputValue::Boolean(value % 2 == 0)]),
                Ok(_) => Err(ModelError::new("wrong input kind")),
                Err(error) => Err(ModelError::new(error.to_string())),
            });
        let result = evaluator
            .evaluate(vec![request(10, vec![0, 1, 2])])
            .next()
            .unwrap()
            .unwrap();

        assert!(matches!(result.rows()[0], RowOutcome::Success(_)));
        assert!(matches!(
            result.rows()[1],
            RowOutcome::Failure(RowFailure::Model(_))
        ));
        assert!(matches!(result.rows()[2], RowOutcome::Success(_)));
    }

    #[test]
    fn model_panics_are_per_row_data() {
        let schema = schema();
        let x = schema.input_position("x").unwrap();
        let evaluator =
            InProcessEvaluator::new(schema, move |row: InputRow<'_>, _| match row.value(x) {
                Ok(InputValue::Integer(1)) => panic!("singular model state"),
                Ok(InputValue::Integer(value)) => Ok(vec![OutputValue::Boolean(value % 2 == 0)]),
                Ok(_) => Err(ModelError::new("wrong input kind")),
                Err(error) => Err(ModelError::new(error.to_string())),
            });
        let result = evaluator
            .evaluate(vec![request(10, vec![0, 1, 2])])
            .next()
            .unwrap()
            .unwrap();

        assert!(matches!(result.rows()[0], RowOutcome::Success(_)));
        assert!(matches!(
            &result.rows()[1],
            RowOutcome::Failure(RowFailure::Panic(ModelPanic::Message(message)))
                if message == "singular model state"
        ));
        assert!(matches!(result.rows()[2], RowOutcome::Success(_)));
    }

    #[test]
    fn evaluator_revalidates_requests_against_its_own_schema() {
        let evaluator = InProcessEvaluator::new(schema(), |_: InputRow<'_>, _| {
            panic!("a schema-invalid request must not invoke the model")
        });
        let wider_schema = ModelSchema::new(
            vec![InputSchema::integer("x", 0, 3).unwrap()],
            vec![OutputSchema::boolean("is_even").unwrap()],
        )
        .unwrap();
        let inputs =
            InputChunk::new(&wider_schema, vec![Feature::integer("x", vec![3]).unwrap()]).unwrap();
        let result = evaluator
            .evaluate(vec![EvalRequest::new(EvaluationId::new(12), 17, inputs)])
            .next()
            .unwrap()
            .unwrap();

        assert!(matches!(
            result.rows(),
            [RowOutcome::Failure(RowFailure::InvalidInput(
                InputChunkError::IntegerOutOfBounds { row: 0, .. }
            ))]
        ));
    }

    #[test]
    fn conversion_rejects_failed_rows_instead_of_dropping_them() {
        let schema = schema();
        let x = schema.input_position("x").unwrap();
        let output = schema.output_position("is_even").unwrap();
        let evaluator = InProcessEvaluator::new(schema.clone(), move |row: InputRow<'_>, _| {
            match row.value(x) {
                Ok(InputValue::Integer(1)) => Err(ModelError::new("singular region")),
                Ok(InputValue::Integer(value)) => Ok(vec![OutputValue::Boolean(value % 2 == 0)]),
                Ok(_) => Err(ModelError::new("wrong input kind")),
                Err(error) => Err(ModelError::new(error.to_string())),
            }
        });
        let request = request(13, vec![0, 1]);
        let result = evaluator
            .evaluate(vec![request.clone()])
            .next()
            .unwrap()
            .unwrap();

        assert!(matches!(
            evaluation_to_dataset(&schema, request, result, output),
            Err(EvaluationDatasetError::FailedRow {
                row: 1,
                failure: RowFailure::Model(_)
            })
        ));
    }

    #[test]
    fn invalid_output_count_becomes_a_row_failure() {
        let evaluator = InProcessEvaluator::new(schema(), |_: InputRow<'_>, _| Ok(Vec::new()));
        let result = evaluator
            .evaluate(vec![request(10, vec![0])])
            .next()
            .unwrap()
            .unwrap();
        assert!(matches!(
            result.rows(),
            [RowOutcome::Failure(RowFailure::InvalidOutput(
                OutputRowError::CountMismatch {
                    expected: 1,
                    actual: 0
                }
            ))]
        ));
    }

    #[test]
    fn row_seeds_are_stable_and_distinct() {
        let first = RowContext {
            evaluation_id: EvaluationId::new(1),
            request_seed: 9,
            row_index: 0,
        };
        let second = RowContext {
            row_index: 1,
            ..first
        };
        assert_eq!(first.seed(), first.seed());
        assert_ne!(first.seed(), second.seed());
    }
}
