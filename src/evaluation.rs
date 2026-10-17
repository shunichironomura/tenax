//! Transport-independent evaluation requests, streamed chunk results, and the
//! single-threaded in-process reference evaluator.

use std::any::Any;
use std::convert::Infallible;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use graphcal_model_protocol::{
    OutputSlot, PortName, RequestBatch, RowFailure as ProtocolRowFailure, SessionSchema, Value,
};
use thiserror::Error;

use crate::input::InputRow;
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

/// One requested batch of input rows with explicit reproducibility metadata.
///
/// The [`RequestBatch`] carries the session schema it was validated against,
/// so an evaluator can reject a request built for a different session.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalRequest {
    id: EvaluationId,
    seed: u64,
    inputs: RequestBatch,
}

impl EvalRequest {
    /// Constructs an evaluation request.
    #[must_use]
    pub const fn new(id: EvaluationId, seed: u64, inputs: RequestBatch) -> Self {
        Self { id, seed, inputs }
    }

    /// Returns the stable request identifier.
    #[must_use]
    pub const fn id(&self) -> EvaluationId {
        self.id
    }

    /// Returns the explicit model seed for this request.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the validated input batch.
    #[must_use]
    pub const fn inputs(&self) -> &RequestBatch {
        &self.inputs
    }

    /// Consumes the request and returns its input batch.
    #[must_use]
    pub fn into_inputs(self) -> RequestBatch {
        self.inputs
    }

    /// Checks that this request was validated against `schema`, by identity or
    /// deep equality.
    ///
    /// # Errors
    ///
    /// Returns [`SessionMismatch`] when the request belongs to another session.
    pub fn validate_against(&self, schema: &Arc<SessionSchema>) -> Result<(), SessionMismatch> {
        let requested = self.inputs.schema();
        if Arc::ptr_eq(requested, schema) || requested.as_ref() == schema.as_ref() {
            Ok(())
        } else {
            Err(SessionMismatch)
        }
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
    pub(crate) const fn new(
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

    /// Returns the seed attached to the complete request.
    #[must_use]
    pub const fn request_seed(&self) -> u64 {
        self.request_seed
    }

    /// Returns the row's zero-based position within the request.
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

/// Errors raised when a successful model row disagrees with its session's
/// outputs.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum OutputRowError {
    /// The number of values differs from the number of session outputs.
    #[error("model returned {actual} outputs, but its session reads {expected}")]
    CountMismatch {
        /// Number of session outputs.
        expected: usize,
        /// Number of returned values.
        actual: usize,
    },

    /// One value does not conform to its output's wire type.
    #[error("the value of output '{output}' does not conform to its wire type")]
    KindMismatch {
        /// The output whose value does not conform.
        output: PortName,
    },
}

/// A schema-validated set of values returned for one input row.
///
/// Each value conforms to the wire type of the session output at the same
/// position, including membership of a key variant in its key type.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputRow {
    values: Vec<Value>,
}

impl OutputRow {
    /// Validates one successful output row against `schema`'s outputs.
    ///
    /// # Errors
    ///
    /// Returns [`OutputRowError`] when the value count differs from the
    /// session's outputs or a value does not conform to its output.
    pub fn new(schema: &SessionSchema, values: Vec<Value>) -> Result<Self, OutputRowError> {
        validate_output_values(schema, &values)?;
        Ok(Self { values })
    }

    /// Returns values in session output order.
    #[must_use]
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Returns the value of the output at `slot`.
    #[must_use]
    pub fn value(&self, slot: OutputSlot) -> Option<&Value> {
        self.values.get(slot.get())
    }

    /// Revalidates this row against `schema`'s outputs.
    ///
    /// # Errors
    ///
    /// Returns [`OutputRowError`] when the row does not conform to `schema`.
    pub fn validate_against(&self, schema: &SessionSchema) -> Result<(), OutputRowError> {
        validate_output_values(schema, &self.values)
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

/// A request was validated against a different session schema than the
/// evaluator's.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("the request was built for a different session schema than the evaluator's")]
pub struct SessionMismatch;

/// Why one input row did not produce schema-valid outputs.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RowFailure {
    /// The model reported a failure for this row: an assertion that did not
    /// hold, an input outside its declared domain, or another evaluation
    /// error.
    #[error("model evaluation failed: {}", ModelFailureDisplay(.0))]
    Model(ProtocolRowFailure),

    /// The in-process model panicked while evaluating this row.
    #[error("model evaluation panicked: {0}")]
    Panic(#[source] ModelPanic),

    /// The request does not belong to this evaluator's session.
    #[error("invalid model input: {0}")]
    InvalidInput(#[source] SessionMismatch),

    /// The model returned outputs that do not satisfy its session's outputs.
    #[error("invalid model output: {0}")]
    InvalidOutput(#[source] OutputRowError),
}

struct ModelFailureDisplay<'a>(&'a ProtocolRowFailure);

impl fmt::Display for ModelFailureDisplay<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            ProtocolRowFailure::AssertionFailed { assertion, message } => {
                write!(formatter, "assertion `{assertion}` failed: {message}")
            }
            ProtocolRowFailure::InputOutOfDomain { parameter, message } => {
                write!(formatter, "input '{parameter}' is out of domain: {message}")
            }
            ProtocolRowFailure::EvaluationError { message } => formatter.write_str(message),
        }
    }
}

/// The result of evaluating one input row.
#[derive(Clone, Debug, PartialEq)]
pub enum RowOutcome {
    /// All session outputs were produced successfully.
    Success(OutputRow),
    /// This row failed without invalidating other rows in its request.
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
    pub(crate) fn from_rows(id: EvaluationId, rows: Vec<RowOutcome>) -> Self {
        debug_assert_ne!(rows.len(), 0);
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

    /// Consumes the result and returns its per-row outcomes.
    #[must_use]
    pub fn into_rows(self) -> Vec<RowOutcome> {
        self.rows
    }
}

/// A synchronous, transport-independent batch evaluator.
///
/// A call accepts multiple requests and yields each completed result via an
/// iterator. Implementations may return results out of request order. The
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

    /// Returns the session schema: the inputs every request supplies and the
    /// outputs every successful row returns.
    fn schema(&self) -> &Arc<SessionSchema>;

    /// Starts evaluating a batch and streams completed results or evaluator
    /// failures.
    fn evaluate(
        &self,
        requests: Vec<EvalRequest>,
    ) -> Box<dyn Iterator<Item = Result<ChunkResult, Self::Error>> + '_>;
}

/// A single-threaded evaluator backed by a row-at-a-time Rust closure.
///
/// The closure receives a zero-copy [`InputRow`] and deterministic
/// [`RowContext`], and returns either one value per session output, in
/// session order, or a [`graphcal_model_protocol::RowFailure`] that reports an
/// infeasible row with the same vocabulary as a Graphcal model.
///
/// Before invoking the model, the evaluator checks that each request was built
/// for its session schema ([`RowFailure::InvalidInput`] otherwise) and that
/// every row's quantity and integer inputs lie within their declared domains
/// ([`RowFailure::Model`] with
/// [`graphcal_model_protocol::RowFailure::InputOutOfDomain`] otherwise). An
/// unwinding model panic is retained as [`RowFailure::Panic`] and does not stop
/// later rows. Panics compiled with `panic = "abort"` cannot be caught.
pub struct InProcessEvaluator<F> {
    schema: Arc<SessionSchema>,
    model: F,
}

impl<F> InProcessEvaluator<F> {
    /// Constructs an in-process evaluator around a session schema and model.
    #[must_use]
    pub const fn new(schema: Arc<SessionSchema>, model: F) -> Self {
        Self { schema, model }
    }
}

impl<F> InProcessEvaluator<F>
where
    F: for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<Value>, ProtocolRowFailure>,
{
    fn evaluate_request(&self, request: &EvalRequest) -> ChunkResult {
        let id = request.id;
        let rows = match request.validate_against(&self.schema) {
            Ok(()) => InputRow::all(&request.inputs)
                .map(|row| {
                    let context = RowContext::new(id, request.seed, row.index());
                    evaluate_model_row(&self.schema, &self.model, row, context)
                })
                .collect(),
            Err(mismatch) => rejected_rows(request, mismatch),
        };
        ChunkResult::from_rows(id, rows)
    }
}

/// One [`RowFailure::InvalidInput`] outcome per row of `request`.
pub fn rejected_rows(request: &EvalRequest, mismatch: SessionMismatch) -> Vec<RowOutcome> {
    vec![RowOutcome::Failure(RowFailure::InvalidInput(mismatch)); request.inputs.rows()]
}

/// Evaluates one in-bounds row: checks input domains, invokes the model while
/// catching unwinding panics, and validates the returned outputs.
pub fn evaluate_model_row<F>(
    schema: &SessionSchema,
    model: &F,
    row: InputRow<'_>,
    context: RowContext,
) -> RowOutcome
where
    F: for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<Value>, ProtocolRowFailure>,
{
    if let Some(failure) = row.domain_violation() {
        return RowOutcome::Failure(RowFailure::Model(failure));
    }
    match catch_unwind(AssertUnwindSafe(|| model(row, context))) {
        Ok(Ok(values)) => match OutputRow::new(schema, values) {
            Ok(outputs) => RowOutcome::Success(outputs),
            Err(error) => RowOutcome::Failure(RowFailure::InvalidOutput(error)),
        },
        Ok(Err(failure)) => RowOutcome::Failure(RowFailure::Model(failure)),
        Err(payload) => RowOutcome::Failure(RowFailure::Panic(ModelPanic::from_payload(payload))),
    }
}

impl<F> Evaluator for InProcessEvaluator<F>
where
    F: for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<Value>, ProtocolRowFailure>,
{
    type Error = Infallible;

    fn schema(&self) -> &Arc<SessionSchema> {
        &self.schema
    }

    fn evaluate(
        &self,
        requests: Vec<EvalRequest>,
    ) -> Box<dyn Iterator<Item = Result<ChunkResult, Self::Error>> + '_> {
        Box::new(
            requests
                .into_iter()
                .map(|request| Ok(self.evaluate_request(&request))),
        )
    }
}

fn validate_output_values(schema: &SessionSchema, values: &[Value]) -> Result<(), OutputRowError> {
    let outputs = schema.outputs();
    if values.len() != outputs.len() {
        return Err(OutputRowError::CountMismatch {
            expected: outputs.len(),
            actual: values.len(),
        });
    }
    outputs
        .iter()
        .zip(values)
        .find(|(output, value)| !value.conforms_to(output.wire()))
        .map_or(Ok(()), |(output, _)| {
            Err(OutputRowError::KindMismatch {
                output: output.name().clone(),
            })
        })
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::InputColumn;

    use super::*;
    use crate::ModelSchemaBuilder;

    fn schema() -> Arc<SessionSchema> {
        schema_with_upper(2)
    }

    fn schema_with_upper(upper: i64) -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .integer("x", 0, upper)
            .unwrap()
            .boolean_output("is_even")
            .unwrap()
            .build()
            .unwrap()
    }

    fn request_for(schema: &Arc<SessionSchema>, id: u128, values: Vec<i64>) -> EvalRequest {
        let inputs = RequestBatch::new(Arc::clone(schema), vec![InputColumn::Int(values)]).unwrap();
        EvalRequest::new(EvaluationId::new(id), 17, inputs)
    }

    fn request(id: u128, values: Vec<i64>) -> EvalRequest {
        request_for(&schema(), id, values)
    }

    fn parity_model(
        schema: &SessionSchema,
    ) -> impl for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<Value>, ProtocolRowFailure> + use<>
    {
        let x = schema.input_slot("x").unwrap();
        move |row: InputRow<'_>, _| match row.int(x)? {
            1 => Err(ProtocolRowFailure::AssertionFailed {
                assertion: "x != 1".to_owned(),
                message: "singular region".to_owned(),
            }),
            value => Ok(vec![Value::Bool(value % 2 == 0)]),
        }
    }

    fn success(schema: &SessionSchema, value: bool) -> RowOutcome {
        RowOutcome::Success(OutputRow::new(schema, vec![Value::Bool(value)]).unwrap())
    }

    #[test]
    fn in_process_evaluator_preserves_chunk_and_row_order() {
        let schema = schema();
        let x = schema.input_slot("x").unwrap();
        let evaluator =
            InProcessEvaluator::new(Arc::clone(&schema), move |row: InputRow<'_>, _| {
                Ok(vec![Value::Bool(row.int(x)? % 2 == 0)])
            });
        let results = evaluator
            .evaluate(vec![request(10, vec![0, 1]), request(11, vec![2])])
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results[0].id(), EvaluationId::new(10));
        assert_eq!(results[1].id(), EvaluationId::new(11));
        assert_eq!(
            results[0].rows(),
            [success(&schema, true), success(&schema, false)]
        );
    }

    #[test]
    fn model_failures_are_per_row_data() {
        let schema = schema();
        let evaluator = InProcessEvaluator::new(Arc::clone(&schema), parity_model(&schema));
        let result = evaluator
            .evaluate(vec![request(10, vec![0, 1, 2])])
            .next()
            .unwrap()
            .unwrap();

        assert!(matches!(result.rows()[0], RowOutcome::Success(_)));
        assert!(matches!(
            &result.rows()[1],
            RowOutcome::Failure(RowFailure::Model(ProtocolRowFailure::AssertionFailed { message, .. }))
                if message == "singular region"
        ));
        assert!(matches!(result.rows()[2], RowOutcome::Success(_)));
    }

    #[test]
    fn model_panics_are_per_row_data() {
        let schema = schema();
        let x = schema.input_slot("x").unwrap();
        let evaluator =
            InProcessEvaluator::new(Arc::clone(&schema), move |row: InputRow<'_>, _| {
                match row.int(x)? {
                    1 => panic!("singular model state"),
                    value => Ok(vec![Value::Bool(value % 2 == 0)]),
                }
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
    fn evaluator_rejects_requests_of_another_session() {
        let evaluator = InProcessEvaluator::new(schema(), |_: InputRow<'_>, _| {
            panic!("a request of another session must not invoke the model")
        });
        let wider_schema = schema_with_upper(3);
        let result = evaluator
            .evaluate(vec![request_for(&wider_schema, 12, vec![3, 0])])
            .next()
            .unwrap()
            .unwrap();

        assert_eq!(
            result.rows(),
            [
                RowOutcome::Failure(RowFailure::InvalidInput(SessionMismatch)),
                RowOutcome::Failure(RowFailure::InvalidInput(SessionMismatch))
            ]
        );
    }

    #[test]
    fn equal_schemas_from_separate_builds_are_accepted() {
        let evaluator = InProcessEvaluator::new(schema(), parity_model(&schema()));
        let result = evaluator
            .evaluate(vec![request_for(&schema(), 12, vec![0])])
            .next()
            .unwrap()
            .unwrap();
        assert!(matches!(result.rows(), [RowOutcome::Success(_)]));
    }

    #[test]
    fn out_of_domain_rows_fail_without_invoking_the_model() {
        let schema = schema();
        let x = schema.input_slot("x").unwrap();
        let evaluator =
            InProcessEvaluator::new(Arc::clone(&schema), move |row: InputRow<'_>, _| {
                assert!(
                    row.int(x)? <= 2,
                    "out-of-domain rows must not reach the model"
                );
                Ok(vec![Value::Bool(true)])
            });
        let result = evaluator
            .evaluate(vec![request(14, vec![0, 3])])
            .next()
            .unwrap()
            .unwrap();
        assert!(matches!(result.rows()[0], RowOutcome::Success(_)));
        assert!(matches!(
            &result.rows()[1],
            RowOutcome::Failure(RowFailure::Model(ProtocolRowFailure::InputOutOfDomain { parameter, .. }))
                if parameter.as_str() == "x"
        ));
    }

    #[test]
    fn invalid_outputs_become_row_failures() {
        let evaluator = InProcessEvaluator::new(schema(), |row: InputRow<'_>, _| {
            Ok(match row.index() {
                0 => Vec::new(),
                _ => vec![Value::Int(1)],
            })
        });
        let result = evaluator
            .evaluate(vec![request(10, vec![0, 0])])
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(
            result.rows(),
            [
                RowOutcome::Failure(RowFailure::InvalidOutput(OutputRowError::CountMismatch {
                    expected: 1,
                    actual: 0
                })),
                RowOutcome::Failure(RowFailure::InvalidOutput(OutputRowError::KindMismatch {
                    output: PortName::new("is_even").unwrap()
                }))
            ]
        );
    }

    #[test]
    fn model_failures_render_their_protocol_diagnostic() {
        let failure = RowFailure::Model(ProtocolRowFailure::InputOutOfDomain {
            parameter: PortName::new("x").unwrap(),
            message: "value 3 lies outside [0, 2]".to_owned(),
        });
        assert_eq!(
            failure.to_string(),
            "model evaluation failed: input 'x' is out of domain: value 3 lies outside [0, 2]"
        );
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
