//! Rayon-backed execution for row-at-a-time in-process models.

use std::convert::Infallible;
use std::num::NonZeroUsize;
use std::sync::{Arc, mpsc};

use rayon::prelude::*;
use thiserror::Error;

use crate::evaluation::{
    ChunkResult, EvalRequest, Evaluator, ModelError, OutputValue, RowContext, RowFailure,
    RowOutcome, evaluate_model_row,
};
use crate::input::{InputChunk, InputRow};
use crate::schema::ModelSchema;

/// Errors raised while configuring parallel row chunking.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ChunkingPolicyError {
    /// A Rayon work chunk cannot contain zero rows.
    #[error("parallel evaluation requires at least one row per work chunk")]
    ZeroRowsPerChunk,
}

/// A fixed upper bound on rows evaluated sequentially in one Rayon work item.
///
/// A request is partitioned into consecutive chunks of this size, with a
/// possibly shorter final chunk. Chunks execute in parallel, while outcomes are
/// restored to input-row order before their [`ChunkResult`] is emitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkingPolicy {
    rows_per_chunk: NonZeroUsize,
}

impl ChunkingPolicy {
    /// Constructs a fixed-row chunking policy.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkingPolicyError::ZeroRowsPerChunk`] when
    /// `rows_per_chunk` is zero.
    pub fn new(rows_per_chunk: usize) -> Result<Self, ChunkingPolicyError> {
        NonZeroUsize::new(rows_per_chunk)
            .map(|rows_per_chunk| Self { rows_per_chunk })
            .ok_or(ChunkingPolicyError::ZeroRowsPerChunk)
    }

    /// Returns the maximum number of rows in one Rayon work item.
    #[must_use]
    pub const fn rows_per_chunk(self) -> usize {
        self.rows_per_chunk.get()
    }
}

enum EvaluationPool {
    Global,
    #[cfg(test)]
    Dedicated(Arc<rayon::ThreadPool>),
}

impl EvaluationPool {
    fn spawn<Operation>(&self, operation: Operation)
    where
        Operation: FnOnce() + Send + 'static,
    {
        match self {
            Self::Global => rayon::spawn(operation),
            #[cfg(test)]
            Self::Dedicated(pool) => pool.spawn(operation),
        }
    }
}

/// A Rayon-parallel evaluator backed by a row-at-a-time Rust closure.
///
/// Requests are scheduled independently on Rayon's global thread pool and may
/// therefore be yielded out of request order. Rows within each request are
/// partitioned according to [`ChunkingPolicy`] and evaluated in parallel. The
/// result always restores those rows to input order.
///
/// The model closure is shared across worker threads and must consequently be
/// `Send + Sync + 'static`. It receives a zero-copy [`InputRow`] and the same
/// deterministic [`RowContext`] it would receive from the sequential
/// [`crate::InProcessEvaluator`]. Returned errors, schema-invalid outputs, and
/// unwinding panics affect only their own rows. Panics compiled with
/// `panic = "abort"` cannot be caught.
///
/// Model calls have no defined invocation order. Reproducible models should
/// avoid order-dependent shared state and ambient randomness, deriving any
/// stochastic stream from [`RowContext::seed`] instead.
pub struct ParallelInProcessEvaluator<F> {
    schema: Arc<ModelSchema>,
    model: Arc<F>,
    chunking: ChunkingPolicy,
    pool: EvaluationPool,
}

impl<F> ParallelInProcessEvaluator<F> {
    /// Constructs a parallel evaluator using Rayon's global thread pool.
    #[must_use]
    pub fn new(schema: ModelSchema, model: F, chunking: ChunkingPolicy) -> Self {
        Self {
            schema: Arc::new(schema),
            model: Arc::new(model),
            chunking,
            pool: EvaluationPool::Global,
        }
    }

    #[cfg(test)]
    fn with_pool(
        schema: ModelSchema,
        model: F,
        chunking: ChunkingPolicy,
        pool: rayon::ThreadPool,
    ) -> Self {
        Self {
            schema: Arc::new(schema),
            model: Arc::new(model),
            chunking,
            pool: EvaluationPool::Dedicated(Arc::new(pool)),
        }
    }
}

impl<F> Evaluator for ParallelInProcessEvaluator<F>
where
    F: for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<OutputValue>, ModelError>
        + Send
        + Sync
        + 'static,
{
    type Error = Infallible;

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn evaluate(
        &self,
        requests: Vec<EvalRequest>,
    ) -> Box<dyn Iterator<Item = Result<ChunkResult, Self::Error>> + '_> {
        let (sender, receiver) = mpsc::channel();
        for request in requests {
            let schema = Arc::clone(&self.schema);
            let model = Arc::clone(&self.model);
            let sender = sender.clone();
            let chunking = self.chunking;
            self.pool.spawn(move || {
                let result = evaluate_request(&schema, model.as_ref(), request, chunking);
                drop(sender.send(result));
            });
        }
        drop(sender);
        Box::new(receiver.into_iter().map(Ok))
    }
}

fn evaluate_request<F>(
    schema: &ModelSchema,
    model: &F,
    request: EvalRequest,
    chunking: ChunkingPolicy,
) -> ChunkResult
where
    F: for<'data> Fn(InputRow<'data>, RowContext) -> Result<Vec<OutputValue>, ModelError>
        + Send
        + Sync,
{
    let id = request.id();
    let seed = request.seed();
    let inputs: InputChunk = request.into_inputs();
    let rows = match inputs.validate_against(schema) {
        Ok(()) => {
            // `fold_chunks` remains indexed, so collection restores chunk and
            // row order regardless of the workers' completion order.
            (0..inputs.row_count())
                .into_par_iter()
                .fold_chunks(
                    chunking.rows_per_chunk(),
                    Vec::new,
                    |mut outcomes, row_index| {
                        let row = inputs.row_in_bounds(row_index);
                        let context = RowContext::new(id, seed, row_index);
                        outcomes.push(evaluate_model_row(schema, model, row, context));
                        outcomes
                    },
                )
                .collect::<Vec<Vec<RowOutcome>>>()
                .into_iter()
                .flatten()
                .collect()
        }
        Err(error) => (0..inputs.row_count())
            .map(|_| RowOutcome::Failure(RowFailure::InvalidInput(error.clone())))
            .collect(),
    };
    ChunkResult::from_rows(id, rows)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    use rayon::ThreadPoolBuilder;

    use super::*;
    use crate::{
        EvaluationId, Feature, InputSchema, InputValue, ModelPanic, OutputRow, OutputSchema,
    };

    fn schema() -> ModelSchema {
        ModelSchema::new(
            vec![InputSchema::integer("x", 0, 3).unwrap()],
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

    fn two_thread_pool() -> rayon::ThreadPool {
        ThreadPoolBuilder::new().num_threads(2).build().unwrap()
    }

    #[test]
    fn chunking_policy_rejects_zero_rows() {
        assert_eq!(
            ChunkingPolicy::new(0),
            Err(ChunkingPolicyError::ZeroRowsPerChunk)
        );
        assert_eq!(ChunkingPolicy::new(7).unwrap().rows_per_chunk(), 7);
    }

    #[test]
    fn chunking_does_not_change_row_contexts_or_result_order() {
        let model = |_: InputRow<'_>, context: RowContext| {
            Ok(vec![OutputValue::Boolean(context.seed() & 1 == 0)])
        };
        let fine =
            ParallelInProcessEvaluator::new(schema(), model, ChunkingPolicy::new(1).unwrap());
        let coarse =
            ParallelInProcessEvaluator::new(schema(), model, ChunkingPolicy::new(3).unwrap());
        let request = request(9, vec![0, 1, 2, 3]);

        let fine_result = fine
            .evaluate(vec![request.clone()])
            .next()
            .unwrap()
            .unwrap();
        let coarse_result = coarse.evaluate(vec![request]).next().unwrap().unwrap();

        assert_eq!(fine_result, coarse_result);
    }

    #[test]
    fn completed_request_chunks_are_streamed_out_of_order() {
        let first_id = EvaluationId::new(10);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let model_gate = Arc::clone(&gate);
        let evaluator = ParallelInProcessEvaluator::with_pool(
            schema(),
            move |_: InputRow<'_>, context: RowContext| {
                if context.evaluation_id() == first_id {
                    let (lock, condition) = &*model_gate;
                    let (released, timeout) = condition
                        .wait_timeout_while(
                            lock.lock().unwrap(),
                            Duration::from_secs(5),
                            |released| !*released,
                        )
                        .unwrap();
                    let timed_out = timeout.timed_out() && !*released;
                    drop(released);
                    if timed_out {
                        return Err(ModelError::new("test release gate timed out"));
                    }
                }
                Ok(vec![OutputValue::Boolean(true)])
            },
            ChunkingPolicy::new(1).unwrap(),
            two_thread_pool(),
        );
        let mut results = evaluator.evaluate(vec![request(10, vec![0]), request(11, vec![1])]);

        let completed_first = results.next().unwrap().unwrap();
        let (lock, condition) = &*gate;
        *lock.lock().unwrap() = true;
        condition.notify_one();
        let completed_second = results.next().unwrap().unwrap();

        assert_eq!(completed_first.id(), EvaluationId::new(11));
        assert_eq!(completed_second.id(), EvaluationId::new(10));
    }

    #[test]
    fn erroring_and_panicking_rows_are_failure_data_without_reordering_rows() {
        let schema = schema();
        let x = schema.input_position("x").unwrap();
        let evaluator = ParallelInProcessEvaluator::with_pool(
            schema,
            move |row: InputRow<'_>, _| {
                let InputValue::Integer(value) = row
                    .value(x)
                    .map_err(|error| ModelError::new(error.to_string()))?
                else {
                    return Err(ModelError::new("x must be an integer"));
                };
                match value {
                    1 => Err(ModelError::new("model rejected row")),
                    2 => panic!("model panicked on row"),
                    _ => Ok(vec![OutputValue::Boolean(value % 2 == 0)]),
                }
            },
            ChunkingPolicy::new(1).unwrap(),
            two_thread_pool(),
        );
        let result = evaluator
            .evaluate(vec![request(20, vec![0, 1, 2, 3])])
            .next()
            .unwrap()
            .unwrap();

        assert_eq!(
            result.rows()[0],
            RowOutcome::Success(
                OutputRow::new(evaluator.schema(), vec![OutputValue::Boolean(true)]).unwrap()
            )
        );
        assert!(matches!(
            &result.rows()[1],
            RowOutcome::Failure(RowFailure::Model(error))
                if error.message() == "model rejected row"
        ));
        assert!(matches!(
            &result.rows()[2],
            RowOutcome::Failure(RowFailure::Panic(ModelPanic::Message(message)))
                if message == "model panicked on row"
        ));
        assert_eq!(
            result.rows()[3],
            RowOutcome::Success(
                OutputRow::new(evaluator.schema(), vec![OutputValue::Boolean(false)]).unwrap()
            )
        );
    }
}
