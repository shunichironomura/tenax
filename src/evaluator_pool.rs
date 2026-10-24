//! Spreading evaluation requests over several evaluators of one session
//! schema, one worker thread per evaluator.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::thread::{self, JoinHandle};

use graphcal_model_protocol::SessionSchema;
use thiserror::Error;

use crate::evaluation::{ChunkResult, EvalRequest, EvaluationId, Evaluator, ModelPanic};

/// Errors raised while assembling an [`EvaluatorPool`].
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum EvaluatorPoolError {
    /// A pool needs at least one evaluator.
    #[error("an evaluator pool needs at least one evaluator")]
    NoEvaluators,

    /// An evaluator's session schema differs from the first evaluator's.
    #[error("evaluator {position} has a different session schema from evaluator 0")]
    SchemaMismatch {
        /// Zero-based position of the evaluator in the list given to the pool.
        position: usize,
    },
}

/// Why an [`EvaluatorPool`] produced no result for one request.
#[derive(Debug, Error)]
pub enum PoolEvaluationError<E> {
    /// The evaluator that took the request failed as a whole.
    #[error("the evaluator failed on request {id}")]
    Evaluator {
        /// The request's identifier.
        id: EvaluationId,
        /// The evaluator's failure.
        #[source]
        source: E,
    },

    /// The evaluator that took the request panicked.
    #[error("the evaluator panicked on request {id}: {panic}")]
    Panic {
        /// The request's identifier.
        id: EvaluationId,
        /// The panic's diagnostic.
        panic: ModelPanic,
    },

    /// The evaluator that took the request returned no result for it.
    #[error("the evaluator returned no result for request {id}")]
    MissingResult {
        /// The request's identifier.
        id: EvaluationId,
    },

    /// Every evaluator had stopped before the request was taken.
    #[error("request {id} was not evaluated: every evaluator of the pool has stopped")]
    NotEvaluated {
        /// The request's identifier.
        id: EvaluationId,
    },
}

impl<E> PoolEvaluationError<E> {
    /// Returns the identifier of the request that has no result.
    #[must_use]
    pub const fn id(&self) -> EvaluationId {
        match self {
            Self::Evaluator { id, .. }
            | Self::Panic { id, .. }
            | Self::MissingResult { id }
            | Self::NotEvaluated { id } => *id,
        }
    }
}

struct Member<E> {
    evaluator: E,
    stopped: AtomicBool,
}

/// Evaluates requests on several evaluators of one session schema at once.
///
/// Each evaluator gets a worker thread that takes the next waiting request as
/// soon as its previous one is done, so slow requests do not hold back the
/// others. The pool is itself an [`Evaluator`]: every request yields exactly
/// one item, in completion order, which may differ from request order. A
/// typical use runs several `graphcal model serve` sessions, each of which
/// evaluates one request at a time, on the chunks of one design split with
/// [`EvalRequest::split`].
///
/// An evaluator that fails as a whole, panics, or returns no result stops: the
/// request it took yields a [`PoolEvaluationError`], the remaining evaluators
/// take the waiting requests, and the pool does not use it again in later
/// calls. When every evaluator has stopped, each request still waiting yields
/// [`PoolEvaluationError::NotEvaluated`]. Results already yielded are
/// unaffected.
///
/// Dropping the result iterator early discards the requests not yet taken and
/// waits for the requests in progress, whose results are discarded.
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use tenax::graphcal_model_protocol::{RowFailure, Value};
/// use tenax::{
///     EvaluationTable, Evaluator, EvaluatorPool, InProcessEvaluator, InputRow,
///     ModelSchemaBuilder, RowContext, sample_latin_hypercube,
/// };
///
/// let schema = ModelSchemaBuilder::new()
///     .continuous("load", 0.0, 1.0)?
///     .boolean_output("failure")?
///     .build()?;
/// let load = schema.input_slot("load").expect("load is an input");
/// let model = move |row: InputRow<'_>, _: RowContext| -> Result<Vec<Value>, RowFailure> {
///     Ok(vec![Value::Bool(row.quantity(load)? >= 0.7)])
/// };
/// let pool = EvaluatorPool::new(
///     (0..4)
///         .map(|_| InProcessEvaluator::new(schema.clone(), model))
///         .collect(),
/// )?;
///
/// let design = sample_latin_hypercube(&schema, 1_000, 42, 0)?;
/// let chunks = design.split(NonZeroUsize::new(100).expect("non-zero"))?;
/// let mut table = EvaluationTable::new(schema.clone());
/// for result in pool.evaluate(chunks.clone()) {
///     let result = result?;
///     let request = chunks
///         .iter()
///         .find(|chunk| chunk.id() == result.id())
///         .expect("every result answers a chunk");
///     table.push(request.clone(), result)?;
/// }
/// table.sort_by_id();
///
/// assert_eq!(table.row_count(), 1_000);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct EvaluatorPool<E> {
    schema: Arc<SessionSchema>,
    members: Vec<Arc<Member<E>>>,
}

impl<E: Evaluator> EvaluatorPool<E> {
    /// Assembles a pool from evaluators that share one session schema, by
    /// identity or deep equality.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluatorPoolError::NoEvaluators`] for an empty list and
    /// [`EvaluatorPoolError::SchemaMismatch`] for the first evaluator whose
    /// schema differs from the first evaluator's.
    pub fn new(evaluators: Vec<E>) -> Result<Self, EvaluatorPoolError> {
        let schema = Arc::clone(
            evaluators
                .first()
                .ok_or(EvaluatorPoolError::NoEvaluators)?
                .schema(),
        );
        if let Some(position) = evaluators.iter().position(|evaluator| {
            let other = evaluator.schema();
            !(Arc::ptr_eq(other, &schema) || other.as_ref() == schema.as_ref())
        }) {
            return Err(EvaluatorPoolError::SchemaMismatch { position });
        }
        let members = evaluators
            .into_iter()
            .map(|evaluator| {
                Arc::new(Member {
                    evaluator,
                    stopped: AtomicBool::new(false),
                })
            })
            .collect();
        Ok(Self { schema, members })
    }

    /// Returns the number of evaluators, stopped ones included.
    #[must_use]
    pub const fn evaluator_count(&self) -> usize {
        self.members.len()
    }

    /// Returns the number of evaluators the pool still uses.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.members
            .iter()
            .filter(|member| !member.stopped.load(Ordering::Acquire))
            .count()
    }

    /// Returns the evaluators, stopped ones included, in the order given to
    /// [`EvaluatorPool::new`], for example to shut them down.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "worker threads hold members only while a result iterator, which borrows the pool, is alive, and its drop joins them"
    )]
    pub fn into_evaluators(self) -> Vec<E> {
        self.members
            .into_iter()
            .map(|member| {
                Arc::into_inner(member)
                    .expect("no worker thread outlives the result iterator")
                    .evaluator
            })
            .collect()
    }
}

type PoolItem<E> = Result<ChunkResult, PoolEvaluationError<E>>;

impl<E> Evaluator for EvaluatorPool<E>
where
    E: Evaluator + Send + Sync + 'static,
{
    type Error = PoolEvaluationError<E::Error>;

    fn schema(&self) -> &Arc<SessionSchema> {
        &self.schema
    }

    fn evaluate(
        &self,
        requests: Vec<EvalRequest>,
    ) -> Box<dyn Iterator<Item = Result<ChunkResult, Self::Error>> + '_> {
        let queue = Arc::new(Mutex::new(requests.into_iter().collect::<VecDeque<_>>()));
        let (sender, receiver) = mpsc::channel();
        let workers = self
            .members
            .iter()
            .filter(|member| !member.stopped.load(Ordering::Acquire))
            .map(|member| {
                let member = Arc::clone(member);
                let queue = Arc::clone(&queue);
                let sender = sender.clone();
                thread::spawn(move || work(&member, &queue, &sender))
            })
            .collect();
        drop(sender);
        Box::new(PoolResults {
            receiver,
            queue,
            workers,
        })
    }
}

/// Evaluates waiting requests until none is left, the result receiver is
/// gone, or the evaluator stops.
fn work<E: Evaluator>(
    member: &Member<E>,
    queue: &Mutex<VecDeque<EvalRequest>>,
    sender: &mpsc::Sender<PoolItem<E::Error>>,
) {
    while let Some(request) = take(queue) {
        let id = request.id();
        let item = match catch_unwind(AssertUnwindSafe(|| {
            member.evaluator.evaluate(vec![request]).next()
        })) {
            Ok(Some(Ok(result))) => Ok(result),
            Ok(Some(Err(source))) => Err(PoolEvaluationError::Evaluator { id, source }),
            Ok(None) => Err(PoolEvaluationError::MissingResult { id }),
            Err(payload) => Err(PoolEvaluationError::Panic {
                id,
                panic: ModelPanic::from_payload(payload),
            }),
        };
        let stop = item.is_err();
        if stop {
            member.stopped.store(true, Ordering::Release);
        }
        if sender.send(item).is_err() || stop {
            break;
        }
    }
}

fn take(queue: &Mutex<VecDeque<EvalRequest>>) -> Option<EvalRequest> {
    // A worker never panics while holding the lock, so a poisoned queue is
    // still consistent.
    queue
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .pop_front()
}

struct PoolResults<E> {
    receiver: mpsc::Receiver<PoolItem<E>>,
    queue: Arc<Mutex<VecDeque<EvalRequest>>>,
    workers: Vec<JoinHandle<()>>,
}

impl<E> Iterator for PoolResults<E> {
    type Item = PoolItem<E>;

    fn next(&mut self) -> Option<Self::Item> {
        // The channel closes once every worker has returned; requests still
        // waiting then have no evaluator left.
        self.receiver.recv().ok().or_else(|| {
            take(&self.queue)
                .map(|request| Err(PoolEvaluationError::NotEvaluated { id: request.id() }))
        })
    }
}

impl<E> Drop for PoolResults<E> {
    fn drop(&mut self) {
        self.queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        // Results sent meanwhile are buffered by the unbounded channel, so a
        // worker never blocks here. Worker panics are caught per request.
        for worker in self.workers.drain(..) {
            drop(worker.join());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;
    use std::sync::Condvar;
    use std::sync::atomic::AtomicUsize;

    use graphcal_model_protocol::{RowFailure as ProtocolRowFailure, Value};

    use super::*;
    use crate::{
        EvaluationTable, InProcessEvaluator, InputRow, ModelSchemaBuilder, RowContext,
        sample_latin_hypercube,
    };

    fn schema() -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .boolean_output("high")
            .unwrap()
            .build()
            .unwrap()
    }

    type Model = fn(InputRow<'_>, RowContext) -> Result<Vec<Value>, ProtocolRowFailure>;

    fn model(row: InputRow<'_>, _: RowContext) -> Result<Vec<Value>, ProtocolRowFailure> {
        let x = row.batch().schema().input_slot("x").unwrap();
        Ok(vec![Value::Bool(row.quantity(x)? >= 0.5)])
    }

    fn in_process(schema: &Arc<SessionSchema>) -> InProcessEvaluator<Model> {
        InProcessEvaluator::new(Arc::clone(schema), model as Model)
    }

    fn chunks(schema: &Arc<SessionSchema>, rows: usize, size: usize) -> Vec<EvalRequest> {
        sample_latin_hypercube(schema, rows, 7, 0)
            .unwrap()
            .split(NonZeroUsize::new(size).unwrap())
            .unwrap()
    }

    /// Fails every request after the first `succeed` it takes, then reports a
    /// closed session, like a Graphcal evaluator whose server has exited.
    #[derive(Debug, Error)]
    #[error("session closed")]
    struct Closed;

    struct Flaky {
        inner: InProcessEvaluator<Model>,
        succeed: usize,
        taken: AtomicUsize,
        panics: bool,
        failed: Arc<Signal>,
        waits_for: Option<Arc<Signal>>,
    }

    impl Flaky {
        /// Holds every request back until `signal` is set, so that another
        /// evaluator of the pool is sure to take requests first.
        fn waiting_for(self, signal: Arc<Signal>) -> Self {
            Self {
                waits_for: Some(signal),
                ..self
            }
        }
    }

    /// Set once, when a [`Flaky`] evaluator starts failing.
    #[derive(Default)]
    struct Signal {
        set: Mutex<bool>,
        changed: Condvar,
    }

    impl Signal {
        fn set(&self) {
            *self.set.lock().unwrap() = true;
            self.changed.notify_all();
        }

        fn wait(&self) {
            let set = self.set.lock().unwrap();
            drop(self.changed.wait_while(set, |set| !*set).unwrap());
        }
    }

    impl Evaluator for Flaky {
        type Error = Closed;

        fn schema(&self) -> &Arc<SessionSchema> {
            self.inner.schema()
        }

        fn evaluate(
            &self,
            requests: Vec<EvalRequest>,
        ) -> Box<dyn Iterator<Item = Result<ChunkResult, Closed>> + '_> {
            if let Some(signal) = &self.waits_for {
                signal.wait();
            }
            if self.taken.fetch_add(1, Ordering::SeqCst) < self.succeed {
                Box::new(
                    self.inner
                        .evaluate(requests)
                        .map(|result| result.map_err(|never| match never {})),
                )
            } else {
                self.failed.set();
                assert!(!self.panics, "evaluator crashed");
                Box::new(std::iter::once(Err(Closed)))
            }
        }
    }

    fn flaky(schema: &Arc<SessionSchema>, succeed: usize, panics: bool) -> Flaky {
        Flaky {
            inner: in_process(schema),
            succeed,
            taken: AtomicUsize::new(0),
            panics,
            failed: Arc::default(),
            waits_for: None,
        }
    }

    #[test]
    fn rejects_empty_and_mismatched_pools() {
        let other = ModelSchemaBuilder::new()
            .continuous("y", 0.0, 1.0)
            .unwrap()
            .boolean_output("high")
            .unwrap()
            .build()
            .unwrap();

        assert!(matches!(
            EvaluatorPool::<InProcessEvaluator<Model>>::new(Vec::new()),
            Err(EvaluatorPoolError::NoEvaluators)
        ));
        assert!(matches!(
            EvaluatorPool::new(vec![
                in_process(&schema()),
                in_process(&schema()),
                in_process(&other)
            ]),
            Err(EvaluatorPoolError::SchemaMismatch { position: 2 })
        ));
    }

    #[test]
    fn every_chunk_is_evaluated_once_and_matches_one_evaluator() {
        let schema = schema();
        let requests = chunks(&schema, 101, 10);
        let pool = EvaluatorPool::new((0..3).map(|_| in_process(&schema)).collect()).unwrap();

        let mut pooled = EvaluationTable::new(Arc::clone(&schema));
        for result in pool.evaluate(requests.clone()) {
            let result = result.unwrap();
            let request = requests
                .iter()
                .find(|request| request.id() == result.id())
                .unwrap();
            pooled.push(request.clone(), result).unwrap();
        }
        pooled.sort_by_id();

        let single = in_process(&schema);
        let mut sequential = EvaluationTable::new(Arc::clone(&schema));
        for (request, result) in requests.iter().zip(single.evaluate(requests.clone())) {
            sequential.push(request.clone(), result.unwrap()).unwrap();
        }
        assert_eq!(pooled, sequential);
        assert_eq!(pool.into_evaluators().len(), 3);
    }

    #[test]
    fn a_failing_evaluator_stops_and_the_others_finish() {
        let schema = schema();
        let requests = chunks(&schema, 20, 2);
        // The healthy evaluator waits until the failing one fails, so that the
        // failing one takes a second request however the threads are scheduled.
        let failing = flaky(&schema, 1, false);
        let healthy = flaky(&schema, usize::MAX, false).waiting_for(Arc::clone(&failing.failed));
        let pool = EvaluatorPool::new(vec![failing, healthy]).unwrap();

        let items = pool.evaluate(requests.clone()).collect::<Vec<_>>();

        assert_eq!(items.len(), requests.len());
        let failed = items
            .iter()
            .filter_map(|item| item.as_ref().err())
            .collect::<Vec<_>>();
        assert_eq!(failed.len(), 1);
        assert!(matches!(failed[0], PoolEvaluationError::Evaluator { .. }));
        let answered = items
            .iter()
            .map(|item| match item {
                Ok(result) => result.id(),
                Err(error) => error.id(),
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            answered,
            requests
                .iter()
                .map(EvalRequest::id)
                .collect::<BTreeSet<_>>()
        );
        assert_eq!(pool.running_count(), 1);

        // The stopped evaluator is no longer used.
        let later = pool.evaluate(chunks(&schema, 6, 2)).collect::<Vec<_>>();
        assert!(later.iter().all(Result::is_ok));
    }

    #[test]
    fn requests_left_when_every_evaluator_stops_are_not_evaluated() {
        let schema = schema();
        let requests = chunks(&schema, 10, 2);
        let pool = EvaluatorPool::new(vec![flaky(&schema, 1, true)]).unwrap();

        let items = pool.evaluate(requests.clone()).collect::<Vec<_>>();

        assert_eq!(items.len(), 5);
        assert!(items[0].is_ok());
        assert!(matches!(
            &items[1],
            Err(PoolEvaluationError::Panic { panic, .. })
                if panic.message() == Some("evaluator crashed")
        ));
        assert!(
            items[2..]
                .iter()
                .all(|item| matches!(item, Err(PoolEvaluationError::NotEvaluated { .. })))
        );
        assert_eq!(
            items
                .iter()
                .map(|item| match item {
                    Ok(result) => result.id(),
                    Err(error) => error.id(),
                })
                .collect::<Vec<_>>(),
            requests.iter().map(EvalRequest::id).collect::<Vec<_>>()
        );
        assert_eq!(pool.running_count(), 0);
        assert!(
            pool.evaluate(chunks(&schema, 2, 2))
                .all(|item| matches!(item, Err(PoolEvaluationError::NotEvaluated { .. })))
        );
    }

    #[test]
    fn dropping_the_results_early_releases_the_evaluators() {
        let schema = schema();
        let pool = EvaluatorPool::new((0..2).map(|_| in_process(&schema)).collect()).unwrap();
        let mut results = pool.evaluate(chunks(&schema, 100, 1));
        assert!(results.next().unwrap().is_ok());
        drop(results);

        assert_eq!(pool.into_evaluators().len(), 2);
    }
}
