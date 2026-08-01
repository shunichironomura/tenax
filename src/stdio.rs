//! Persistent subprocess evaluation over standard Arrow IPC streams on stdio.
//!
//! The child process is a model server. Its stdout contains two consecutive
//! Arrow IPC streams: a schema-only discovery stream, then a long-lived result
//! stream. Its stdin contains one long-lived request stream. The child must
//! finish and flush discovery, then write and flush the result-stream schema,
//! before reading the request-stream schema. This startup order avoids a
//! bidirectional pipe deadlock.
//!
//! Request IDs and seeds are physical Arrow columns, so every request batch has
//! one fixed stream schema. Result batches similarly carry IDs and per-row
//! status columns. A child may evaluate requests concurrently and emit one
//! result batch per request in completion order. Tenax validates IDs, row
//! counts, schemas, nulls, and status/output consistency before yielding a
//! [`ChunkResult`].
//!
//! See [`crate::arrow`] for the batch mapping and
//! `docs/stdio-arrow-ipc.md` in the repository for the language-neutral
//! protocol specification intended for Graphcal, Python, and other servers.

use std::collections::BTreeMap;
use std::io;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::thread::{self, JoinHandle};

use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::ArrowError as ArrowRsError;
use thiserror::Error;

use crate::arrow::{
    ArrowConversionError, ArrowSchema, ChunkResultRef, EvalRequestRef, RecordBatch,
    evaluation_request_schema, evaluation_result_schema, validate_evaluation_request_schema,
    validate_evaluation_result_schema,
};
use crate::{ChunkResult, EvalRequest, EvaluationId, Evaluator, InputChunkError, ModelSchema};

/// Discovery-schema metadata key selecting the stdio stream protocol version.
pub const STDIO_PROTOCOL_VERSION_METADATA_KEY: &str = "tenax.stdio.version";
/// Current value of [`STDIO_PROTOCOL_VERSION_METADATA_KEY`].
pub const STDIO_PROTOCOL_VERSION: &str = "1";

type RequestWriter = StreamWriter<ChildStdin>;
type ResultReader = StreamReader<ChildStdout>;

/// Process-level failures from [`StdioEvaluator`].
///
/// These errors are distinct from per-row [`crate::RowFailure`] values. A
/// driver may retry the affected stable evaluation IDs after replacing a
/// failed subprocess.
#[derive(Debug, Error)]
pub enum StdioError {
    /// The configured child command could not be started.
    #[error("failed to spawn stdio evaluator process: {source}")]
    Spawn {
        /// Operating-system process error.
        #[source]
        source: io::Error,
    },

    /// A pipe requested from `Command` was unexpectedly unavailable.
    #[error("spawned stdio evaluator has no piped {pipe}")]
    MissingPipe {
        /// Missing standard stream.
        pipe: &'static str,
    },

    /// The discovery IPC stream could not be initialized or read.
    #[error("failed to read stdio evaluator discovery stream: {source}")]
    DiscoveryIpc {
        /// Arrow IPC error.
        #[source]
        source: ArrowRsError,
    },

    /// Discovery contained data even though it must be schema-only.
    #[error("stdio evaluator discovery stream must not contain record batches")]
    DiscoveryContainedBatch,

    /// Discovery omitted its stdio protocol version.
    #[error("stdio evaluator discovery schema is missing required metadata '{key}'")]
    MissingProtocolVersion {
        /// Required metadata key.
        key: &'static str,
    },

    /// Discovery selected a protocol version this client does not implement.
    #[error("unsupported stdio protocol version '{actual}'; expected '{expected}'")]
    UnsupportedProtocolVersion {
        /// Supported version.
        expected: &'static str,
        /// Version emitted by the child.
        actual: String,
    },

    /// A discovery, request, or result value violated the Arrow contract.
    #[error("stdio Arrow contract violation: {source}")]
    ArrowContract {
        /// Validated boundary-conversion error.
        #[source]
        source: ArrowConversionError,
    },

    /// The result IPC stream could not be initialized or read.
    #[error("failed to read stdio evaluator result stream: {source}")]
    ResultIpc {
        /// Arrow IPC error.
        #[source]
        source: ArrowRsError,
    },

    /// A request could not be written to Arrow IPC.
    #[error("failed to write stdio evaluator request stream: {source}")]
    RequestIpc {
        /// Arrow IPC error.
        #[source]
        source: ArrowRsError,
    },

    /// An input request does not satisfy the discovered child schema.
    #[error("request {id} does not satisfy the stdio evaluator schema: {source}")]
    InvalidRequest {
        /// Stable request identifier.
        id: EvaluationId,
        /// Input schema/domain error.
        #[source]
        source: InputChunkError,
    },

    /// One call submitted the same stable ID more than once.
    #[error("evaluation request ID {id} occurs more than once in one stdio batch")]
    DuplicateRequestId {
        /// Duplicated identifier.
        id: EvaluationId,
    },

    /// A second evaluation was started while the prior stream was still live.
    #[error("this stdio evaluator already has an active evaluation stream")]
    EvaluationInProgress,

    /// The persistent process has already failed or been shut down.
    #[error("the stdio evaluator process is no longer available")]
    Unavailable,

    /// Internal synchronization state was poisoned by a panic.
    #[error("stdio evaluator {component} synchronization state is poisoned")]
    Poisoned {
        /// Affected state component.
        component: &'static str,
    },

    /// The background request writer thread could not be created.
    #[error("failed to start stdio request writer thread: {source}")]
    WriterThreadSpawn {
        /// Thread creation error.
        #[source]
        source: io::Error,
    },

    /// The background request writer panicked.
    #[error("stdio request writer thread panicked")]
    WriterThreadPanicked,

    /// The child returned an ID that was not pending in this call.
    #[error("stdio evaluator returned unexpected or duplicate evaluation ID {id}")]
    UnexpectedResultId {
        /// Unexpected identifier.
        id: EvaluationId,
    },

    /// A result's row count differs from its identified request.
    #[error("stdio evaluator result {id} has {actual} rows, but its request has {expected} rows")]
    ResultRowCountMismatch {
        /// Stable request identifier.
        id: EvaluationId,
        /// Number of request rows.
        expected: usize,
        /// Number of result rows.
        actual: usize,
    },

    /// The result stream ended while requests were still pending.
    #[error("stdio evaluator closed its result stream with {pending} request(s) still pending")]
    UnexpectedEnd {
        /// Number of requests without results.
        pending: usize,
    },

    /// The child exited before all pending results arrived.
    #[error(
        "stdio evaluator process exited with status {status} while {pending} request(s) were pending"
    )]
    ChildExited {
        /// Process exit status.
        status: ExitStatus,
        /// Number of requests without results.
        pending: usize,
    },

    /// Querying the child process state failed.
    #[error("failed to query stdio evaluator process status: {source}")]
    ChildStatus {
        /// Operating-system process error.
        #[source]
        source: io::Error,
    },

    /// Graceful shutdown could not finish the request stream.
    #[error("failed to finish stdio evaluator request stream: {source}")]
    ShutdownRequestIpc {
        /// Arrow IPC error.
        #[source]
        source: ArrowRsError,
    },

    /// The child returned an unrequested result while shutting down.
    #[error("stdio evaluator returned an unrequested result while shutting down")]
    UnexpectedResultDuringShutdown,

    /// Waiting for a graceful child shutdown failed.
    #[error("failed to wait for stdio evaluator process: {source}")]
    ChildWait {
        /// Operating-system process error.
        #[source]
        source: io::Error,
    },
}

struct ProcessState {
    child: Mutex<Child>,
    terminal: AtomicBool,
}

impl ProcessState {
    const fn new(child: Child) -> Self {
        Self {
            child: Mutex::new(child),
            terminal: AtomicBool::new(false),
        }
    }

    fn is_terminal(&self) -> bool {
        self.terminal.load(Ordering::Acquire)
    }

    fn terminate(&self) {
        if self.terminal.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut child = match self.child.lock() {
            Ok(child) => child,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !matches!(child.try_wait(), Ok(Some(_))) {
            drop(child.kill());
            drop(child.wait());
        }
    }

    fn unexpected_end(&self, pending: usize) -> StdioError {
        let Ok(mut child) = self.child.lock() else {
            return StdioError::Poisoned {
                component: "child process",
            };
        };
        match child.try_wait() {
            Ok(Some(status)) => StdioError::ChildExited { status, pending },
            Ok(None) => StdioError::UnexpectedEnd { pending },
            Err(source) => StdioError::ChildStatus { source },
        }
    }
}

/// A persistent child process implementing the Tenax stdio Arrow IPC protocol.
///
/// Construction performs schema discovery and validates the child's fixed
/// result-stream schema. Each [`Evaluator::evaluate`] call writes all request
/// batches on a background thread while the returned iterator reads results,
/// preventing full pipe buffers in either direction from deadlocking the
/// exchange. Results may arrive out of request order.
///
/// Only one iterator may be active at a time. Dropping it before exhaustion
/// terminates the process because unread result bytes would otherwise make the
/// persistent stream ambiguous. Use [`StdioEvaluator::shutdown`] to finish the
/// request stream and let a conforming child exit cleanly.
pub struct StdioEvaluator {
    schema: ModelSchema,
    request_writer: Arc<Mutex<RequestWriter>>,
    result_reader: Mutex<ResultReader>,
    process: Arc<ProcessState>,
    evaluation_session: Mutex<()>,
}

impl StdioEvaluator {
    /// Spawns and handshakes with a model-server command.
    ///
    /// The command is passed directly to [`Command`]; no shell parses its
    /// executable, arguments, or environment. This method overrides stdin and
    /// stdout with pipes and leaves stderr configured as supplied by the
    /// caller (inherited by default).
    ///
    /// # Errors
    ///
    /// Returns [`StdioError`] when spawning fails, the child does not emit the
    /// required two-stream prelude, or either discovered schema violates the
    /// protocol.
    pub fn spawn(mut command: Command) -> Result<Self, StdioError> {
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|source| StdioError::Spawn { source })?;
        let Some(stdin) = child.stdin.take() else {
            terminate_child(&mut child);
            return Err(StdioError::MissingPipe { pipe: "stdin" });
        };
        let Some(stdout) = child.stdout.take() else {
            terminate_child(&mut child);
            return Err(StdioError::MissingPipe { pipe: "stdout" });
        };

        match initialize_streams(stdin, stdout) {
            Ok((schema, request_writer, result_reader)) => Ok(Self {
                schema,
                request_writer: Arc::new(Mutex::new(request_writer)),
                result_reader: Mutex::new(result_reader),
                process: Arc::new(ProcessState::new(child)),
                evaluation_session: Mutex::new(()),
            }),
            Err(error) => {
                terminate_child(&mut child);
                Err(error)
            }
        }
    }

    /// Finishes the request stream and waits for a conforming child to exit.
    ///
    /// # Errors
    ///
    /// Returns [`StdioError`] if IPC shutdown fails, the child emits an
    /// unrequested result, or waiting for process exit fails.
    pub fn shutdown(self) -> Result<ExitStatus, StdioError> {
        {
            let mut writer = self
                .request_writer
                .lock()
                .map_err(|_| StdioError::Poisoned {
                    component: "request writer",
                })?;
            writer
                .finish()
                .map_err(|source| StdioError::ShutdownRequestIpc { source })?;
        }
        {
            let mut reader = self
                .result_reader
                .lock()
                .map_err(|_| StdioError::Poisoned {
                    component: "result reader",
                })?;
            match reader.next() {
                None => {}
                Some(Ok(_)) => return Err(StdioError::UnexpectedResultDuringShutdown),
                Some(Err(source)) => return Err(StdioError::ResultIpc { source }),
            }
        }
        let status = self
            .process
            .child
            .lock()
            .map_err(|_| StdioError::Poisoned {
                component: "child process",
            })?
            .wait()
            .map_err(|source| StdioError::ChildWait { source })?;
        self.process.terminal.store(true, Ordering::Release);
        Ok(status)
    }

    fn prepare_requests(
        &self,
        requests: &[EvalRequest],
    ) -> Result<BTreeMap<EvaluationId, usize>, StdioError> {
        if self.process.is_terminal() {
            return Err(StdioError::Unavailable);
        }
        requests
            .iter()
            .try_fold(BTreeMap::new(), |mut pending, request| {
                request
                    .inputs()
                    .validate_against(&self.schema)
                    .map_err(|source| StdioError::InvalidRequest {
                        id: request.id(),
                        source,
                    })?;
                if pending
                    .insert(request.id(), request.inputs().row_count())
                    .is_some()
                {
                    return Err(StdioError::DuplicateRequestId { id: request.id() });
                }
                Ok(pending)
            })
    }
}

impl Evaluator for StdioEvaluator {
    type Error = StdioError;

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn evaluate(
        &self,
        requests: Vec<EvalRequest>,
    ) -> Box<dyn Iterator<Item = Result<ChunkResult, Self::Error>> + '_> {
        if requests.is_empty() {
            return Box::new(std::iter::empty());
        }
        let session = match self.evaluation_session.try_lock() {
            Ok(session) => session,
            Err(TryLockError::WouldBlock) => {
                return Box::new(std::iter::once(Err(StdioError::EvaluationInProgress)));
            }
            Err(TryLockError::Poisoned(_)) => {
                return Box::new(std::iter::once(Err(StdioError::Poisoned {
                    component: "evaluation session",
                })));
            }
        };
        let pending = match self.prepare_requests(&requests) {
            Ok(pending) => pending,
            Err(error) => return Box::new(std::iter::once(Err(error))),
        };

        let writer = Arc::clone(&self.request_writer);
        let process = Arc::clone(&self.process);
        let schema = self.schema.clone();
        let (status_sender, status_receiver) = mpsc::channel();
        let writer_thread = match thread::Builder::new()
            .name("tenax-stdio-request-writer".to_owned())
            .spawn(move || {
                let status = write_requests(&schema, &writer, requests);
                if status.is_err() {
                    process.terminate();
                }
                drop(status_sender.send(status));
            }) {
            Ok(thread) => thread,
            Err(source) => {
                return Box::new(std::iter::once(Err(StdioError::WriterThreadSpawn {
                    source,
                })));
            }
        };

        Box::new(ActiveEvaluation {
            evaluator: self,
            _session: session,
            pending,
            writer_status: status_receiver,
            writer_thread: Some(writer_thread),
            done: false,
        })
    }
}

impl Drop for StdioEvaluator {
    fn drop(&mut self) {
        self.process.terminate();
    }
}

struct ActiveEvaluation<'evaluator> {
    evaluator: &'evaluator StdioEvaluator,
    _session: MutexGuard<'evaluator, ()>,
    pending: BTreeMap<EvaluationId, usize>,
    writer_status: Receiver<Result<(), StdioError>>,
    writer_thread: Option<JoinHandle<()>>,
    done: bool,
}

impl ActiveEvaluation<'_> {
    fn finish_writer(&mut self) -> Option<Result<ChunkResult, StdioError>> {
        let status = self.writer_status.recv();
        let joined = self
            .writer_thread
            .take()
            .is_none_or(|thread| thread.join().is_ok());
        self.done = true;
        match (status, joined) {
            (Ok(Ok(())), true) => None,
            (Ok(Err(error)), _) => Some(Err(error)),
            (Err(_), _) | (_, false) => Some(Err(StdioError::WriterThreadPanicked)),
        }
    }

    fn fail(&mut self, error: StdioError) -> Result<ChunkResult, StdioError> {
        self.evaluator.process.terminate();
        if let Some(writer_thread) = self.writer_thread.take() {
            drop(writer_thread.join());
        }
        self.done = true;
        Err(error)
    }
}

impl Iterator for ActiveEvaluation<'_> {
    type Item = Result<ChunkResult, StdioError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if self.pending.is_empty() {
            return self.finish_writer();
        }

        let next_batch = match self.evaluator.result_reader.lock() {
            Ok(mut reader) => reader.next(),
            Err(_) => {
                return Some(self.fail(StdioError::Poisoned {
                    component: "result reader",
                }));
            }
        };
        let batch = match next_batch {
            Some(Ok(batch)) => batch,
            Some(Err(source)) => return Some(self.fail(StdioError::ResultIpc { source })),
            None => {
                let error = match self.writer_status.try_recv() {
                    Ok(Err(error)) => error,
                    Err(TryRecvError::Disconnected) => StdioError::WriterThreadPanicked,
                    Ok(Ok(())) | Err(TryRecvError::Empty) => {
                        self.evaluator.process.unexpected_end(self.pending.len())
                    }
                };
                return Some(self.fail(error));
            }
        };
        let result = match ChunkResult::try_from((&self.evaluator.schema, &batch)) {
            Ok(result) => result,
            Err(source) => return Some(self.fail(StdioError::ArrowContract { source })),
        };
        let id = result.id();
        let Some(expected_rows) = self.pending.remove(&id) else {
            return Some(self.fail(StdioError::UnexpectedResultId { id }));
        };
        if result.rows().len() != expected_rows {
            return Some(self.fail(StdioError::ResultRowCountMismatch {
                id,
                expected: expected_rows,
                actual: result.rows().len(),
            }));
        }
        Some(Ok(result))
    }
}

impl Drop for ActiveEvaluation<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        if self.pending.is_empty() {
            drop(self.finish_writer());
            return;
        }
        self.evaluator.process.terminate();
        if let Some(writer_thread) = self.writer_thread.take() {
            drop(writer_thread.join());
        }
        self.done = true;
    }
}

fn initialize_streams(
    stdin: ChildStdin,
    mut stdout: ChildStdout,
) -> Result<(ModelSchema, RequestWriter, ResultReader), StdioError> {
    let discovery_schema = {
        let mut discovery = StreamReader::try_new(&mut stdout, None)
            .map_err(|source| StdioError::DiscoveryIpc { source })?;
        let schema = discovery.schema();
        match discovery.next() {
            None => {}
            Some(Ok(_)) => return Err(StdioError::DiscoveryContainedBatch),
            Some(Err(source)) => return Err(StdioError::DiscoveryIpc { source }),
        }
        schema
    };
    validate_stdio_discovery_schema(&discovery_schema)?;
    let schema = ModelSchema::try_from(discovery_schema.as_ref())
        .map_err(|source| StdioError::ArrowContract { source })?;

    let result_reader =
        StreamReader::try_new(stdout, None).map_err(|source| StdioError::ResultIpc { source })?;
    validate_evaluation_result_schema(&schema, result_reader.schema().as_ref())
        .map_err(|source| StdioError::ArrowContract { source })?;

    let request_schema = evaluation_request_schema(&schema)
        .map_err(|source| StdioError::ArrowContract { source })?;
    let mut request_writer = StreamWriter::try_new(stdin, &request_schema)
        .map_err(|source| StdioError::RequestIpc { source })?;
    request_writer
        .flush()
        .map_err(|source| StdioError::RequestIpc { source })?;
    Ok((schema, request_writer, result_reader))
}

fn write_requests(
    schema: &ModelSchema,
    writer: &Mutex<RequestWriter>,
    requests: Vec<EvalRequest>,
) -> Result<(), StdioError> {
    let mut writer = writer.lock().map_err(|_| StdioError::Poisoned {
        component: "request writer",
    })?;
    for request in requests {
        let batch = RecordBatch::try_from(EvalRequestRef::new(schema, &request))
            .map_err(|source| StdioError::ArrowContract { source })?;
        writer
            .write(&batch)
            .map_err(|source| StdioError::RequestIpc { source })?;
        writer
            .flush()
            .map_err(|source| StdioError::RequestIpc { source })?;
    }
    drop(writer);
    Ok(())
}

fn terminate_child(child: &mut Child) {
    if !matches!(child.try_wait(), Ok(Some(_))) {
        drop(child.kill());
        drop(child.wait());
    }
}

/// Adds the stdio protocol version to a normal model discovery schema.
///
/// # Errors
///
/// Returns [`ArrowConversionError`] if `schema` cannot be represented by the
/// Arrow model contract.
pub fn stdio_discovery_schema(schema: &ModelSchema) -> Result<ArrowSchema, ArrowConversionError> {
    let arrow = ArrowSchema::try_from(schema)?;
    let mut metadata = arrow.metadata().clone();
    metadata.insert(
        STDIO_PROTOCOL_VERSION_METADATA_KEY.to_owned(),
        STDIO_PROTOCOL_VERSION.to_owned(),
    );
    Ok(ArrowSchema::new_with_metadata(
        arrow.fields().clone(),
        metadata,
    ))
}

/// Validates stdio-specific discovery metadata.
///
/// Model fields and the general Arrow contract are validated separately by
/// `ModelSchema::try_from`.
///
/// # Errors
///
/// Returns [`StdioError`] when the version is absent or unsupported.
pub fn validate_stdio_discovery_schema(schema: &ArrowSchema) -> Result<(), StdioError> {
    let actual = schema
        .metadata()
        .get(STDIO_PROTOCOL_VERSION_METADATA_KEY)
        .ok_or(StdioError::MissingProtocolVersion {
            key: STDIO_PROTOCOL_VERSION_METADATA_KEY,
        })?;
    if actual == STDIO_PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(StdioError::UnsupportedProtocolVersion {
            expected: STDIO_PROTOCOL_VERSION,
            actual: actual.clone(),
        })
    }
}

/// Failures raised by the reference [`serve_stdio`] process shell.
#[derive(Debug, Error)]
pub enum StdioServerError {
    /// Discovery or batch conversion failed.
    #[error("stdio server Arrow contract failure: {source}")]
    ArrowContract {
        /// Boundary-conversion error.
        #[source]
        source: ArrowConversionError,
    },

    /// Reading or writing Arrow IPC failed.
    #[error("stdio server IPC failure: {source}")]
    Ipc {
        /// Arrow IPC error.
        #[source]
        source: ArrowRsError,
    },

    /// The wrapped evaluator failed outside an individual row.
    #[error("wrapped evaluator failed: {message}")]
    Evaluator {
        /// Evaluator diagnostic.
        message: String,
    },

    /// One request did not produce exactly one result chunk.
    #[error("wrapped evaluator returned no result for request {id}")]
    MissingResult {
        /// Request identifier.
        id: EvaluationId,
    },

    /// One request produced more than one result chunk.
    #[error("wrapped evaluator returned more than one result for request {id}")]
    MultipleResults {
        /// Request identifier.
        id: EvaluationId,
    },

    /// The wrapped evaluator returned the wrong request identifier.
    #[error("wrapped evaluator returned result {actual} for request {expected}")]
    MismatchedResultId {
        /// Submitted request identifier.
        expected: EvaluationId,
        /// Returned result identifier.
        actual: EvaluationId,
    },

    /// The wrapped evaluator returned the wrong row count.
    #[error("wrapped evaluator result {id} has {actual} rows, but its request has {expected} rows")]
    ResultRowCountMismatch {
        /// Request identifier.
        id: EvaluationId,
        /// Request row count.
        expected: usize,
        /// Result row count.
        actual: usize,
    },
}

/// Serves an [`Evaluator`] on this process's stdin and stdout.
///
/// This blocking reference shell is intended for Rust model servers and
/// protocol tests. It writes the schema-only discovery stream and result
/// schema before reading requests, evaluates one request batch at a time, and
/// flushes each result. A Graphcal server may instead read ahead, evaluate in
/// parallel, and emit result batches in completion order as long as it follows
/// the same wire contract.
///
/// The function writes no textual data to stdout. Model logs and diagnostics
/// must use stderr or another sink.
///
/// # Errors
///
/// Returns [`StdioServerError`] for malformed requests, evaluator-level
/// failures, wrong result identity/count, or IPC I/O failures.
pub fn serve_stdio<E>(evaluator: &E) -> Result<(), StdioServerError>
where
    E: Evaluator,
{
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let discovery_schema = stdio_discovery_schema(evaluator.schema())
        .map_err(|source| StdioServerError::ArrowContract { source })?;
    {
        let mut discovery = StreamWriter::try_new(&mut stdout, &discovery_schema)
            .map_err(|source| StdioServerError::Ipc { source })?;
        discovery
            .finish()
            .map_err(|source| StdioServerError::Ipc { source })?;
    }

    let result_schema = evaluation_result_schema(evaluator.schema());
    let mut results = StreamWriter::try_new(&mut stdout, &result_schema)
        .map_err(|source| StdioServerError::Ipc { source })?;
    results
        .flush()
        .map_err(|source| StdioServerError::Ipc { source })?;

    let stdin = io::stdin();
    let mut requests = StreamReader::try_new(stdin.lock(), None)
        .map_err(|source| StdioServerError::Ipc { source })?;
    validate_evaluation_request_schema(evaluator.schema(), requests.schema().as_ref())
        .map_err(|source| StdioServerError::ArrowContract { source })?;

    for batch in requests.by_ref() {
        let batch = batch.map_err(|source| StdioServerError::Ipc { source })?;
        let request = EvalRequest::try_from((evaluator.schema(), &batch))
            .map_err(|source| StdioServerError::ArrowContract { source })?;
        let id = request.id();
        let row_count = request.inputs().row_count();
        let mut evaluated = evaluator.evaluate(vec![request]);
        let result = evaluated
            .next()
            .ok_or(StdioServerError::MissingResult { id })?
            .map_err(|error| StdioServerError::Evaluator {
                message: error.to_string(),
            })?;
        if evaluated.next().is_some() {
            return Err(StdioServerError::MultipleResults { id });
        }
        if result.id() != id {
            return Err(StdioServerError::MismatchedResultId {
                expected: id,
                actual: result.id(),
            });
        }
        if result.rows().len() != row_count {
            return Err(StdioServerError::ResultRowCountMismatch {
                id,
                expected: row_count,
                actual: result.rows().len(),
            });
        }
        let batch = RecordBatch::try_from(ChunkResultRef::new(evaluator.schema(), &result))
            .map_err(|source| StdioServerError::ArrowContract { source })?;
        results
            .write(&batch)
            .map_err(|source| StdioServerError::Ipc { source })?;
        results
            .flush()
            .map_err(|source| StdioServerError::Ipc { source })?;
    }
    drop(requests);
    results
        .finish()
        .map_err(|source| StdioServerError::Ipc { source })
}
