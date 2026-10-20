//! Arrow IPC streams of evaluation tables.
//!
//! An evaluation stream is an Arrow IPC stream (conventionally a `.arrows`
//! file) in the session's [`evaluation_schema`]: one record batch per
//! evaluation, appended and flushed as each evaluation completes. A run that
//! stops part-way therefore leaves every evaluation written before the stop
//! readable (see [`EvaluationReader::recover_table`]).
//!
//! The Arrow IPC stream format does not distinguish a stream that was
//! finished from one that stopped between two messages, so a reader cannot
//! tell whether evaluations are missing after the last complete one.

use std::io::{Read, Write};
use std::sync::Arc;

use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::ArrowError;
use graphcal_model_protocol::SessionSchema;
use thiserror::Error;

use crate::evaluation::SessionMismatch;
use crate::evaluation_arrow::{EvaluationDecodeError, EvaluationDecoder, evaluation_schema};
use crate::evaluation_table::{EvaluatedRequest, EvaluationTable};

/// Errors raised while writing an evaluation stream.
#[derive(Debug, Error)]
pub enum EvaluationWriteError {
    /// The evaluation belongs to another session than the stream's.
    #[error(transparent)]
    Session(#[from] SessionMismatch),

    /// Arrow could not encode or write the evaluation.
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

/// Appends evaluations of one session to an Arrow IPC stream.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
///
/// use tenax::{EvaluationReader, EvaluationTable, EvaluationWriter, ModelSchemaBuilder};
///
/// let schema = ModelSchemaBuilder::new()
///     .continuous("load", 0.0, 1.0)?
///     .boolean_output("failure")?
///     .build()?;
/// let table = EvaluationTable::new(Arc::clone(&schema));
///
/// let mut writer = EvaluationWriter::try_new(Vec::new(), &schema)?;
/// writer.write_table(&table)?;
/// let bytes = writer.finish()?;
///
/// let read = EvaluationReader::try_new(bytes.as_slice())?.read_table()?;
/// assert_eq!(read, table);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct EvaluationWriter<W: Write> {
    session: Arc<SessionSchema>,
    writer: StreamWriter<W>,
}

impl<W: Write> EvaluationWriter<W> {
    /// Starts a stream of `session`'s evaluations by writing its schema.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrowError`] when the schema cannot be written.
    pub fn try_new(writer: W, session: &Arc<SessionSchema>) -> Result<Self, ArrowError> {
        let mut writer = StreamWriter::try_new(writer, &evaluation_schema(session))?;
        writer.flush()?;
        Ok(Self {
            session: Arc::clone(session),
            writer,
        })
    }

    /// Appends one evaluation as one record batch and flushes it.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationWriteError`] when the evaluation belongs to another
    /// session or cannot be written.
    pub fn write(&mut self, evaluation: &EvaluatedRequest) -> Result<(), EvaluationWriteError> {
        evaluation.request().validate_against(&self.session)?;
        self.writer.write(&evaluation.to_record_batch()?)?;
        self.writer.flush()?;
        Ok(())
    }

    /// Appends every evaluation of `table`, in table order.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationWriteError`] for the first evaluation that cannot
    /// be written; earlier evaluations remain written.
    pub fn write_table(&mut self, table: &EvaluationTable) -> Result<(), EvaluationWriteError> {
        table
            .evaluations()
            .iter()
            .try_for_each(|evaluation| self.write(evaluation))
    }

    /// Ends the stream and returns the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrowError`] when the end-of-stream marker cannot be
    /// written or flushed.
    pub fn finish(mut self) -> Result<W, ArrowError> {
        self.writer.finish()?;
        self.writer.into_inner()
    }
}

/// Errors raised while reading an evaluation stream.
#[derive(Debug, Error)]
pub enum EvaluationReadError {
    /// The stream could not be read as Arrow IPC.
    #[error(transparent)]
    Arrow(#[from] ArrowError),

    /// The stream's contents are not a valid evaluation table.
    #[error(transparent)]
    Decode(#[from] EvaluationDecodeError),
}

/// The evaluations recovered from a stream that may have stopped part-way.
#[derive(Debug)]
pub struct RecoveredTable {
    table: EvaluationTable,
    stopped_by: Option<ArrowError>,
}

impl RecoveredTable {
    /// Returns the evaluations read before the stream ended or stopped.
    #[must_use]
    pub const fn table(&self) -> &EvaluationTable {
        &self.table
    }

    /// Consumes the recovery and returns its evaluations.
    #[must_use]
    pub fn into_table(self) -> EvaluationTable {
        self.table
    }

    /// Returns the Arrow error that stopped reading, or `None` when the
    /// stream ended at a message boundary.
    #[must_use]
    pub const fn stopped_by(&self) -> Option<&ArrowError> {
        self.stopped_by.as_ref()
    }
}

/// Reads an evaluation stream written by [`EvaluationWriter`] or any writer of
/// the same schema.
///
/// The stream's session is restored from its schema, without the model.
pub struct EvaluationReader<R: Read> {
    reader: StreamReader<R>,
    decoder: EvaluationDecoder,
}

impl<R: Read> EvaluationReader<R> {
    /// Reads the stream's schema and restores its session.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationReadError`] when the stream does not start with an
    /// Arrow schema, or the schema is not an evaluation table schema.
    pub fn try_new(reader: R) -> Result<Self, EvaluationReadError> {
        let reader = StreamReader::try_new(reader, None)?;
        let decoder = EvaluationDecoder::new(&reader.schema())?;
        Ok(Self { reader, decoder })
    }

    /// Returns the session restored from the stream's schema.
    #[must_use]
    pub const fn session(&self) -> &Arc<SessionSchema> {
        self.decoder.session()
    }

    /// Reads every evaluation in the stream.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationReadError`] when any message cannot be read,
    /// including a stream that stops inside a message, or when the contents
    /// are not a valid evaluation table.
    pub fn read_table(mut self) -> Result<EvaluationTable, EvaluationReadError> {
        for batch in &mut self.reader {
            self.decoder.push(&batch?)?;
        }
        Ok(self.decoder.finish()?)
    }

    /// Reads evaluations until the stream ends or a message cannot be read,
    /// such as a final message cut short when a run stopped while writing it.
    ///
    /// The evaluation being read when the stream stops keeps the rows read
    /// before the stop; with [`EvaluationWriter`], which writes each
    /// evaluation in one message, that evaluation is complete.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationDecodeError`] when a readable message is not a
    /// valid part of an evaluation table. Unreadable messages are not errors;
    /// see [`RecoveredTable::stopped_by`].
    pub fn recover_table(mut self) -> Result<RecoveredTable, EvaluationDecodeError> {
        let mut stopped_by = None;
        for batch in &mut self.reader {
            match batch {
                Ok(batch) => self.decoder.push(&batch)?,
                Err(error) => {
                    stopped_by = Some(error);
                    break;
                }
            }
        }
        Ok(RecoveredTable {
            table: self.decoder.finish()?,
            stopped_by,
        })
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{FiniteF64, InputColumn, RequestBatch, Value, WireType};

    use super::*;
    use crate::{
        ChunkResult, EvalRequest, EvaluationId, ModelSchemaBuilder, OutputRow, RowOutcome,
    };

    fn session() -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .categorical("mode", "Mode", ["safe", "risky"])
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap()
    }

    fn evaluation(session: &Arc<SessionSchema>, id: u128, xs: &[f64]) -> EvaluatedRequest {
        let WireType::Key(mode) = session.inputs()[1].wire() else {
            panic!("mode is a key");
        };
        let first = mode.positions().next().unwrap();
        let inputs = RequestBatch::new(
            Arc::clone(session),
            vec![
                InputColumn::Quantity(xs.iter().map(|x| FiniteF64::new(*x).unwrap()).collect()),
                InputColumn::Key(vec![first; xs.len()]),
            ],
        )
        .unwrap();
        let rows = xs
            .iter()
            .map(|x| {
                RowOutcome::Success(OutputRow::new(session, vec![Value::Bool(*x > 0.5)]).unwrap())
            })
            .collect();
        EvaluatedRequest::new(
            session,
            EvalRequest::new(EvaluationId::new(id), 0, inputs),
            ChunkResult::new(EvaluationId::new(id), rows).unwrap(),
        )
        .unwrap()
    }

    fn table(session: &Arc<SessionSchema>) -> EvaluationTable {
        EvaluationTable::from_evaluations(
            Arc::clone(session),
            [
                evaluation(session, 1, &[0.1, 0.9]),
                evaluation(session, 2, &[0.4, 0.6, 0.8]),
            ],
        )
        .unwrap()
    }

    fn stream(table: &EvaluationTable) -> Vec<u8> {
        let mut writer = EvaluationWriter::try_new(Vec::new(), table.schema()).unwrap();
        writer.write_table(table).unwrap();
        writer.finish().unwrap()
    }

    #[test]
    fn streams_round_trip_and_restore_their_session() {
        let session = session();
        let table = table(&session);
        let bytes = stream(&table);
        let reader = EvaluationReader::try_new(bytes.as_slice()).unwrap();
        assert_eq!(reader.session().as_ref(), session.as_ref());
        assert_eq!(reader.read_table().unwrap(), table);
    }

    #[test]
    fn writers_reject_evaluations_of_other_sessions() {
        let session = session();
        let other = ModelSchemaBuilder::new()
            .continuous("y", 0.0, 1.0)
            .unwrap()
            .categorical("mode", "Mode", ["safe", "risky"])
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        let mut writer = EvaluationWriter::try_new(Vec::new(), &session).unwrap();
        assert!(matches!(
            writer.write(&evaluation(&other, 1, &[0.1])),
            Err(EvaluationWriteError::Session(_))
        ));
    }

    #[test]
    fn truncated_streams_fail_strictly_and_recover_complete_evaluations() {
        let session = session();
        let table = table(&session);
        let bytes = stream(&table);

        // Cut inside the last evaluation's message.
        let first_only = {
            let mut writer = EvaluationWriter::try_new(Vec::new(), &session).unwrap();
            writer.write(&table.evaluations()[0]).unwrap();
            writer.finish().unwrap()
        };
        // The end-of-stream marker is 8 bytes; the first evaluation ends
        // before it.
        let cut = &bytes[..first_only.len() - 8 + 20];
        assert!(matches!(
            EvaluationReader::try_new(cut).unwrap().read_table(),
            Err(EvaluationReadError::Arrow(_))
        ));
        let recovered = EvaluationReader::try_new(cut)
            .unwrap()
            .recover_table()
            .unwrap();
        assert!(recovered.stopped_by().is_some());
        assert_eq!(recovered.table().evaluations(), &table.evaluations()[..1]);

        // A stream that stops between messages reads as complete.
        let boundary = &bytes[..first_only.len() - 8];
        let recovered = EvaluationReader::try_new(boundary)
            .unwrap()
            .recover_table()
            .unwrap();
        assert!(recovered.stopped_by().is_none());
        assert_eq!(recovered.into_table().row_count(), 2);
    }
}
