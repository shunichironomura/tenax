//! Arrow encoding of evaluation tables.
//!
//! One Arrow row describes one evaluated input row. The schema extends a
//! Graphcal session record (see
//! [`RecordSchema`](graphcal_model_protocol::arrow::RecordSchema)), whose
//! `inputs` and `outputs` structs and `graphcal` metadata Graphcal encodes,
//! with Tenax's identity and outcome columns:
//!
//! | Field | Arrow type | Content |
//! |---|---|---|
//! | `evaluation_id` | `FixedSizeBinary(16)` | [`EvaluationId`], big-endian |
//! | `row` | `UInt64` | Zero-based row within the request |
//! | `seed` | `UInt64` | The request's model seed |
//! | `inputs` | `Struct` | One child per varied parameter |
//! | `outputs` | `Struct` | One nullable child per read output, null in failed rows |
//! | `outcome` | `Struct` | `kind`, `subject`, `message`, `expected_count`, `actual_count` |
//!
//! `outcome.kind` is a `Dictionary<Int8, Utf8>` naming how the row ended:
//! `success`, `assertion_failed`, `input_out_of_domain`, `evaluation_error`,
//! `panic`, `non_string_panic`, `invalid_input`, `output_count_mismatch`, or
//! `output_kind_mismatch`. `subject` holds the failed assertion, the
//! out-of-domain parameter, or the mismatched output; `message` holds the
//! diagnostic; and the counts describe an output count mismatch. The schema's
//! `tenax` metadata key holds the format version and the writing Tenax
//! version as JSON.

use std::sync::Arc;

use arrow_array::types::Int8Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeBinaryArray, Int8Array, RecordBatch, StringArray,
    StructArray, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema, SchemaRef};
use graphcal_model_protocol::arrow::{
    OutputDecodeError, RecordSchema, RecordSchemaError, RequestDecodeError, SchemaMismatch,
    record_inputs_field, record_metadata, record_outputs_field, validate_schema,
};
use graphcal_model_protocol::{
    InputColumn, LabelError, OutputSlot, OutputValues, OutputValuesBuilder, PortName, RequestBatch,
    RowFailure as ProtocolRowFailure, SessionSchema, Value, ValueRef,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::evaluation::{
    ChunkResult, EvalRequest, EvaluationId, ModelPanic, OutputRow, OutputRowError, RowFailure,
    RowOutcome, SessionMismatch,
};
use crate::evaluation_dataset::concatenate;
use crate::evaluation_table::{EvaluatedRequest, EvaluationTable, EvaluationTableError};

const EVALUATION_ID_FIELD: &str = "evaluation_id";
const ROW_FIELD: &str = "row";
const SEED_FIELD: &str = "seed";
const OUTCOME_FIELD: &str = "outcome";
const KIND_FIELD: &str = "kind";
const SUBJECT_FIELD: &str = "subject";
const MESSAGE_FIELD: &str = "message";
const EXPECTED_COUNT_FIELD: &str = "expected_count";
const ACTUAL_COUNT_FIELD: &str = "actual_count";
const EVALUATION_ID_BYTES: i32 = 16;

/// The schema metadata key of Tenax's own metadata.
pub const TENAX_METADATA_KEY: &str = "tenax";

/// The version of the evaluation table layout this Tenax writes and reads.
pub const EVALUATION_FORMAT_VERSION: u32 = 1;

/// The value of the schema's `tenax` metadata key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TenaxMetadata {
    format_version: u32,
    tenax_version: String,
}

/// How one row ended, as recorded in `outcome.kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutcomeKind {
    Success,
    AssertionFailed,
    InputOutOfDomain,
    EvaluationError,
    Panic,
    NonStringPanic,
    InvalidInput,
    OutputCountMismatch,
    OutputKindMismatch,
}

impl OutcomeKind {
    /// Every kind, in dictionary order.
    const ALL: [Self; 9] = [
        Self::Success,
        Self::AssertionFailed,
        Self::InputOutOfDomain,
        Self::EvaluationError,
        Self::Panic,
        Self::NonStringPanic,
        Self::InvalidInput,
        Self::OutputCountMismatch,
        Self::OutputKindMismatch,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::AssertionFailed => "assertion_failed",
            Self::InputOutOfDomain => "input_out_of_domain",
            Self::EvaluationError => "evaluation_error",
            Self::Panic => "panic",
            Self::NonStringPanic => "non_string_panic",
            Self::InvalidInput => "invalid_input",
            Self::OutputCountMismatch => "output_count_mismatch",
            Self::OutputKindMismatch => "output_kind_mismatch",
        }
    }

    fn parse(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.label() == label)
    }

    /// The dictionary code of this kind.
    fn code(self) -> i8 {
        Self::ALL
            .iter()
            .position(|kind| *kind == self)
            .and_then(|position| i8::try_from(position).ok())
            .unwrap_or_default()
    }
}

/// One row outcome flattened into the `outcome` struct's columns.
struct OutcomeParts {
    kind: OutcomeKind,
    subject: Option<String>,
    message: Option<String>,
    expected_count: Option<u64>,
    actual_count: Option<u64>,
}

impl OutcomeParts {
    const fn new(kind: OutcomeKind) -> Self {
        Self {
            kind,
            subject: None,
            message: None,
            expected_count: None,
            actual_count: None,
        }
    }

    fn of(outcome: &RowOutcome) -> Self {
        let failure = match outcome {
            RowOutcome::Success(_) => return Self::new(OutcomeKind::Success),
            RowOutcome::Failure(failure) => failure,
        };
        match failure {
            RowFailure::Model(ProtocolRowFailure::AssertionFailed { assertion, message }) => Self {
                subject: Some(assertion.clone()),
                message: Some(message.clone()),
                ..Self::new(OutcomeKind::AssertionFailed)
            },
            RowFailure::Model(ProtocolRowFailure::InputOutOfDomain { parameter, message }) => {
                Self {
                    subject: Some(parameter.as_str().to_owned()),
                    message: Some(message.clone()),
                    ..Self::new(OutcomeKind::InputOutOfDomain)
                }
            }
            RowFailure::Model(ProtocolRowFailure::EvaluationError { message }) => Self {
                message: Some(message.clone()),
                ..Self::new(OutcomeKind::EvaluationError)
            },
            RowFailure::Panic(ModelPanic::Message(message)) => Self {
                message: Some(message.clone()),
                ..Self::new(OutcomeKind::Panic)
            },
            RowFailure::Panic(ModelPanic::NonStringPayload) => {
                Self::new(OutcomeKind::NonStringPanic)
            }
            RowFailure::InvalidInput(SessionMismatch) => Self::new(OutcomeKind::InvalidInput),
            RowFailure::InvalidOutput(OutputRowError::CountMismatch { expected, actual }) => Self {
                expected_count: u64::try_from(*expected).ok(),
                actual_count: u64::try_from(*actual).ok(),
                ..Self::new(OutcomeKind::OutputCountMismatch)
            },
            RowFailure::InvalidOutput(OutputRowError::KindMismatch { output }) => Self {
                subject: Some(output.as_str().to_owned()),
                ..Self::new(OutcomeKind::OutputKindMismatch)
            },
        }
    }
}

fn outcome_fields() -> Fields {
    Fields::from(vec![
        Field::new(
            KIND_FIELD,
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(SUBJECT_FIELD, DataType::Utf8, true),
        Field::new(MESSAGE_FIELD, DataType::Utf8, true),
        Field::new(EXPECTED_COUNT_FIELD, DataType::UInt64, true),
        Field::new(ACTUAL_COUNT_FIELD, DataType::UInt64, true),
    ])
}

/// The `tenax` metadata, which holds only a number and a string and so
/// always serializes.
fn tenax_metadata_json() -> String {
    serde_json::to_string(&TenaxMetadata {
        format_version: EVALUATION_FORMAT_VERSION,
        tenax_version: env!("CARGO_PKG_VERSION").to_owned(),
    })
    .unwrap_or_default()
}

/// The Arrow schema of `session`'s evaluation tables.
#[must_use]
pub fn evaluation_schema(session: &SessionSchema) -> SchemaRef {
    let mut metadata = record_metadata(session);
    metadata.insert(TENAX_METADATA_KEY.to_owned(), tenax_metadata_json());
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new(
                EVALUATION_ID_FIELD,
                DataType::FixedSizeBinary(EVALUATION_ID_BYTES),
                false,
            ),
            Field::new(ROW_FIELD, DataType::UInt64, false),
            Field::new(SEED_FIELD, DataType::UInt64, false),
            record_inputs_field(session),
            record_outputs_field(session),
            Field::new(OUTCOME_FIELD, DataType::Struct(outcome_fields()), false),
        ],
        metadata,
    ))
}

fn invalid(message: impl std::fmt::Display) -> ArrowError {
    ArrowError::InvalidArgumentError(message.to_string())
}

impl EvaluatedRequest {
    /// Encodes every row of this evaluation in its session's
    /// [`evaluation_schema`].
    ///
    /// # Errors
    ///
    /// Returns an [`ArrowError`] only if Arrow rejects arrays built from the
    /// schema's own field types, which indicates a bug.
    pub fn to_record_batch(&self) -> Result<RecordBatch, ArrowError> {
        let inputs = self.request().inputs();
        let session = inputs.schema();
        let rows = self.rows().len();
        let id = self.id().get().to_be_bytes();
        let evaluation_ids = FixedSizeBinaryArray::try_from_iter(std::iter::repeat_n(id, rows))?;
        let row_numbers = (0..rows)
            .map(u64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(invalid)?;
        let seeds = vec![self.request().seed(); rows];

        let mut values = OutputValuesBuilder::new(Arc::clone(session));
        for outcome in self.rows() {
            let row = match outcome {
                RowOutcome::Success(outputs) => Some(outputs.values().to_vec()),
                RowOutcome::Failure(_) => None,
            };
            values.push(row).map_err(invalid)?;
        }
        let outputs = values.finish().map_err(invalid)?.to_struct_array()?;

        let parts = self.rows().iter().map(OutcomeParts::of).collect::<Vec<_>>();
        let kinds = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from_iter_values(parts.iter().map(|part| part.kind.code())),
            Arc::new(StringArray::from_iter_values(
                OutcomeKind::ALL.iter().map(|kind| kind.label()),
            )),
        )?;
        let outcome: Vec<ArrayRef> = vec![
            Arc::new(kinds),
            Arc::new(StringArray::from_iter(
                parts.iter().map(|part| part.subject.as_deref()),
            )),
            Arc::new(StringArray::from_iter(
                parts.iter().map(|part| part.message.as_deref()),
            )),
            Arc::new(UInt64Array::from_iter(
                parts.iter().map(|part| part.expected_count),
            )),
            Arc::new(UInt64Array::from_iter(
                parts.iter().map(|part| part.actual_count),
            )),
        ];
        RecordBatch::try_new(
            evaluation_schema(session),
            vec![
                Arc::new(evaluation_ids),
                Arc::new(UInt64Array::from(row_numbers)),
                Arc::new(UInt64Array::from(seeds)),
                Arc::new(inputs.to_struct_array()?),
                Arc::new(outputs),
                Arc::new(StructArray::try_new(outcome_fields(), outcome, None)?),
            ],
        )
    }
}

impl EvaluationTable {
    /// Encodes the table as one record batch per evaluation, in table order,
    /// all in the session's [`evaluation_schema`].
    ///
    /// # Errors
    ///
    /// Returns an [`ArrowError`] only if Arrow rejects arrays built from the
    /// schema's own field types, which indicates a bug.
    pub fn to_record_batches(&self) -> Result<Vec<RecordBatch>, ArrowError> {
        self.evaluations()
            .iter()
            .map(EvaluatedRequest::to_record_batch)
            .collect()
    }

    /// Decodes record batches of an evaluation table whose schema is
    /// `schema`.
    ///
    /// The session is restored from `schema` itself (see
    /// [`decode_evaluation_schema`]). Rows of one evaluation must be
    /// contiguous and in row order, but may span batches; batches may also
    /// hold several evaluations.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationDecodeError`] when the schema is not an
    /// evaluation table schema, a batch has another schema, a value or
    /// outcome is malformed, rows are out of order, or the evaluations do not
    /// form a valid table.
    pub fn from_record_batches(
        schema: &Schema,
        batches: impl IntoIterator<Item = RecordBatch>,
    ) -> Result<Self, EvaluationDecodeError> {
        let mut decoder = EvaluationDecoder::new(schema)?;
        for batch in batches {
            decoder.push(&batch)?;
        }
        decoder.finish()
    }
}

/// Why Arrow data is not a valid evaluation table.
#[derive(Debug, Error, PartialEq)]
pub enum EvaluationDecodeError {
    /// The schema is not a Graphcal session record.
    #[error(transparent)]
    Record(#[from] RecordSchemaError),

    /// The schema is not the evaluation table schema of its session.
    #[error(transparent)]
    Schema(#[from] SchemaMismatch),

    /// The schema carries no valid `tenax` metadata.
    #[error("the schema carries no valid tenax metadata")]
    TenaxMetadata,

    /// The table was written in another layout version.
    #[error("the table has format version {received}; this Tenax reads version {expected}")]
    FormatVersion {
        /// The version this Tenax reads.
        expected: u32,
        /// The version the table declares.
        received: u32,
    },

    /// A record batch does not have the table's schema.
    #[error("a record batch does not have the evaluation table schema")]
    BatchSchema,

    /// A column does not have its Arrow type.
    #[error("column `{column}` does not have its Arrow type")]
    ColumnType {
        /// The column's field name.
        column: &'static str,
    },

    /// The rows of an evaluation are not contiguous and in order.
    #[error("evaluation {id} continues at row {found}; expected row {expected}")]
    RowOrder {
        /// The evaluation.
        id: EvaluationId,
        /// The next row number of the evaluation.
        expected: u64,
        /// The row number found.
        found: u64,
    },

    /// Rows of one evaluation record different seeds.
    #[error("evaluation {id} records more than one seed")]
    SeedMismatch {
        /// The evaluation.
        id: EvaluationId,
    },

    /// The input values are malformed.
    #[error(transparent)]
    Inputs(#[from] RequestDecodeError),

    /// The output values are malformed.
    #[error(transparent)]
    Outputs(#[from] OutputDecodeError),

    /// A row's outcome kind is unknown.
    #[error("row {row} of evaluation {id} has unknown outcome kind `{label}`")]
    UnknownOutcome {
        /// The evaluation.
        id: EvaluationId,
        /// Zero-based row within the evaluation.
        row: u64,
        /// The unknown label.
        label: String,
    },

    /// A failed row lacks a part its outcome kind needs.
    #[error("row {row} of evaluation {id} lacks its outcome {part}")]
    MissingOutcomePart {
        /// The evaluation.
        id: EvaluationId,
        /// Zero-based row within the evaluation.
        row: u64,
        /// The missing column.
        part: &'static str,
    },

    /// A row's output values disagree with its outcome: a success without
    /// values, or a failure with them.
    #[error("row {row} of evaluation {id} has output values inconsistent with its outcome")]
    OutcomeValues {
        /// The evaluation.
        id: EvaluationId,
        /// Zero-based row within the evaluation.
        row: u64,
    },

    /// A failed row names an invalid parameter or output.
    #[error("row {row} of evaluation {id} names an invalid port: {source}")]
    Subject {
        /// The evaluation.
        id: EvaluationId,
        /// Zero-based row within the evaluation.
        row: u64,
        /// Why the name is invalid.
        source: LabelError,
    },

    /// A row count does not fit this platform.
    #[error("row {row} of evaluation {id} records a count that does not fit this platform")]
    CountOverflow {
        /// The evaluation.
        id: EvaluationId,
        /// Zero-based row within the evaluation.
        row: u64,
    },

    /// The decoded evaluations do not form a valid table.
    #[error(transparent)]
    Table(#[from] EvaluationTableError),
}

/// Restores the session of an evaluation table schema and checks that the
/// schema is exactly that session's [`evaluation_schema`] in this format
/// version.
///
/// # Errors
///
/// Returns [`EvaluationDecodeError`] when the schema is not a Graphcal
/// session record, carries no valid `tenax` metadata, declares another format
/// version, or differs from the session's evaluation table schema.
pub fn decode_evaluation_schema(
    schema: &Schema,
) -> Result<Arc<SessionSchema>, EvaluationDecodeError> {
    let metadata = schema
        .metadata()
        .get(TENAX_METADATA_KEY)
        .and_then(|value| serde_json::from_str::<TenaxMetadata>(value).ok())
        .ok_or(EvaluationDecodeError::TenaxMetadata)?;
    if metadata.format_version != EVALUATION_FORMAT_VERSION {
        return Err(EvaluationDecodeError::FormatVersion {
            expected: EVALUATION_FORMAT_VERSION,
            received: metadata.format_version,
        });
    }
    let session = Arc::clone(RecordSchema::decode(schema)?.session());
    validate_schema(&evaluation_schema(&session), schema)?;
    Ok(session)
}

/// The rows of one evaluation decoded so far.
struct OpenEvaluation {
    id: EvaluationId,
    seed: u64,
    columns: Vec<InputColumn>,
    rows: Vec<RowOutcome>,
}

/// Decodes evaluation table record batches incrementally.
pub struct EvaluationDecoder {
    schema: SchemaRef,
    session: Arc<SessionSchema>,
    slots: Vec<OutputSlot>,
    table: EvaluationTable,
    open: Option<OpenEvaluation>,
}

impl EvaluationDecoder {
    pub fn new(schema: &Schema) -> Result<Self, EvaluationDecodeError> {
        let session = decode_evaluation_schema(schema)?;
        let slots = session
            .outputs()
            .iter()
            .filter_map(|output| session.output_slot(output.name().as_str()))
            .collect();
        Ok(Self {
            schema: Arc::new(schema.clone()),
            table: EvaluationTable::new(Arc::clone(&session)),
            session,
            slots,
            open: None,
        })
    }

    pub const fn session(&self) -> &Arc<SessionSchema> {
        &self.session
    }

    pub fn push(&mut self, batch: &RecordBatch) -> Result<(), EvaluationDecodeError> {
        if batch.schema().fields() != self.schema.fields() {
            return Err(EvaluationDecodeError::BatchSchema);
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let ids = column::<FixedSizeBinaryArray>(batch, 0, EVALUATION_ID_FIELD)?;
        let rows = column::<UInt64Array>(batch, 1, ROW_FIELD)?;
        let seeds = column::<UInt64Array>(batch, 2, SEED_FIELD)?;
        let inputs = column::<StructArray>(batch, 3, "inputs")?;
        let outputs = column::<StructArray>(batch, 4, "outputs")?;
        let outcome = Outcome::new(column::<StructArray>(batch, 5, OUTCOME_FIELD)?)?;

        let id_at = |row: usize| {
            let bytes: [u8; 16] = ids.value(row).try_into().unwrap_or_default();
            EvaluationId::new(u128::from_be_bytes(bytes))
        };
        let mut start = 0;
        while start < batch.num_rows() {
            let id = id_at(start);
            let end = (start..batch.num_rows())
                .find(|row| id_at(*row) != id)
                .unwrap_or_else(|| batch.num_rows());
            let length = end - start;
            let request = RequestBatch::from_struct_array(
                Arc::clone(&self.session),
                &inputs.slice(start, length),
            )?;
            let values = OutputValues::from_struct_array(
                Arc::clone(&self.session),
                &outputs.slice(start, length),
            )?;
            let seed = seeds.value(start);
            for offset in 0..length {
                let row = start + offset;
                let open = self.open_for(id, seed)?;
                let expected = u64::try_from(open.rows.len()).unwrap_or(u64::MAX);
                if rows.value(row) != expected {
                    return Err(EvaluationDecodeError::RowOrder {
                        id,
                        expected,
                        found: rows.value(row),
                    });
                }
                if open.seed != seeds.value(row) {
                    return Err(EvaluationDecodeError::SeedMismatch { id });
                }
                let row_values = values.row(offset).flatten().map(|row_values| {
                    self.slots
                        .iter()
                        .filter_map(|slot| row_values.get(*slot).map(ValueRef::to_owned_value))
                        .collect::<Vec<Value>>()
                });
                let outcome = outcome.row_outcome(&self.session, id, expected, row, row_values)?;
                self.open_for(id, seed)?.rows.push(outcome);
            }
            let open = self.open_for(id, seed)?;
            open.columns = if open.columns.is_empty() {
                request.columns().to_vec()
            } else {
                open.columns
                    .iter()
                    .zip(request.columns())
                    .map(|pair| concatenate(&<[&InputColumn; 2]>::from(pair)))
                    .collect()
            };
            start = end;
        }
        Ok(())
    }

    /// The open evaluation `id`, closing the previous one when `id` starts a
    /// new evaluation.
    fn open_for(
        &mut self,
        id: EvaluationId,
        seed: u64,
    ) -> Result<&mut OpenEvaluation, EvaluationDecodeError> {
        if self.open.as_ref().is_some_and(|open| open.id != id) {
            self.close()?;
        }
        Ok(self.open.get_or_insert_with(|| OpenEvaluation {
            id,
            seed,
            columns: Vec::new(),
            rows: Vec::new(),
        }))
    }

    fn close(&mut self) -> Result<(), EvaluationDecodeError> {
        let Some(open) = self.open.take() else {
            return Ok(());
        };
        let inputs = RequestBatch::new(Arc::clone(&self.session), open.columns)
            .map_err(RequestDecodeError::from)?;
        self.table.push(
            EvalRequest::new(open.id, open.seed, inputs),
            ChunkResult::from_rows(open.id, open.rows),
        )?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<EvaluationTable, EvaluationDecodeError> {
        self.close()?;
        Ok(self.table)
    }
}

fn column<'a, T: 'static>(
    batch: &'a RecordBatch,
    position: usize,
    name: &'static str,
) -> Result<&'a T, EvaluationDecodeError> {
    batch
        .column(position)
        .as_any()
        .downcast_ref::<T>()
        .ok_or(EvaluationDecodeError::ColumnType { column: name })
}

/// The `outcome` struct of one record batch.
struct Outcome<'a> {
    kinds: Vec<Option<OutcomeKind>>,
    labels: Vec<String>,
    subjects: &'a StringArray,
    messages: &'a StringArray,
    expected_counts: &'a UInt64Array,
    actual_counts: &'a UInt64Array,
}

impl<'a> Outcome<'a> {
    fn new(outcome: &'a StructArray) -> Result<Self, EvaluationDecodeError> {
        let child = |position: usize, name: &'static str| {
            outcome
                .columns()
                .get(position)
                .ok_or(EvaluationDecodeError::ColumnType { column: name })
        };
        let kinds = child(0, KIND_FIELD)?
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .ok_or(EvaluationDecodeError::ColumnType { column: KIND_FIELD })?;
        let dictionary = kinds
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(EvaluationDecodeError::ColumnType { column: KIND_FIELD })?;
        // Kinds are resolved by their dictionary value, never by code.
        let labels = kinds
            .keys()
            .iter()
            .map(|key| {
                key.and_then(|key| usize::try_from(key).ok())
                    .filter(|key| *key < dictionary.len())
                    .map_or_else(String::new, |key| dictionary.value(key).to_owned())
            })
            .collect::<Vec<_>>();
        let strings = |position, name| {
            child(position, name)?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(EvaluationDecodeError::ColumnType { column: name })
        };
        let counts = |position, name| {
            child(position, name)?
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or(EvaluationDecodeError::ColumnType { column: name })
        };
        Ok(Self {
            kinds: labels
                .iter()
                .map(|label| OutcomeKind::parse(label))
                .collect(),
            labels,
            subjects: strings(1, SUBJECT_FIELD)?,
            messages: strings(2, MESSAGE_FIELD)?,
            expected_counts: counts(3, EXPECTED_COUNT_FIELD)?,
            actual_counts: counts(4, ACTUAL_COUNT_FIELD)?,
        })
    }

    /// The outcome of batch row `position`, which is row `row` of evaluation
    /// `id` and has output `values` when present.
    fn row_outcome(
        &self,
        session: &SessionSchema,
        id: EvaluationId,
        row: u64,
        position: usize,
        values: Option<Vec<Value>>,
    ) -> Result<RowOutcome, EvaluationDecodeError> {
        let kind = self.kinds[position].ok_or_else(|| EvaluationDecodeError::UnknownOutcome {
            id,
            row,
            label: self.labels[position].clone(),
        })?;
        let inconsistent = || EvaluationDecodeError::OutcomeValues { id, row };
        if kind == OutcomeKind::Success {
            let values = values.ok_or_else(inconsistent)?;
            // A value that fails validation was already rejected by the
            // output decoder, so this only re-checks the session.
            return OutputRow::new(session, values)
                .map(RowOutcome::Success)
                .map_err(|_| inconsistent());
        }
        if values.is_some() {
            return Err(inconsistent());
        }
        let missing = |part| EvaluationDecodeError::MissingOutcomePart { id, row, part };
        let string = |array: &StringArray, part| {
            (!array.is_null(position))
                .then(|| array.value(position).to_owned())
                .ok_or_else(|| missing(part))
        };
        let count = |array: &UInt64Array, part| {
            if array.is_null(position) {
                return Err(missing(part));
            }
            usize::try_from(array.value(position))
                .map_err(|_| EvaluationDecodeError::CountOverflow { id, row })
        };
        let port = |name: String| {
            PortName::new(name).map_err(|source| EvaluationDecodeError::Subject { id, row, source })
        };
        let failure = match kind {
            // Successes returned above.
            OutcomeKind::Success | OutcomeKind::EvaluationError => {
                RowFailure::Model(ProtocolRowFailure::EvaluationError {
                    message: string(self.messages, MESSAGE_FIELD)?,
                })
            }
            OutcomeKind::AssertionFailed => {
                RowFailure::Model(ProtocolRowFailure::AssertionFailed {
                    assertion: string(self.subjects, SUBJECT_FIELD)?,
                    message: string(self.messages, MESSAGE_FIELD)?,
                })
            }
            OutcomeKind::InputOutOfDomain => {
                RowFailure::Model(ProtocolRowFailure::InputOutOfDomain {
                    parameter: port(string(self.subjects, SUBJECT_FIELD)?)?,
                    message: string(self.messages, MESSAGE_FIELD)?,
                })
            }
            OutcomeKind::Panic => {
                RowFailure::Panic(ModelPanic::Message(string(self.messages, MESSAGE_FIELD)?))
            }
            OutcomeKind::NonStringPanic => RowFailure::Panic(ModelPanic::NonStringPayload),
            OutcomeKind::InvalidInput => RowFailure::InvalidInput(SessionMismatch),
            OutcomeKind::OutputCountMismatch => {
                RowFailure::InvalidOutput(OutputRowError::CountMismatch {
                    expected: count(self.expected_counts, EXPECTED_COUNT_FIELD)?,
                    actual: count(self.actual_counts, ACTUAL_COUNT_FIELD)?,
                })
            }
            OutcomeKind::OutputKindMismatch => {
                RowFailure::InvalidOutput(OutputRowError::KindMismatch {
                    output: port(string(self.subjects, SUBJECT_FIELD)?)?,
                })
            }
        };
        Ok(RowOutcome::Failure(failure))
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{FiniteF64, KeyRef, VariantName, WireType};

    use super::*;
    use crate::ModelSchemaBuilder;

    fn session() -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .categorical("mode", "Mode", ["safe", "risky"])
            .unwrap()
            .quantity_output(
                "margin",
                graphcal_model_protocol::QuantityType::new(
                    graphcal_model_protocol::Dimension::dimensionless(),
                    None,
                )
                .unwrap(),
            )
            .unwrap()
            .key_output("verdict", "Mode", ["safe", "risky"])
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap()
    }

    fn request(session: &Arc<SessionSchema>, id: u128, rows: usize) -> EvalRequest {
        let WireType::Key(mode) = session.inputs()[1].wire() else {
            panic!("mode is a key");
        };
        let risky = mode
            .position(KeyRef::Variant(&VariantName::new("risky").unwrap()))
            .unwrap();
        let inputs = RequestBatch::new(
            Arc::clone(session),
            vec![
                InputColumn::Quantity(
                    (0..rows)
                        .map(|row| {
                            FiniteF64::new(0.1 * f64::from(u32::try_from(row).unwrap())).unwrap()
                        })
                        .collect(),
                ),
                InputColumn::Key(vec![risky; rows]),
            ],
        )
        .unwrap();
        EvalRequest::new(EvaluationId::new(id), u64::MAX - 3, inputs)
    }

    fn success(session: &SessionSchema, margin: f64) -> RowOutcome {
        RowOutcome::Success(
            OutputRow::new(
                session,
                vec![
                    Value::Quantity(FiniteF64::new(margin).unwrap()),
                    Value::Key(graphcal_model_protocol::KeyValue::Variant(
                        VariantName::new("safe").unwrap(),
                    )),
                    Value::Bool(margin < 0.0),
                ],
            )
            .unwrap(),
        )
    }

    fn every_failure() -> Vec<RowOutcome> {
        let port = |name| PortName::new(name).unwrap();
        [
            RowFailure::Model(ProtocolRowFailure::AssertionFailed {
                assertion: "limit".to_owned(),
                message: "limit exceeded".to_owned(),
            }),
            RowFailure::Model(ProtocolRowFailure::InputOutOfDomain {
                parameter: port("x"),
                message: "above 1".to_owned(),
            }),
            RowFailure::Model(ProtocolRowFailure::EvaluationError {
                message: "diverged".to_owned(),
            }),
            RowFailure::Panic(ModelPanic::Message("boom".to_owned())),
            RowFailure::Panic(ModelPanic::NonStringPayload),
            RowFailure::InvalidInput(SessionMismatch),
            RowFailure::InvalidOutput(OutputRowError::CountMismatch {
                expected: 3,
                actual: 1,
            }),
            RowFailure::InvalidOutput(OutputRowError::KindMismatch {
                output: port("margin"),
            }),
        ]
        .into_iter()
        .map(RowOutcome::Failure)
        .collect()
    }

    fn table() -> EvaluationTable {
        let session = session();
        let mut table = EvaluationTable::new(Arc::clone(&session));
        let mut first = vec![success(&session, 0.5)];
        first.extend(every_failure());
        first.push(success(&session, -1.5));
        table
            .push(
                request(&session, 1 << 100, first.len()),
                ChunkResult::new(EvaluationId::new(1 << 100), first).unwrap(),
            )
            .unwrap();
        table
            .push(
                request(&session, 7, 2),
                ChunkResult::new(
                    EvaluationId::new(7),
                    vec![success(&session, 2.0), success(&session, -0.5)],
                )
                .unwrap(),
            )
            .unwrap();
        table
    }

    #[test]
    fn tables_round_trip_with_every_outcome_kind() {
        let table = table();
        let batches = table.to_record_batches().unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 10);
        let schema = evaluation_schema(table.schema());
        let decoded = EvaluationTable::from_record_batches(&schema, batches).unwrap();
        assert_eq!(decoded, table);
    }

    /// A session reading indexed, algebraic, complex, and datetime outputs.
    fn rich_session() -> Arc<SessionSchema> {
        use graphcal_model_protocol::{
            AlgebraicType, Axis, AxisKind, ConstructorName, ConstructorType, DatetimeType,
            Dimension, FieldName, FieldType, IndexName, IndexedType, ModelInterface,
            OutputDescription, ParameterDescription, PortWire, QuantityType, SessionSpec,
            TimeScale, TypeName,
        };

        let name = |name: &str| PortName::new(name).unwrap();
        let length = QuantityType::new(Dimension::dimensionless(), None).unwrap();
        let phases = Axis::new(
            IndexName::new("Phase").unwrap(),
            AxisKind::Named {
                variants: vec![
                    VariantName::new("Climb").unwrap(),
                    VariantName::new("Cruise").unwrap(),
                ],
            },
        )
        .unwrap();
        let profile =
            WireType::Indexed(IndexedType::new(phases, WireType::Quantity(length.clone())));
        let shape = WireType::Algebraic(
            AlgebraicType::new(
                TypeName::new("Shape").unwrap(),
                vec![
                    ConstructorType::new(
                        ConstructorName::new("Circle").unwrap(),
                        vec![FieldType::new(
                            FieldName::new("r").unwrap(),
                            WireType::Quantity(length.clone()),
                        )],
                    ),
                    ConstructorType::new(ConstructorName::new("Dot").unwrap(), Vec::new()),
                ],
            )
            .unwrap(),
        );
        let output =
            |port: &str, wire| OutputDescription::new(name(port), port, PortWire::Supported(wire));
        let interface = ModelInterface::new(
            vec![
                ParameterDescription::new(
                    name("x"),
                    "Dimensionless",
                    PortWire::Supported(WireType::Quantity(length.clone())),
                    None,
                    false,
                )
                .unwrap(),
            ],
            vec![
                output("profile", profile),
                output("shape", shape),
                output("z", WireType::Complex(length)),
                output("at", WireType::Datetime(DatetimeType::new(TimeScale::Tai))),
            ],
        )
        .unwrap();
        Arc::new(
            interface
                .configure(
                    &SessionSpec::new(
                        vec![name("x")],
                        vec![name("profile"), name("shape"), name("z"), name("at")],
                    ),
                    graphcal_model_protocol::CanonicalUnitsOnly,
                )
                .unwrap(),
        )
    }

    #[test]
    fn rich_outputs_round_trip() {
        use graphcal_model_protocol::{ComplexValue, ConstructorPosition};

        let finite = |value: f64| FiniteF64::new(value).unwrap();
        let session = rich_session();
        let values = |offset: f64, constructor: usize| {
            vec![
                Value::Indexed(vec![
                    Value::Quantity(finite(offset)),
                    Value::Quantity(finite(offset + 1.0)),
                ]),
                Value::Algebraic {
                    constructor: ConstructorPosition::new(constructor),
                    fields: if constructor == 0 {
                        vec![Value::Quantity(finite(offset))]
                    } else {
                        Vec::new()
                    },
                },
                Value::Complex(ComplexValue {
                    re: finite(offset),
                    im: finite(-offset),
                }),
                Value::Datetime(finite(1.0e8 + offset)),
            ]
        };
        let inputs = RequestBatch::new(
            Arc::clone(&session),
            vec![InputColumn::Quantity(vec![
                finite(0.1),
                finite(0.2),
                finite(0.3),
            ])],
        )
        .unwrap();
        let rows = vec![
            RowOutcome::Success(OutputRow::new(&session, values(1.0, 0)).unwrap()),
            RowOutcome::Failure(RowFailure::Panic(ModelPanic::NonStringPayload)),
            RowOutcome::Success(OutputRow::new(&session, values(2.0, 1)).unwrap()),
        ];
        let mut table = EvaluationTable::new(Arc::clone(&session));
        table
            .push(
                EvalRequest::new(EvaluationId::new(3), 0, inputs),
                ChunkResult::new(EvaluationId::new(3), rows).unwrap(),
            )
            .unwrap();
        let schema = evaluation_schema(&session);
        let decoded =
            EvaluationTable::from_record_batches(&schema, table.to_record_batches().unwrap())
                .unwrap();
        assert_eq!(decoded, table);
    }

    #[test]
    fn evaluations_may_span_and_share_batches() {
        let table = table();
        let schema = evaluation_schema(table.schema());
        let batches = table.to_record_batches().unwrap();
        let joined = arrow_select::concat::concat_batches(&schema, &batches).unwrap();
        let split = vec![joined.slice(0, 4), joined.slice(4, 7), joined.slice(11, 1)];
        let decoded = EvaluationTable::from_record_batches(&schema, split).unwrap();
        assert_eq!(decoded, table);
    }

    #[test]
    fn decoding_rejects_out_of_order_rows_and_unknown_kinds() {
        let table = table();
        let schema = evaluation_schema(table.schema());
        let batches = table.to_record_batches().unwrap();

        let reversed = vec![batches[1].slice(1, 1), batches[1].slice(0, 1)];
        assert!(matches!(
            EvaluationTable::from_record_batches(&schema, reversed),
            Err(EvaluationDecodeError::RowOrder {
                expected: 0,
                found: 1,
                ..
            })
        ));

        let mut columns = batches[1].columns().to_vec();
        let outcome = columns[5]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .clone();
        let (fields, mut children, nulls) = outcome.into_parts();
        children[0] = Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![0, 0]),
                Arc::new(StringArray::from(vec!["exploded"])),
            )
            .unwrap(),
        );
        columns[5] = Arc::new(StructArray::new(fields, children, nulls));
        let unknown = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        assert!(matches!(
            EvaluationTable::from_record_batches(&schema, [unknown]),
            Err(EvaluationDecodeError::UnknownOutcome { row: 0, .. })
        ));
    }

    #[test]
    fn decoding_rejects_foreign_schemas() {
        let table = table();
        let request_schema =
            graphcal_model_protocol::arrow::request_schema(table.schema().as_ref());
        assert!(matches!(
            decode_evaluation_schema(&request_schema),
            Err(EvaluationDecodeError::TenaxMetadata)
        ));

        let schema = evaluation_schema(table.schema());
        let mut metadata = schema.metadata().clone();
        metadata.insert(
            TENAX_METADATA_KEY.to_owned(),
            r#"{"format_version":99,"tenax_version":"0"}"#.to_owned(),
        );
        assert!(matches!(
            decode_evaluation_schema(&schema.as_ref().clone().with_metadata(metadata)),
            Err(EvaluationDecodeError::FormatVersion { received: 99, .. })
        ));

        let mut fields = schema.fields().iter().cloned().collect::<Vec<_>>();
        fields[2] = Arc::new(Field::new(SEED_FIELD, DataType::Int64, false));
        let retyped = Schema::new_with_metadata(fields, schema.metadata().clone());
        assert!(matches!(
            decode_evaluation_schema(&retyped),
            Err(EvaluationDecodeError::Schema(_))
        ));
    }
}
