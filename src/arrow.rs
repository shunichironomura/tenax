//! Apache Arrow model-schema, request, and result boundary conversions.
//!
//! This module is available with the `arrow` crate feature. It keeps Arrow out
//! of the algorithmic core while defining the column and metadata contract used
//! by future process and network adapters.
//!
//! A model discovery [`ArrowSchema`] contains all inputs followed by all
//! outputs. Evaluation-request [`RecordBatch`] values contain input columns
//! followed by fixed-size evaluation-ID and seed context columns. Result
//! batches contain nullable output columns followed by evaluation-ID, outcome
//! status, and failure-message columns. Keeping request-specific context in
//! columns (rather than schema metadata) lets many batches share one standard
//! Arrow IPC stream, which is required by the stdio and future HTTP bindings.
//!
//! | Tenax value | Arrow data type |
//! | --- | --- |
//! | Continuous input | `Float64` |
//! | Integer input | `Int64` |
//! | Categorical input | `Dictionary<Int32, Utf8>` |
//! | Boolean output | `Boolean` |
//!
//! Unknown metadata keys are ignored by Tenax readers, so Graphcal and other
//! model languages can retain richer type, unit, diagnostic, and provenance
//! annotations. Unknown trailing result fields marked with the `extension`
//! role are also ignored. Tenax-owned metadata is namespaced with `tenax.` and
//! versioned by [`SCHEMA_VERSION_METADATA_KEY`].
//!
//! # Example
//!
//! ```
//! use tenax::{
//!     EvalRequest, EvaluationId, Feature, InputChunk, InputSchema, ModelSchema,
//!     OutputSchema,
//! };
//! use tenax::arrow::{ArrowSchema, EvalRequestRef, RecordBatch};
//!
//! let schema = ModelSchema::new(
//!     vec![InputSchema::continuous("load", 0.0, 1.0)?.with_unit("MW")?],
//!     vec![OutputSchema::boolean("failure")?],
//! )?;
//! let inputs = InputChunk::new(
//!     &schema,
//!     vec![Feature::continuous("load", vec![0.25, 0.75])?],
//! )?;
//! let request = EvalRequest::new(EvaluationId::new(7), 42, inputs);
//!
//! let discovery_schema = ArrowSchema::try_from(&schema)?;
//! let decoded_schema = ModelSchema::try_from(&discovery_schema)?;
//! assert_eq!(decoded_schema, schema);
//!
//! let batch = RecordBatch::try_from(EvalRequestRef::new(&schema, &request))?;
//! let decoded_request = EvalRequest::try_from((&schema, &batch))?;
//! assert_eq!(decoded_request, request);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, FixedSizeBinaryArray, Float64Array, Int32Array,
    Int64Array, StringArray, UInt8Array, UInt64Array,
};
use arrow_schema::{ArrowError as ArrowRsError, DataType, Field};
use thiserror::Error;

pub use arrow_array::RecordBatch;
pub use arrow_schema::Schema as ArrowSchema;

use crate::DataError;
use crate::data::{Feature, FeatureView};
use crate::evaluation::{
    ChunkResult, ChunkResultError, EvalRequest, EvaluationId, EvaluatorFailure,
    EvaluatorFailureKind, ModelError, ModelPanic, OutputRow, OutputRowError, OutputValue,
    RowFailure, RowOutcome,
};
use crate::input::{InputChunk, InputChunkError};
use crate::schema::{
    FeatureDomain, InputSchema, ModelFieldRole, ModelSchema, OutputKind, OutputSchema, SchemaError,
};

/// Metadata key containing the Tenax Arrow contract version.
pub const SCHEMA_VERSION_METADATA_KEY: &str = "tenax.schema.version";
/// Current value of [`SCHEMA_VERSION_METADATA_KEY`].
pub const SCHEMA_VERSION: &str = "2";
/// Field metadata key identifying a model or protocol field role.
pub const FIELD_ROLE_METADATA_KEY: &str = "tenax.field.role";
/// Value of [`FIELD_ROLE_METADATA_KEY`] for a model input.
pub const INPUT_FIELD_ROLE: &str = "input";
/// Value of [`FIELD_ROLE_METADATA_KEY`] for a model output.
pub const OUTPUT_FIELD_ROLE: &str = "output";
/// Value of [`FIELD_ROLE_METADATA_KEY`] for request/result context.
pub const CONTEXT_FIELD_ROLE: &str = "context";
/// Value of [`FIELD_ROLE_METADATA_KEY`] for per-row result state.
pub const OUTCOME_FIELD_ROLE: &str = "outcome";
/// Value of [`FIELD_ROLE_METADATA_KEY`] for optional peer-specific result data.
pub const EXTENSION_FIELD_ROLE: &str = "extension";
/// Metadata key identifying a context, outcome, or extension field's purpose.
pub const FIELD_KIND_METADATA_KEY: &str = "tenax.field.kind";
/// Optional input-field metadata key containing a unit annotation.
pub const FIELD_UNIT_METADATA_KEY: &str = "tenax.input.unit";
/// Input-field metadata key containing a continuous or integer lower bound.
pub const INPUT_LOWER_BOUND_METADATA_KEY: &str = "tenax.input.lower";
/// Input-field metadata key containing a continuous or integer upper bound.
pub const INPUT_UPPER_BOUND_METADATA_KEY: &str = "tenax.input.upper";
/// Categorical-input metadata key containing a JSON array of permitted values.
pub const INPUT_CATEGORIES_METADATA_KEY: &str = "tenax.input.categories";
/// Schema metadata key identifying the semantic kind of one record batch.
pub const BATCH_KIND_METADATA_KEY: &str = "tenax.batch.kind";
/// Value of [`BATCH_KIND_METADATA_KEY`] for model-schema discovery.
pub const MODEL_SCHEMA_BATCH_KIND: &str = "model_schema";
/// Value of [`BATCH_KIND_METADATA_KEY`] for an evaluation request.
pub const EVALUATION_REQUEST_BATCH_KIND: &str = "evaluation_request";
/// Value of [`BATCH_KIND_METADATA_KEY`] for an evaluation result.
pub const EVALUATION_RESULT_BATCH_KIND: &str = "evaluation_result";
/// Canonical name of the fixed-size binary evaluation-ID context field.
pub const EVALUATION_ID_FIELD_NAME: &str = "tenax.evaluation_id";
/// [`FIELD_KIND_METADATA_KEY`] value for an evaluation-ID context field.
pub const EVALUATION_ID_FIELD_KIND: &str = "evaluation_id";
/// Canonical name of the unsigned request-seed context field.
pub const EVALUATION_SEED_FIELD_NAME: &str = "tenax.evaluation_seed";
/// [`FIELD_KIND_METADATA_KEY`] value for a request-seed context field.
pub const EVALUATION_SEED_FIELD_KIND: &str = "evaluation_seed";
/// Canonical name of the unsigned per-row outcome-status field.
pub const OUTCOME_STATUS_FIELD_NAME: &str = "tenax.outcome_status";
/// [`FIELD_KIND_METADATA_KEY`] value for an outcome-status field.
pub const OUTCOME_STATUS_FIELD_KIND: &str = "outcome_status";
/// Canonical name of the nullable per-row failure-message field.
pub const FAILURE_MESSAGE_FIELD_NAME: &str = "tenax.failure_message";
/// [`FIELD_KIND_METADATA_KEY`] value for a failure-message field.
pub const FAILURE_MESSAGE_FIELD_KIND: &str = "failure_message";
/// Byte width of the big-endian `u128` evaluation-ID representation.
pub const EVALUATION_ID_BYTE_WIDTH: i32 = 16;
/// Result status code for a schema-valid successful row.
pub const OUTCOME_STATUS_SUCCESS: u8 = 0;
/// Result status code for a model-returned row failure.
pub const OUTCOME_STATUS_MODEL_ERROR: u8 = 1;
/// Result status code for a panic carrying a string diagnostic.
pub const OUTCOME_STATUS_PANIC: u8 = 2;
/// Result status code for a non-string panic payload.
pub const OUTCOME_STATUS_NON_STRING_PANIC: u8 = 3;
/// Result status code for an input rejected by the peer.
pub const OUTCOME_STATUS_INVALID_INPUT: u8 = 4;
/// Result status code for output values rejected by the peer.
pub const OUTCOME_STATUS_INVALID_OUTPUT: u8 = 5;

/// Semantic role encoded on an Arrow field at the interchange boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrowFieldRole {
    /// A declared model input.
    Input,
    /// A declared model output.
    Output,
    /// Evaluation identity or reproducibility context.
    Context,
    /// Per-row success/failure state.
    Outcome,
    /// Optional peer-specific result data ignored by Tenax.
    Extension,
}

impl std::fmt::Display for ArrowFieldRole {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => formatter.write_str(INPUT_FIELD_ROLE),
            Self::Output => formatter.write_str(OUTPUT_FIELD_ROLE),
            Self::Context => formatter.write_str(CONTEXT_FIELD_ROLE),
            Self::Outcome => formatter.write_str(OUTCOME_FIELD_ROLE),
            Self::Extension => formatter.write_str(EXTENSION_FIELD_ROLE),
        }
    }
}

/// Errors raised while converting between validated Tenax values and Arrow.
#[derive(Debug, Error)]
pub enum ArrowConversionError {
    /// Required schema-level metadata is absent.
    #[error("Arrow schema is missing required metadata '{key}'")]
    MissingSchemaMetadata {
        /// Missing metadata key.
        key: &'static str,
    },

    /// Required field-level metadata is absent.
    #[error("Arrow field '{field}' is missing required metadata '{key}'")]
    MissingFieldMetadata {
        /// Affected field name.
        field: String,
        /// Missing metadata key.
        key: &'static str,
    },

    /// The batch or discovery schema uses an unsupported contract version.
    #[error("unsupported Tenax Arrow schema version '{actual}'; expected '{expected}'")]
    UnsupportedSchemaVersion {
        /// Supported version.
        expected: &'static str,
        /// Version read from metadata.
        actual: String,
    },

    /// A schema or record batch has a different semantic kind.
    #[error("Arrow batch kind is '{actual}', but an '{expected}' batch is required")]
    UnexpectedBatchKind {
        /// Required batch kind.
        expected: &'static str,
        /// Batch kind read from metadata.
        actual: String,
    },

    /// A field role has an unknown boundary representation.
    #[error("Arrow field '{field}' has unknown role '{value}'")]
    InvalidFieldRole {
        /// Affected field name.
        field: String,
        /// Rejected metadata value.
        value: String,
    },

    /// A known field role occurs where a different role is required.
    #[error("Arrow field '{field}' has role '{actual}', but {expected} is required")]
    UnexpectedFieldRole {
        /// Affected field name.
        field: String,
        /// Required semantic role description.
        expected: &'static str,
        /// Parsed but misplaced role.
        actual: ArrowFieldRole,
    },

    /// A required scalar metadata value cannot be parsed.
    #[error(
        "Arrow field '{field}' metadata '{key}' has invalid value '{value}'; expected {expected}"
    )]
    InvalidFieldMetadata {
        /// Affected field name.
        field: String,
        /// Metadata key.
        key: &'static str,
        /// Rejected metadata value.
        value: String,
        /// Required representation.
        expected: &'static str,
    },

    /// Categorical-domain JSON metadata cannot be encoded or decoded.
    #[error("Arrow field '{field}' has invalid categorical-domain metadata: {source}")]
    CategoricalMetadata {
        /// Affected field name.
        field: String,
        /// JSON boundary error.
        #[source]
        source: serde_json::Error,
    },

    /// An input occurs after the first output in a discovery schema.
    #[error(
        "Arrow discovery schema input '{field}' occurs after an output; inputs must precede outputs"
    )]
    InputAfterOutput {
        /// Out-of-order input name.
        field: String,
    },

    /// Arrow field nullability disagrees with Tenax's no-missing-values contract.
    #[error("Arrow field '{field}' is nullable; this Tenax field must be non-nullable")]
    NullableField {
        /// Nullable field name.
        field: String,
    },

    /// A result output is non-nullable even though failed rows require nulls.
    #[error("Arrow result output field '{field}' is non-nullable; result outputs must be nullable")]
    NonNullableResultOutput {
        /// Non-nullable result output name.
        field: String,
    },

    /// A field has no supported mapping for its declared semantic role.
    #[error("Arrow {role} field '{field}' has unsupported data type {data_type:?}")]
    UnsupportedFieldType {
        /// Affected field name.
        field: String,
        /// Parsed field role.
        role: ModelFieldRole,
        /// Rejected Arrow data type.
        data_type: DataType,
    },

    /// The number of request columns differs from inputs plus protocol context.
    #[error(
        "Arrow evaluation request has {actual} fields, but {expected} input and context fields are required"
    )]
    RequestFieldCountMismatch {
        /// Number of required request fields.
        expected: usize,
        /// Number of Arrow fields.
        actual: usize,
    },

    /// A request field declaration differs from the expected model input.
    #[error(
        "Arrow request field {position} ('{field}') does not match the corresponding model input declaration"
    )]
    RequestInputMismatch {
        /// Zero-based model input position.
        position: usize,
        /// Arrow field name.
        field: String,
    },

    /// A required context or outcome field differs from the protocol contract.
    #[error(
        "Arrow field {position} ('{field}') does not match required protocol field '{expected}'"
    )]
    ProtocolFieldMismatch {
        /// Zero-based batch field position.
        position: usize,
        /// Supplied field name.
        field: String,
        /// Canonical protocol field name.
        expected: &'static str,
    },

    /// A result has too few fields for its outputs and required protocol data.
    #[error(
        "Arrow evaluation result has {actual} fields, but at least {expected} output and protocol fields are required"
    )]
    ResultFieldCountMismatch {
        /// Minimum number of required result fields.
        expected: usize,
        /// Number of supplied fields.
        actual: usize,
    },

    /// A result output field differs from the corresponding model declaration.
    #[error(
        "Arrow result field {position} ('{field}') does not match the corresponding model output declaration"
    )]
    ResultOutputMismatch {
        /// Zero-based model output position.
        position: usize,
        /// Supplied field name.
        field: String,
    },

    /// A record batch cannot represent a non-empty Tenax request or result.
    #[error("Arrow {kind} batch must contain at least one row")]
    EmptyBatch {
        /// Semantic batch kind.
        kind: &'static str,
    },

    /// Evaluation-ID context changes within one request/result batch.
    #[error("Arrow evaluation ID differs from the first row at row {row}")]
    InconsistentEvaluationId {
        /// First row with a different identifier.
        row: usize,
    },

    /// Request-seed context changes within one request batch.
    #[error("Arrow evaluation seed differs from the first row at row {row}")]
    InconsistentEvaluationSeed {
        /// First row with a different seed.
        row: usize,
    },

    /// A result uses an outcome status code unknown to this schema version.
    #[error("Arrow evaluation result has unknown outcome status {status} at row {row}")]
    UnknownOutcomeStatus {
        /// Affected row.
        row: usize,
        /// Rejected status code.
        status: u8,
    },

    /// Success/failure state and output nulls disagree.
    #[error("Arrow evaluation result row {row} has inconsistent outcome data: {message}")]
    InconsistentOutcome {
        /// Affected row.
        row: usize,
        /// Contract violation diagnostic.
        message: &'static str,
    },

    /// A successful native output row disagrees with the supplied model schema.
    #[error("evaluation result row {row} has invalid outputs: {source}")]
    InvalidOutputRow {
        /// Affected row.
        row: usize,
        /// Output schema violation.
        #[source]
        source: OutputRowError,
    },

    /// A supposedly non-nullable array contains null values.
    #[error("Arrow field '{field}' contains {null_count} null value(s)")]
    NullValues {
        /// Affected field name.
        field: String,
        /// Number of physical null values.
        null_count: usize,
    },

    /// A dictionary's value array contains a null.
    #[error("Arrow categorical field '{field}' has a null dictionary value at position {position}")]
    NullDictionaryValue {
        /// Affected field name.
        field: String,
        /// Zero-based dictionary position.
        position: usize,
    },

    /// A categorical dictionary repeats one logical category.
    #[error("Arrow categorical field '{field}' repeats dictionary value '{category}'")]
    DuplicateDictionaryValue {
        /// Affected field name.
        field: String,
        /// Duplicated category.
        category: String,
    },

    /// A categorical dictionary contains a value outside the model domain.
    #[error("Arrow categorical field '{field}' contains unknown dictionary value '{category}'")]
    UnknownDictionaryValue {
        /// Affected field name.
        field: String,
        /// Rejected category.
        category: String,
    },

    /// A categorical key cannot index its dictionary.
    #[error("Arrow categorical field '{field}' has invalid dictionary key {code} at row {row}")]
    InvalidDictionaryKey {
        /// Affected field name.
        field: String,
        /// Zero-based row position.
        row: usize,
        /// Rejected dictionary key.
        code: i32,
    },

    /// An Arrow array cannot be downcast to the concrete representation named by its field.
    #[error("Arrow field '{field}' cannot be read as its declared data type {data_type:?}")]
    InvalidArrayRepresentation {
        /// Affected field name.
        field: String,
        /// Declared Arrow data type.
        data_type: DataType,
    },

    /// Decoded result rows violate the non-empty chunk invariant.
    #[error(transparent)]
    ChunkResult(#[from] ChunkResultError),

    /// Arrow rejected a schema, array, or record-batch construction.
    #[error(transparent)]
    Arrow(#[from] ArrowRsError),

    /// Decoded model declarations violate a Tenax schema invariant.
    #[error(transparent)]
    Schema(#[from] SchemaError),

    /// Decoded feature values violate a native column invariant.
    #[error(transparent)]
    Data(#[from] DataError),

    /// Decoded request inputs violate their model schema.
    #[error(transparent)]
    InputChunk(#[from] InputChunkError),
}

/// A borrowed native evaluation request paired with the model schema needed to
/// encode complete Arrow field metadata.
///
/// This type supplies the context that [`EvalRequest`] intentionally does not
/// retain. Convert it with `RecordBatch::try_from`.
#[derive(Clone, Copy, Debug)]
pub struct EvalRequestRef<'data> {
    schema: &'data ModelSchema,
    request: &'data EvalRequest,
}

impl<'data> EvalRequestRef<'data> {
    /// Pairs a request with the model schema against which it will be validated
    /// and encoded.
    #[must_use]
    pub const fn new(schema: &'data ModelSchema, request: &'data EvalRequest) -> Self {
        Self { schema, request }
    }

    /// Returns the model schema used for conversion.
    #[must_use]
    pub const fn schema(&self) -> &'data ModelSchema {
        self.schema
    }

    /// Returns the native request used for conversion.
    #[must_use]
    pub const fn request(&self) -> &'data EvalRequest {
        self.request
    }
}

/// A borrowed native result chunk paired with its model schema for Arrow
/// encoding.
#[derive(Clone, Copy, Debug)]
pub struct ChunkResultRef<'data> {
    schema: &'data ModelSchema,
    result: &'data ChunkResult,
}

impl<'data> ChunkResultRef<'data> {
    /// Pairs a result with the model schema used to validate and encode it.
    #[must_use]
    pub const fn new(schema: &'data ModelSchema, result: &'data ChunkResult) -> Self {
        Self { schema, result }
    }

    /// Returns the model schema used for conversion.
    #[must_use]
    pub const fn schema(&self) -> &'data ModelSchema {
        self.schema
    }

    /// Returns the native result used for conversion.
    #[must_use]
    pub const fn result(&self) -> &'data ChunkResult {
        self.result
    }
}

impl TryFrom<&ModelSchema> for ArrowSchema {
    type Error = ArrowConversionError;

    fn try_from(schema: &ModelSchema) -> Result<Self, Self::Error> {
        let fields = schema
            .inputs()
            .iter()
            .map(input_field)
            .chain(
                schema
                    .outputs()
                    .iter()
                    .map(|output| Ok(output_field(output, false))),
            )
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new_with_metadata(
            fields,
            batch_metadata(MODEL_SCHEMA_BATCH_KIND),
        ))
    }
}

impl TryFrom<ModelSchema> for ArrowSchema {
    type Error = ArrowConversionError;

    fn try_from(schema: ModelSchema) -> Result<Self, Self::Error> {
        Self::try_from(&schema)
    }
}

impl TryFrom<&ArrowSchema> for ModelSchema {
    type Error = ArrowConversionError;

    fn try_from(schema: &ArrowSchema) -> Result<Self, Self::Error> {
        validate_version(schema.metadata())?;
        validate_batch_kind(schema.metadata(), MODEL_SCHEMA_BATCH_KIND)?;

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut saw_output = false;
        for field in schema.fields() {
            match field_role(field)? {
                ArrowFieldRole::Input => {
                    if saw_output {
                        return Err(ArrowConversionError::InputAfterOutput {
                            field: field.name().clone(),
                        });
                    }
                    inputs.push(input_schema(field)?);
                }
                ArrowFieldRole::Output => {
                    saw_output = true;
                    outputs.push(output_schema(field, false)?);
                }
                actual => {
                    return Err(ArrowConversionError::UnexpectedFieldRole {
                        field: field.name().clone(),
                        expected: "a model input or output role",
                        actual,
                    });
                }
            }
        }

        Self::new(inputs, outputs).map_err(Into::into)
    }
}

impl TryFrom<ArrowSchema> for ModelSchema {
    type Error = ArrowConversionError;

    fn try_from(schema: ArrowSchema) -> Result<Self, Self::Error> {
        Self::try_from(&schema)
    }
}

/// Builds the fixed Arrow schema shared by every request batch for `schema`.
pub fn evaluation_request_schema(
    schema: &ModelSchema,
) -> Result<ArrowSchema, ArrowConversionError> {
    let fields = schema
        .inputs()
        .iter()
        .map(input_field)
        .chain(
            [evaluation_id_field(), evaluation_seed_field()]
                .into_iter()
                .map(Ok),
        )
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ArrowSchema::new_with_metadata(
        fields,
        batch_metadata(EVALUATION_REQUEST_BATCH_KIND),
    ))
}

/// Validates a request-stream schema without waiting for its first batch.
pub fn validate_evaluation_request_schema(
    schema: &ModelSchema,
    arrow_schema: &ArrowSchema,
) -> Result<(), ArrowConversionError> {
    validate_version(arrow_schema.metadata())?;
    validate_batch_kind(arrow_schema.metadata(), EVALUATION_REQUEST_BATCH_KIND)?;
    let expected_count = schema.inputs().len() + 2;
    if arrow_schema.fields().len() != expected_count {
        return Err(ArrowConversionError::RequestFieldCountMismatch {
            expected: expected_count,
            actual: arrow_schema.fields().len(),
        });
    }
    for (position, (expected, field)) in schema
        .inputs()
        .iter()
        .zip(arrow_schema.fields())
        .enumerate()
    {
        if field_role(field)? != ArrowFieldRole::Input || input_schema(field)? != *expected {
            return Err(ArrowConversionError::RequestInputMismatch {
                position,
                field: field.name().clone(),
            });
        }
    }
    validate_protocol_field(
        arrow_schema.field(schema.inputs().len()),
        schema.inputs().len(),
        ProtocolField::EvaluationId,
    )?;
    validate_protocol_field(
        arrow_schema.field(schema.inputs().len() + 1),
        schema.inputs().len() + 1,
        ProtocolField::EvaluationSeed,
    )
}

/// Builds the standard prefix schema for every result batch for `schema`.
///
/// A peer may append fields carrying the [`EXTENSION_FIELD_ROLE`]. Tenax
/// ignores those fields while retaining the fixed output and outcome prefix.
#[must_use]
pub fn evaluation_result_schema(schema: &ModelSchema) -> ArrowSchema {
    let fields = schema
        .outputs()
        .iter()
        .map(|output| output_field(output, true))
        .chain([
            evaluation_id_field(),
            outcome_status_field(),
            failure_message_field(),
        ])
        .collect::<Vec<_>>();
    ArrowSchema::new_with_metadata(fields, batch_metadata(EVALUATION_RESULT_BATCH_KIND))
}

/// Validates a result-stream schema, including optional extension fields.
pub fn validate_evaluation_result_schema(
    schema: &ModelSchema,
    arrow_schema: &ArrowSchema,
) -> Result<(), ArrowConversionError> {
    validate_version(arrow_schema.metadata())?;
    validate_batch_kind(arrow_schema.metadata(), EVALUATION_RESULT_BATCH_KIND)?;
    let required_count = schema.outputs().len() + 3;
    if arrow_schema.fields().len() < required_count {
        return Err(ArrowConversionError::ResultFieldCountMismatch {
            expected: required_count,
            actual: arrow_schema.fields().len(),
        });
    }
    for (position, (expected, field)) in schema
        .outputs()
        .iter()
        .zip(arrow_schema.fields())
        .enumerate()
    {
        if field_role(field)? != ArrowFieldRole::Output || output_schema(field, true)? != *expected
        {
            return Err(ArrowConversionError::ResultOutputMismatch {
                position,
                field: field.name().clone(),
            });
        }
    }
    let protocol_start = schema.outputs().len();
    validate_protocol_field(
        arrow_schema.field(protocol_start),
        protocol_start,
        ProtocolField::EvaluationId,
    )?;
    validate_protocol_field(
        arrow_schema.field(protocol_start + 1),
        protocol_start + 1,
        ProtocolField::OutcomeStatus,
    )?;
    validate_protocol_field(
        arrow_schema.field(protocol_start + 2),
        protocol_start + 2,
        ProtocolField::FailureMessage,
    )?;
    for field in arrow_schema.fields().iter().skip(required_count) {
        let actual = field_role(field)?;
        if actual != ArrowFieldRole::Extension {
            return Err(ArrowConversionError::UnexpectedFieldRole {
                field: field.name().clone(),
                expected: "an extension role after the result protocol fields",
                actual,
            });
        }
    }
    Ok(())
}

impl TryFrom<EvalRequestRef<'_>> for RecordBatch {
    type Error = ArrowConversionError;

    fn try_from(value: EvalRequestRef<'_>) -> Result<Self, Self::Error> {
        let EvalRequestRef { schema, request } = value;
        request.inputs().validate_against(schema)?;
        let row_count = request.inputs().row_count();
        let mut columns = schema
            .inputs()
            .iter()
            .zip(request.inputs().feature_views())
            .map(|(input, values)| input_array(input, values))
            .collect::<Result<Vec<_>, _>>()?;
        columns.push(evaluation_id_array(request.id(), row_count)?);
        columns.push(Arc::new(UInt64Array::from(vec![request.seed(); row_count])));
        Self::try_new(Arc::new(evaluation_request_schema(schema)?), columns).map_err(Into::into)
    }
}

impl TryFrom<(&ModelSchema, &RecordBatch)> for EvalRequest {
    type Error = ArrowConversionError;

    fn try_from((schema, batch): (&ModelSchema, &RecordBatch)) -> Result<Self, Self::Error> {
        let arrow_schema = batch.schema_ref();
        validate_evaluation_request_schema(schema, arrow_schema)?;
        if batch.num_rows() == 0 {
            return Err(ArrowConversionError::EmptyBatch {
                kind: EVALUATION_REQUEST_BATCH_KIND,
            });
        }

        let context_start = schema.inputs().len();
        let id = evaluation_id_from_array(
            arrow_schema.field(context_start),
            batch.column(context_start),
        )?;
        let seed = evaluation_seed_from_array(
            arrow_schema.field(context_start + 1),
            batch.column(context_start + 1),
        )?;
        let features = schema
            .inputs()
            .iter()
            .zip(arrow_schema.fields())
            .zip(batch.columns())
            .map(|((expected, field), array)| feature_from_array(expected, field, array))
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = InputChunk::new(schema, features)?;
        Ok(Self::new(id, seed, inputs))
    }
}

impl TryFrom<(&ModelSchema, RecordBatch)> for EvalRequest {
    type Error = ArrowConversionError;

    fn try_from((schema, batch): (&ModelSchema, RecordBatch)) -> Result<Self, Self::Error> {
        Self::try_from((schema, &batch))
    }
}

impl TryFrom<ChunkResultRef<'_>> for RecordBatch {
    type Error = ArrowConversionError;

    fn try_from(value: ChunkResultRef<'_>) -> Result<Self, Self::Error> {
        let ChunkResultRef { schema, result } = value;
        for (row, outcome) in result.rows().iter().enumerate() {
            if let RowOutcome::Success(outputs) = outcome {
                outputs
                    .validate_against(schema)
                    .map_err(|source| ArrowConversionError::InvalidOutputRow { row, source })?;
            }
        }

        let mut columns = (0..schema.outputs().len())
            .map(|position| result_output_array(result, position))
            .collect::<Vec<_>>();
        columns.push(evaluation_id_array(result.id(), result.rows().len())?);
        let (statuses, messages): (Vec<_>, Vec<_>) = result.rows().iter().map(wire_outcome).unzip();
        columns.push(Arc::new(UInt8Array::from(statuses)));
        columns.push(Arc::new(StringArray::from(messages)));
        Self::try_new(Arc::new(evaluation_result_schema(schema)), columns).map_err(Into::into)
    }
}

impl TryFrom<(&ModelSchema, &RecordBatch)> for ChunkResult {
    type Error = ArrowConversionError;

    fn try_from((schema, batch): (&ModelSchema, &RecordBatch)) -> Result<Self, Self::Error> {
        let arrow_schema = batch.schema_ref();
        validate_evaluation_result_schema(schema, arrow_schema)?;
        if batch.num_rows() == 0 {
            return Err(ArrowConversionError::EmptyBatch {
                kind: EVALUATION_RESULT_BATCH_KIND,
            });
        }
        let protocol_start = schema.outputs().len();
        let id = evaluation_id_from_array(
            arrow_schema.field(protocol_start),
            batch.column(protocol_start),
        )?;
        let rows = result_rows(schema, batch)?;
        Self::new(id, rows).map_err(Into::into)
    }
}

impl TryFrom<(&ModelSchema, RecordBatch)> for ChunkResult {
    type Error = ArrowConversionError;

    fn try_from((schema, batch): (&ModelSchema, RecordBatch)) -> Result<Self, Self::Error> {
        Self::try_from((schema, &batch))
    }
}

fn input_field(input: &InputSchema) -> Result<Field, ArrowConversionError> {
    let mut metadata = HashMap::from([(
        FIELD_ROLE_METADATA_KEY.to_owned(),
        INPUT_FIELD_ROLE.to_owned(),
    )]);
    let data_type = match input.domain() {
        FeatureDomain::Continuous(domain) => {
            metadata.insert(
                INPUT_LOWER_BOUND_METADATA_KEY.to_owned(),
                domain.lower().to_string(),
            );
            metadata.insert(
                INPUT_UPPER_BOUND_METADATA_KEY.to_owned(),
                domain.upper().to_string(),
            );
            DataType::Float64
        }
        FeatureDomain::Integer(domain) => {
            metadata.insert(
                INPUT_LOWER_BOUND_METADATA_KEY.to_owned(),
                domain.lower().to_string(),
            );
            metadata.insert(
                INPUT_UPPER_BOUND_METADATA_KEY.to_owned(),
                domain.upper().to_string(),
            );
            DataType::Int64
        }
        FeatureDomain::Categorical(domain) => {
            let categories = serde_json::to_string(domain.categories()).map_err(|source| {
                ArrowConversionError::CategoricalMetadata {
                    field: input.name().to_owned(),
                    source,
                }
            })?;
            metadata.insert(INPUT_CATEGORIES_METADATA_KEY.to_owned(), categories);
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
        }
    };
    if let Some(unit) = input.unit() {
        metadata.insert(FIELD_UNIT_METADATA_KEY.to_owned(), unit.to_owned());
    }
    Ok(Field::new(input.name(), data_type, false).with_metadata(metadata))
}

fn output_field(output: &OutputSchema, nullable: bool) -> Field {
    let data_type = match output.kind() {
        OutputKind::Boolean => DataType::Boolean,
    };
    Field::new(output.name(), data_type, nullable).with_metadata(HashMap::from([(
        FIELD_ROLE_METADATA_KEY.to_owned(),
        OUTPUT_FIELD_ROLE.to_owned(),
    )]))
}

#[derive(Clone, Copy, Debug)]
enum ProtocolField {
    EvaluationId,
    EvaluationSeed,
    OutcomeStatus,
    FailureMessage,
}

impl ProtocolField {
    const fn name(self) -> &'static str {
        match self {
            Self::EvaluationId => EVALUATION_ID_FIELD_NAME,
            Self::EvaluationSeed => EVALUATION_SEED_FIELD_NAME,
            Self::OutcomeStatus => OUTCOME_STATUS_FIELD_NAME,
            Self::FailureMessage => FAILURE_MESSAGE_FIELD_NAME,
        }
    }

    const fn role(self) -> &'static str {
        match self {
            Self::EvaluationId | Self::EvaluationSeed => CONTEXT_FIELD_ROLE,
            Self::OutcomeStatus | Self::FailureMessage => OUTCOME_FIELD_ROLE,
        }
    }

    const fn kind(self) -> &'static str {
        match self {
            Self::EvaluationId => EVALUATION_ID_FIELD_KIND,
            Self::EvaluationSeed => EVALUATION_SEED_FIELD_KIND,
            Self::OutcomeStatus => OUTCOME_STATUS_FIELD_KIND,
            Self::FailureMessage => FAILURE_MESSAGE_FIELD_KIND,
        }
    }

    const fn nullable(self) -> bool {
        matches!(self, Self::FailureMessage)
    }

    const fn data_type(self) -> DataType {
        match self {
            Self::EvaluationId => DataType::FixedSizeBinary(EVALUATION_ID_BYTE_WIDTH),
            Self::EvaluationSeed => DataType::UInt64,
            Self::OutcomeStatus => DataType::UInt8,
            Self::FailureMessage => DataType::Utf8,
        }
    }

    fn arrow_field(self) -> Field {
        Field::new(self.name(), self.data_type(), self.nullable()).with_metadata(HashMap::from([
            (FIELD_ROLE_METADATA_KEY.to_owned(), self.role().to_owned()),
            (FIELD_KIND_METADATA_KEY.to_owned(), self.kind().to_owned()),
        ]))
    }
}

fn evaluation_id_field() -> Field {
    ProtocolField::EvaluationId.arrow_field()
}

fn evaluation_seed_field() -> Field {
    ProtocolField::EvaluationSeed.arrow_field()
}

fn outcome_status_field() -> Field {
    ProtocolField::OutcomeStatus.arrow_field()
}

fn failure_message_field() -> Field {
    ProtocolField::FailureMessage.arrow_field()
}

fn validate_protocol_field(
    actual: &Field,
    position: usize,
    protocol_field: ProtocolField,
) -> Result<(), ArrowConversionError> {
    let expected = protocol_field.arrow_field();
    let role_matches = field_role(actual)? == field_role(&expected)?;
    let kind_matches = required_field_metadata(actual, FIELD_KIND_METADATA_KEY)?
        == required_field_metadata(&expected, FIELD_KIND_METADATA_KEY)?;
    if actual.name() == expected.name()
        && actual.data_type() == expected.data_type()
        && actual.is_nullable() == expected.is_nullable()
        && role_matches
        && kind_matches
    {
        Ok(())
    } else {
        Err(ArrowConversionError::ProtocolFieldMismatch {
            position,
            field: actual.name().clone(),
            expected: protocol_field.name(),
        })
    }
}

fn input_schema(field: &Field) -> Result<InputSchema, ArrowConversionError> {
    validate_non_nullable(field)?;
    let input = match field.data_type() {
        DataType::Float64 => InputSchema::continuous(
            field.name(),
            parse_field_metadata::<f64>(field, INPUT_LOWER_BOUND_METADATA_KEY, "a finite f64")?,
            parse_field_metadata::<f64>(field, INPUT_UPPER_BOUND_METADATA_KEY, "a finite f64")?,
        )?,
        DataType::Int64 => InputSchema::integer(
            field.name(),
            parse_field_metadata::<i64>(field, INPUT_LOWER_BOUND_METADATA_KEY, "an i64")?,
            parse_field_metadata::<i64>(field, INPUT_UPPER_BOUND_METADATA_KEY, "an i64")?,
        )?,
        DataType::Dictionary(key, value)
            if key.as_ref() == &DataType::Int32 && value.as_ref() == &DataType::Utf8 =>
        {
            let encoded = required_field_metadata(field, INPUT_CATEGORIES_METADATA_KEY)?;
            let categories = serde_json::from_str::<Vec<String>>(encoded).map_err(|source| {
                ArrowConversionError::CategoricalMetadata {
                    field: field.name().clone(),
                    source,
                }
            })?;
            InputSchema::categorical(field.name(), categories)?
        }
        data_type => {
            return Err(ArrowConversionError::UnsupportedFieldType {
                field: field.name().clone(),
                role: ModelFieldRole::Input,
                data_type: data_type.clone(),
            });
        }
    };

    match field.metadata().get(FIELD_UNIT_METADATA_KEY) {
        Some(unit) => input.with_unit(unit.clone()).map_err(Into::into),
        None => Ok(input),
    }
}

fn output_schema(field: &Field, result_output: bool) -> Result<OutputSchema, ArrowConversionError> {
    if result_output {
        if !field.is_nullable() {
            return Err(ArrowConversionError::NonNullableResultOutput {
                field: field.name().clone(),
            });
        }
    } else {
        validate_non_nullable(field)?;
    }
    match field.data_type() {
        DataType::Boolean => OutputSchema::boolean(field.name()).map_err(Into::into),
        data_type => Err(ArrowConversionError::UnsupportedFieldType {
            field: field.name().clone(),
            role: ModelFieldRole::Output,
            data_type: data_type.clone(),
        }),
    }
}

fn input_array(
    input: &InputSchema,
    values: FeatureView<'_>,
) -> Result<ArrayRef, ArrowConversionError> {
    match (input.domain(), values) {
        (FeatureDomain::Continuous(_), FeatureView::Continuous(values)) => {
            Ok(Arc::new(Float64Array::from(values.to_vec())))
        }
        (FeatureDomain::Integer(_), FeatureView::Integer(values)) => {
            Ok(Arc::new(Int64Array::from(values.to_vec())))
        }
        (FeatureDomain::Categorical(domain), FeatureView::Categorical(values)) => {
            let codes = values
                .codes()
                .iter()
                .copied()
                .enumerate()
                .map(|(row, code)| {
                    let category = values.category(code).ok_or_else(|| {
                        ArrowConversionError::InvalidDictionaryKey {
                            field: input.name().to_owned(),
                            row,
                            code,
                        }
                    })?;
                    domain
                        .categories()
                        .binary_search_by(|candidate| candidate.as_str().cmp(category))
                        .ok()
                        .and_then(|position| i32::try_from(position).ok())
                        .ok_or_else(|| ArrowConversionError::UnknownDictionaryValue {
                            field: input.name().to_owned(),
                            category: category.to_owned(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let dictionary =
                StringArray::from_iter_values(domain.categories().iter().map(String::as_str));
            let array = DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(codes),
                Arc::new(dictionary),
            )?;
            Ok(Arc::new(array))
        }
        (_, actual) => Err(ArrowConversionError::InvalidArrayRepresentation {
            field: input.name().to_owned(),
            data_type: data_type_for_feature(actual),
        }),
    }
}

fn feature_from_array(
    input: &InputSchema,
    field: &Field,
    array: &ArrayRef,
) -> Result<Feature, ArrowConversionError> {
    reject_nulls(field, array.as_ref())?;
    match input.domain() {
        FeatureDomain::Continuous(_) => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| invalid_array(field))?;
            Feature::continuous(input.name(), values.values().to_vec()).map_err(Into::into)
        }
        FeatureDomain::Integer(_) => {
            let values = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| invalid_array(field))?;
            Feature::integer(input.name(), values.values().to_vec()).map_err(Into::into)
        }
        FeatureDomain::Categorical(domain) => {
            let values = array
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| invalid_array(field))?;
            categorical_feature(input.name(), domain.categories(), field, values)
        }
    }
}

fn categorical_feature(
    name: &str,
    domain: &[String],
    field: &Field,
    array: &DictionaryArray<Int32Type>,
) -> Result<Feature, ArrowConversionError> {
    let dictionary = array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| invalid_array(field))?;
    if let Some(position) = (0..dictionary.len()).find(|position| dictionary.is_null(*position)) {
        return Err(ArrowConversionError::NullDictionaryValue {
            field: field.name().clone(),
            position,
        });
    }

    let mut seen = HashSet::with_capacity(dictionary.len());
    let mut categories = Vec::with_capacity(dictionary.len());
    for position in 0..dictionary.len() {
        let category = dictionary.value(position);
        if !seen.insert(category) {
            return Err(ArrowConversionError::DuplicateDictionaryValue {
                field: field.name().clone(),
                category: category.to_owned(),
            });
        }
        if domain
            .binary_search_by(|candidate| candidate.as_str().cmp(category))
            .is_err()
        {
            return Err(ArrowConversionError::UnknownDictionaryValue {
                field: field.name().clone(),
                category: category.to_owned(),
            });
        }
        categories.push(category.to_owned());
    }

    Feature::categorical_from_dictionary(name, array.keys().values().to_vec(), categories)
        .map_err(Into::into)
}

fn evaluation_id_array(
    id: EvaluationId,
    row_count: usize,
) -> Result<ArrayRef, ArrowConversionError> {
    let bytes = id.get().to_be_bytes();
    let array = FixedSizeBinaryArray::try_from_iter((0..row_count).map(|_| bytes))?;
    Ok(Arc::new(array))
}

fn evaluation_id_from_array(
    field: &Field,
    array: &ArrayRef,
) -> Result<EvaluationId, ArrowConversionError> {
    reject_nulls(field, array.as_ref())?;
    let values = array
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| invalid_array(field))?;
    let first_bytes: [u8; EVALUATION_ID_BYTE_WIDTH as usize] = values
        .value(0)
        .try_into()
        .map_err(|_| invalid_array(field))?;
    let first = EvaluationId::new(u128::from_be_bytes(first_bytes));
    if let Some(row) = (1..values.len()).find(|row| values.value(*row) != first_bytes) {
        return Err(ArrowConversionError::InconsistentEvaluationId { row });
    }
    Ok(first)
}

fn evaluation_seed_from_array(
    field: &Field,
    array: &ArrayRef,
) -> Result<u64, ArrowConversionError> {
    reject_nulls(field, array.as_ref())?;
    let values = array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| invalid_array(field))?;
    let first = values.value(0);
    if let Some(row) = values.values().iter().position(|value| *value != first) {
        return Err(ArrowConversionError::InconsistentEvaluationSeed { row });
    }
    Ok(first)
}

fn result_output_array(result: &ChunkResult, position: usize) -> ArrayRef {
    let values = result
        .rows()
        .iter()
        .map(|outcome| match outcome {
            RowOutcome::Success(outputs) => match outputs.values()[position] {
                OutputValue::Boolean(value) => Some(value),
            },
            RowOutcome::Failure(_) => None,
        })
        .collect::<Vec<_>>();
    Arc::new(BooleanArray::from(values))
}

fn wire_outcome(outcome: &RowOutcome) -> (u8, Option<String>) {
    match outcome {
        RowOutcome::Success(_) => (OUTCOME_STATUS_SUCCESS, None),
        RowOutcome::Failure(RowFailure::Model(error)) => {
            (OUTCOME_STATUS_MODEL_ERROR, Some(error.message().to_owned()))
        }
        RowOutcome::Failure(RowFailure::Panic(ModelPanic::Message(message))) => {
            (OUTCOME_STATUS_PANIC, Some(message.clone()))
        }
        RowOutcome::Failure(RowFailure::Panic(ModelPanic::NonStringPayload)) => {
            (OUTCOME_STATUS_NON_STRING_PANIC, None)
        }
        RowOutcome::Failure(RowFailure::InvalidInput(error)) => {
            (OUTCOME_STATUS_INVALID_INPUT, Some(error.to_string()))
        }
        RowOutcome::Failure(RowFailure::InvalidOutput(error)) => {
            (OUTCOME_STATUS_INVALID_OUTPUT, Some(error.to_string()))
        }
        RowOutcome::Failure(RowFailure::Evaluator(error)) => match error.kind() {
            EvaluatorFailureKind::InvalidInput => (
                OUTCOME_STATUS_INVALID_INPUT,
                Some(error.message().to_owned()),
            ),
            EvaluatorFailureKind::InvalidOutput => (
                OUTCOME_STATUS_INVALID_OUTPUT,
                Some(error.message().to_owned()),
            ),
        },
    }
}

fn result_rows(
    schema: &ModelSchema,
    batch: &RecordBatch,
) -> Result<Vec<RowOutcome>, ArrowConversionError> {
    let arrow_schema = batch.schema();
    let output_arrays = schema
        .outputs()
        .iter()
        .enumerate()
        .map(|(position, _)| {
            batch
                .column(position)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| invalid_array(arrow_schema.field(position)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let protocol_start = schema.outputs().len();
    let status_field = arrow_schema.field(protocol_start + 1);
    let statuses = batch
        .column(protocol_start + 1)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| invalid_array(status_field))?;
    reject_nulls(status_field, statuses)?;
    let message_field = arrow_schema.field(protocol_start + 2);
    let messages = batch
        .column(protocol_start + 2)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| invalid_array(message_field))?;

    (0..batch.num_rows())
        .map(|row| {
            let status = statuses.value(row);
            if status == OUTCOME_STATUS_SUCCESS {
                if messages.is_valid(row) {
                    return Err(ArrowConversionError::InconsistentOutcome {
                        row,
                        message: "a successful row must not carry a failure message",
                    });
                }
                let values = output_arrays
                    .iter()
                    .map(|array| {
                        if array.is_null(row) {
                            Err(ArrowConversionError::InconsistentOutcome {
                                row,
                                message: "a successful row must contain every declared output",
                            })
                        } else {
                            Ok(OutputValue::Boolean(array.value(row)))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = OutputRow::new(schema, values)
                    .map_err(|source| ArrowConversionError::InvalidOutputRow { row, source })?;
                return Ok(RowOutcome::Success(outputs));
            }

            if output_arrays.iter().any(|array| array.is_valid(row)) {
                return Err(ArrowConversionError::InconsistentOutcome {
                    row,
                    message: "a failed row must represent every model output as null",
                });
            }
            let message = messages
                .is_valid(row)
                .then(|| messages.value(row).to_owned());
            let failure = match status {
                OUTCOME_STATUS_MODEL_ERROR => {
                    RowFailure::Model(ModelError::new(required_failure_message(row, message)?))
                }
                OUTCOME_STATUS_PANIC => {
                    RowFailure::Panic(ModelPanic::Message(required_failure_message(row, message)?))
                }
                OUTCOME_STATUS_NON_STRING_PANIC => {
                    if message.is_some() {
                        return Err(ArrowConversionError::InconsistentOutcome {
                            row,
                            message: "a non-string panic must not carry a string message",
                        });
                    }
                    RowFailure::Panic(ModelPanic::NonStringPayload)
                }
                OUTCOME_STATUS_INVALID_INPUT => RowFailure::Evaluator(EvaluatorFailure::new(
                    EvaluatorFailureKind::InvalidInput,
                    required_failure_message(row, message)?,
                )),
                OUTCOME_STATUS_INVALID_OUTPUT => RowFailure::Evaluator(EvaluatorFailure::new(
                    EvaluatorFailureKind::InvalidOutput,
                    required_failure_message(row, message)?,
                )),
                status => {
                    return Err(ArrowConversionError::UnknownOutcomeStatus { row, status });
                }
            };
            Ok(RowOutcome::Failure(failure))
        })
        .collect()
}

fn required_failure_message(
    row: usize,
    message: Option<String>,
) -> Result<String, ArrowConversionError> {
    message.ok_or(ArrowConversionError::InconsistentOutcome {
        row,
        message: "this failed-row status requires a failure message",
    })
}

fn field_role(field: &Field) -> Result<ArrowFieldRole, ArrowConversionError> {
    match required_field_metadata(field, FIELD_ROLE_METADATA_KEY)? {
        INPUT_FIELD_ROLE => Ok(ArrowFieldRole::Input),
        OUTPUT_FIELD_ROLE => Ok(ArrowFieldRole::Output),
        CONTEXT_FIELD_ROLE => Ok(ArrowFieldRole::Context),
        OUTCOME_FIELD_ROLE => Ok(ArrowFieldRole::Outcome),
        EXTENSION_FIELD_ROLE => Ok(ArrowFieldRole::Extension),
        value => Err(ArrowConversionError::InvalidFieldRole {
            field: field.name().clone(),
            value: value.to_owned(),
        }),
    }
}

fn parse_field_metadata<T>(
    field: &Field,
    key: &'static str,
    expected: &'static str,
) -> Result<T, ArrowConversionError>
where
    T: std::str::FromStr,
{
    let value = required_field_metadata(field, key)?;
    value
        .parse()
        .map_err(|_| ArrowConversionError::InvalidFieldMetadata {
            field: field.name().clone(),
            key,
            value: value.to_owned(),
            expected,
        })
}

fn required_field_metadata<'field>(
    field: &'field Field,
    key: &'static str,
) -> Result<&'field str, ArrowConversionError> {
    field
        .metadata()
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| ArrowConversionError::MissingFieldMetadata {
            field: field.name().clone(),
            key,
        })
}

fn required_schema_metadata<'metadata>(
    metadata: &'metadata HashMap<String, String>,
    key: &'static str,
) -> Result<&'metadata str, ArrowConversionError> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or(ArrowConversionError::MissingSchemaMetadata { key })
}

fn validate_version(metadata: &HashMap<String, String>) -> Result<(), ArrowConversionError> {
    let actual = required_schema_metadata(metadata, SCHEMA_VERSION_METADATA_KEY)?;
    if actual == SCHEMA_VERSION {
        Ok(())
    } else {
        Err(ArrowConversionError::UnsupportedSchemaVersion {
            expected: SCHEMA_VERSION,
            actual: actual.to_owned(),
        })
    }
}

fn validate_batch_kind(
    metadata: &HashMap<String, String>,
    expected: &'static str,
) -> Result<(), ArrowConversionError> {
    let actual = required_schema_metadata(metadata, BATCH_KIND_METADATA_KEY)?;
    if actual == expected {
        Ok(())
    } else {
        Err(ArrowConversionError::UnexpectedBatchKind {
            expected,
            actual: actual.to_owned(),
        })
    }
}

fn validate_non_nullable(field: &Field) -> Result<(), ArrowConversionError> {
    if field.is_nullable() {
        Err(ArrowConversionError::NullableField {
            field: field.name().clone(),
        })
    } else {
        Ok(())
    }
}

fn reject_nulls(field: &Field, array: &dyn Array) -> Result<(), ArrowConversionError> {
    let null_count = array.null_count();
    if null_count == 0 {
        Ok(())
    } else {
        Err(ArrowConversionError::NullValues {
            field: field.name().clone(),
            null_count,
        })
    }
}

fn batch_metadata(kind: &'static str) -> HashMap<String, String> {
    HashMap::from([
        (
            SCHEMA_VERSION_METADATA_KEY.to_owned(),
            SCHEMA_VERSION.to_owned(),
        ),
        (BATCH_KIND_METADATA_KEY.to_owned(), kind.to_owned()),
    ])
}

fn invalid_array(field: &Field) -> ArrowConversionError {
    ArrowConversionError::InvalidArrayRepresentation {
        field: field.name().clone(),
        data_type: field.data_type().clone(),
    }
}

fn data_type_for_feature(values: FeatureView<'_>) -> DataType {
    match values {
        FeatureView::Continuous(_) => DataType::Float64,
        FeatureView::Integer(_) => DataType::Int64,
        FeatureView::Categorical(_) => {
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BooleanArray, Int32Array};

    fn mixed_schema() -> ModelSchema {
        ModelSchema::new(
            vec![
                InputSchema::continuous("load", 0.0, 1.0)
                    .unwrap()
                    .with_unit("MW")
                    .unwrap(),
                InputSchema::integer("count", -2, 2).unwrap(),
                InputSchema::categorical(
                    "mode",
                    vec!["safe".to_owned(), "risky".to_owned(), "unused".to_owned()],
                )
                .unwrap(),
            ],
            vec![OutputSchema::boolean("failure").unwrap()],
        )
        .unwrap()
    }

    fn mixed_request(schema: &ModelSchema) -> EvalRequest {
        let inputs = InputChunk::new(
            schema,
            vec![
                Feature::continuous("load", vec![0.25, 0.75]).unwrap(),
                Feature::integer("count", vec![-1, 2]).unwrap(),
                Feature::categorical("mode", vec!["risky".to_owned(), "safe".to_owned()]).unwrap(),
            ],
        )
        .unwrap();
        EvalRequest::new(
            EvaluationId::new(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210),
            u64::MAX - 7,
            inputs,
        )
    }

    #[test]
    fn model_schema_round_trip_preserves_domains_roles_and_units() {
        let native = mixed_schema();
        let arrow = ArrowSchema::try_from(&native).unwrap();

        assert_eq!(arrow.fields().len(), 4);
        assert_eq!(arrow.field(0).data_type(), &DataType::Float64);
        assert_eq!(
            arrow.field(2).data_type(),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
        );
        assert_eq!(
            arrow.field(0).metadata().get(FIELD_ROLE_METADATA_KEY),
            Some(&INPUT_FIELD_ROLE.to_owned())
        );
        assert_eq!(
            arrow.field(0).metadata().get(FIELD_UNIT_METADATA_KEY),
            Some(&"MW".to_owned())
        );
        assert_eq!(
            arrow.field(3).metadata().get(FIELD_ROLE_METADATA_KEY),
            Some(&OUTPUT_FIELD_ROLE.to_owned())
        );
        assert!(arrow.fields().iter().all(|field| !field.is_nullable()));
        assert_eq!(ModelSchema::try_from(&arrow).unwrap(), native);
    }

    #[test]
    fn evaluation_request_round_trip_preserves_columns_id_seed_and_full_dictionary() {
        let schema = mixed_schema();
        let native = mixed_request(&schema);
        let arrow = RecordBatch::try_from(EvalRequestRef::new(&schema, &native)).unwrap();

        assert_eq!(arrow.num_rows(), 2);
        assert_eq!(arrow.num_columns(), 5);
        assert_eq!(
            arrow.schema().metadata().get(BATCH_KIND_METADATA_KEY),
            Some(&EVALUATION_REQUEST_BATCH_KIND.to_owned())
        );
        let ids = arrow
            .column(3)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        assert!(
            ids.iter()
                .all(|value| value == Some(native.id().get().to_be_bytes().as_slice()))
        );
        let seeds = arrow
            .column(4)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert!(seeds.values().iter().all(|seed| *seed == native.seed()));
        let dictionary = arrow
            .column(2)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();
        let dictionary_values = dictionary
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            dictionary_values.iter().collect::<Vec<_>>(),
            vec![Some("risky"), Some("safe"), Some("unused")]
        );

        assert_eq!(EvalRequest::try_from((&schema, &arrow)).unwrap(), native);
    }

    #[test]
    fn request_reader_normalizes_dictionary_order() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let original = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        let fields = original
            .schema()
            .fields()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let reordered = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1]),
            Arc::new(StringArray::from(vec!["risky", "safe"])),
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new_with_metadata(
                fields,
                original.schema().metadata().clone(),
            )),
            vec![
                original.column(0).clone(),
                original.column(1).clone(),
                Arc::new(reordered),
                original.column(3).clone(),
                original.column(4).clone(),
            ],
        )
        .unwrap();

        let decoded = EvalRequest::try_from((&schema, &batch)).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn rejects_unknown_values_in_categorical_dictionaries() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let original = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        let dictionary = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1]),
            Arc::new(StringArray::from(vec!["risky", "intruder"])),
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            original.schema(),
            vec![
                original.column(0).clone(),
                original.column(1).clone(),
                Arc::new(dictionary),
                original.column(3).clone(),
                original.column(4).clone(),
            ],
        )
        .unwrap();

        assert!(matches!(
            EvalRequest::try_from((&schema, &batch)),
            Err(ArrowConversionError::UnknownDictionaryValue { field, category })
                if field == "mode" && category == "intruder"
        ));
    }

    #[test]
    fn rejects_nullable_fields_before_accepting_values() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let batch = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        let fields = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(position, field)| {
                if position == 0 {
                    Arc::new(field.as_ref().clone().with_nullable(true))
                } else {
                    field.clone()
                }
            })
            .collect::<Vec<_>>();
        let nullable = RecordBatch::try_new(
            Arc::new(ArrowSchema::new_with_metadata(
                fields,
                batch.schema().metadata().clone(),
            )),
            batch.columns().to_vec(),
        )
        .unwrap();

        assert!(matches!(
            EvalRequest::try_from((&schema, &nullable)),
            Err(ArrowConversionError::NullableField { field }) if field == "load"
        ));
    }

    #[test]
    fn rejects_request_domains_that_differ_from_the_model_schema() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let batch = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        let fields = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(position, field)| {
                if position == 0 {
                    let mut field = field.as_ref().clone();
                    field
                        .metadata_mut()
                        .insert(INPUT_UPPER_BOUND_METADATA_KEY.to_owned(), "2".to_owned());
                    Arc::new(field)
                } else {
                    field.clone()
                }
            })
            .collect::<Vec<_>>();
        let mismatched = RecordBatch::try_new(
            Arc::new(ArrowSchema::new_with_metadata(
                fields,
                batch.schema().metadata().clone(),
            )),
            batch.columns().to_vec(),
        )
        .unwrap();

        assert!(matches!(
            EvalRequest::try_from((&schema, &mismatched)),
            Err(ArrowConversionError::RequestInputMismatch {
                position: 0,
                field
            }) if field == "load"
        ));
    }

    #[test]
    fn rejects_non_finite_arrow_values() {
        let schema = ModelSchema::new(
            vec![InputSchema::continuous("x", 0.0, 1.0).unwrap()],
            vec![OutputSchema::boolean("failure").unwrap()],
        )
        .unwrap();
        let inputs = InputChunk::new(
            &schema,
            vec![Feature::continuous("x", vec![0.25, 0.75]).unwrap()],
        )
        .unwrap();
        let request = EvalRequest::new(EvaluationId::new(1), 2, inputs);
        let original = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        let batch = RecordBatch::try_new(
            original.schema(),
            vec![
                Arc::new(Float64Array::from(vec![0.5, f64::NAN])),
                original.column(1).clone(),
                original.column(2).clone(),
            ],
        )
        .unwrap();

        assert!(matches!(
            EvalRequest::try_from((&schema, &batch)),
            Err(ArrowConversionError::Data(DataError::NonFiniteValue {
                row: 1,
                ..
            }))
        ));
    }

    #[test]
    fn rejects_missing_request_kind_before_copying_columns() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let mut batch = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        batch.schema_metadata_mut().remove(BATCH_KIND_METADATA_KEY);

        assert!(matches!(
            EvalRequest::try_from((&schema, &batch)),
            Err(ArrowConversionError::MissingSchemaMetadata {
                key: BATCH_KIND_METADATA_KEY
            })
        ));
    }

    #[test]
    fn rejects_evaluation_ids_that_change_within_a_batch() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let original = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        let ids = FixedSizeBinaryArray::try_from_iter(
            [EvaluationId::new(1), EvaluationId::new(2)]
                .into_iter()
                .map(|id| id.get().to_be_bytes()),
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            original.schema(),
            vec![
                original.column(0).clone(),
                original.column(1).clone(),
                original.column(2).clone(),
                Arc::new(ids),
                original.column(4).clone(),
            ],
        )
        .unwrap();

        assert!(matches!(
            EvalRequest::try_from((&schema, &batch)),
            Err(ArrowConversionError::InconsistentEvaluationId { row: 1 })
        ));
    }

    #[test]
    fn request_reader_rejects_seeds_that_change_within_a_batch() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let original = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        let inconsistent = RecordBatch::try_new(
            original.schema(),
            vec![
                original.column(0).clone(),
                original.column(1).clone(),
                original.column(2).clone(),
                original.column(3).clone(),
                Arc::new(UInt64Array::from(vec![1, 2])),
            ],
        )
        .unwrap();

        assert!(matches!(
            EvalRequest::try_from((&schema, &inconsistent)),
            Err(ArrowConversionError::InconsistentEvaluationSeed { row: 1 })
        ));
    }

    #[test]
    fn request_schema_is_stable_across_ids_and_seeds() {
        let schema = mixed_schema();
        let first = mixed_request(&schema);
        let second = EvalRequest::new(EvaluationId::new(99), 123, first.inputs().clone());
        let first = RecordBatch::try_from(EvalRequestRef::new(&schema, &first)).unwrap();
        let second = RecordBatch::try_from(EvalRequestRef::new(&schema, &second)).unwrap();
        assert_eq!(first.schema(), second.schema());
    }

    #[test]
    fn evaluation_result_round_trip_preserves_successes_and_failure_categories() {
        let schema = mixed_schema();
        let result = ChunkResult::new(
            EvaluationId::new(7),
            vec![
                RowOutcome::Success(
                    OutputRow::new(&schema, vec![OutputValue::Boolean(true)]).unwrap(),
                ),
                RowOutcome::Failure(RowFailure::Model(ModelError::new("domain failure"))),
                RowOutcome::Failure(RowFailure::Panic(ModelPanic::Message(
                    "panic detail".to_owned(),
                ))),
                RowOutcome::Failure(RowFailure::Panic(ModelPanic::NonStringPayload)),
                RowOutcome::Failure(RowFailure::Evaluator(EvaluatorFailure::new(
                    EvaluatorFailureKind::InvalidOutput,
                    "wrong type",
                ))),
            ],
        )
        .unwrap();

        let arrow = RecordBatch::try_from(ChunkResultRef::new(&schema, &result)).unwrap();
        assert_eq!(arrow.num_columns(), 4);
        assert!(arrow.schema().field(0).is_nullable());
        assert_eq!(
            arrow.schema().metadata().get(BATCH_KIND_METADATA_KEY),
            Some(&EVALUATION_RESULT_BATCH_KIND.to_owned())
        );
        assert_eq!(ChunkResult::try_from((&schema, &arrow)).unwrap(), result);
    }

    #[test]
    fn result_reader_accepts_trailing_extension_fields() {
        let schema = mixed_schema();
        let result = ChunkResult::new(
            EvaluationId::new(8),
            vec![RowOutcome::Success(
                OutputRow::new(&schema, vec![OutputValue::Boolean(false)]).unwrap(),
            )],
        )
        .unwrap();
        let original = RecordBatch::try_from(ChunkResultRef::new(&schema, &result)).unwrap();
        let extension = Field::new("graphcal.assertion_count", DataType::Int64, false)
            .with_metadata(HashMap::from([(
                FIELD_ROLE_METADATA_KEY.to_owned(),
                EXTENSION_FIELD_ROLE.to_owned(),
            )]));
        let mut fields = original
            .schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        fields.push(extension);
        let mut columns = original.columns().to_vec();
        columns.push(Arc::new(Int64Array::from(vec![0])));
        let extended = RecordBatch::try_new(
            Arc::new(ArrowSchema::new_with_metadata(
                fields,
                original.schema().metadata().clone(),
            )),
            columns,
        )
        .unwrap();

        assert_eq!(ChunkResult::try_from((&schema, &extended)).unwrap(), result);
    }

    #[test]
    fn result_reader_rejects_unknown_status_codes() {
        let schema = mixed_schema();
        let result = ChunkResult::new(
            EvaluationId::new(9),
            vec![RowOutcome::Failure(RowFailure::Model(ModelError::new(
                "failed",
            )))],
        )
        .unwrap();
        let original = RecordBatch::try_from(ChunkResultRef::new(&schema, &result)).unwrap();
        let unknown = RecordBatch::try_new(
            original.schema(),
            vec![
                original.column(0).clone(),
                original.column(1).clone(),
                Arc::new(UInt8Array::from(vec![255])),
                original.column(3).clone(),
            ],
        )
        .unwrap();

        assert!(matches!(
            ChunkResult::try_from((&schema, &unknown)),
            Err(ArrowConversionError::UnknownOutcomeStatus {
                row: 0,
                status: 255
            })
        ));
    }

    #[test]
    fn result_reader_rejects_outputs_on_failed_rows() {
        let schema = mixed_schema();
        let result = ChunkResult::new(
            EvaluationId::new(9),
            vec![RowOutcome::Failure(RowFailure::Model(ModelError::new(
                "failed",
            )))],
        )
        .unwrap();
        let original = RecordBatch::try_from(ChunkResultRef::new(&schema, &result)).unwrap();
        let inconsistent = RecordBatch::try_new(
            original.schema(),
            vec![
                Arc::new(BooleanArray::from(vec![Some(true)])),
                original.column(1).clone(),
                original.column(2).clone(),
                original.column(3).clone(),
            ],
        )
        .unwrap();

        assert!(matches!(
            ChunkResult::try_from((&schema, &inconsistent)),
            Err(ArrowConversionError::InconsistentOutcome { row: 0, .. })
        ));
    }

    #[test]
    fn rejects_outputs_before_inputs_in_discovery_schema() {
        let native = mixed_schema();
        let arrow = ArrowSchema::try_from(&native).unwrap();
        let reordered = ArrowSchema::new_with_metadata(
            vec![arrow.field(3).clone(), arrow.field(0).clone()],
            arrow.metadata().clone(),
        );

        assert!(matches!(
            ModelSchema::try_from(&reordered),
            Err(ArrowConversionError::InputAfterOutput { field }) if field == "load"
        ));
    }

    #[test]
    fn output_fields_map_to_non_nullable_boolean() {
        let schema = mixed_schema();
        let arrow = ArrowSchema::try_from(schema).unwrap();
        let output = arrow.field(3);
        assert_eq!(output.data_type(), &DataType::Boolean);
        assert!(!output.is_nullable());

        let array: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        assert_eq!(array.data_type(), output.data_type());
    }
}
