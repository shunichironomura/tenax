//! Apache Arrow schema and evaluation-request boundary conversions.
//!
//! This module is available with the `arrow` crate feature. It keeps Arrow out
//! of the algorithmic core while defining the column and metadata contract used
//! by future process and network adapters.
//!
//! A model discovery [`ArrowSchema`] contains all inputs followed by all
//! outputs. Evaluation-request [`RecordBatch`] values contain only input
//! columns and carry one evaluation ID and request seed in schema metadata.
//! Every field is non-nullable.
//!
//! | Tenax value | Arrow data type |
//! | --- | --- |
//! | Continuous input | `Float64` |
//! | Integer input | `Int64` |
//! | Categorical input | `Dictionary<Int32, Utf8>` |
//! | Boolean output | `Boolean` |
//!
//! Unknown metadata keys are ignored by Tenax readers, so the contract can gain
//! optional annotations. Tenax-owned metadata is namespaced
//! with `tenax.` and versioned by [`SCHEMA_VERSION_METADATA_KEY`].
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
    Array, ArrayRef, DictionaryArray, Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow_schema::{ArrowError as ArrowRsError, DataType, Field};
use thiserror::Error;

pub use arrow_array::RecordBatch;
pub use arrow_schema::Schema as ArrowSchema;

use crate::DataError;
use crate::data::{Feature, FeatureView};
use crate::evaluation::{EvalRequest, EvaluationId};
use crate::input::{InputChunk, InputChunkError};
use crate::schema::{
    FeatureDomain, InputSchema, ModelFieldRole, ModelSchema, OutputKind, OutputSchema, SchemaError,
};

/// Metadata key containing the Tenax Arrow contract version.
pub const SCHEMA_VERSION_METADATA_KEY: &str = "tenax.schema.version";
/// Current value of [`SCHEMA_VERSION_METADATA_KEY`].
pub const SCHEMA_VERSION: &str = "1";
/// Field metadata key identifying an input or output role.
pub const FIELD_ROLE_METADATA_KEY: &str = "tenax.field.role";
/// Value of [`FIELD_ROLE_METADATA_KEY`] for a model input.
pub const INPUT_FIELD_ROLE: &str = "input";
/// Value of [`FIELD_ROLE_METADATA_KEY`] for a model output.
pub const OUTPUT_FIELD_ROLE: &str = "output";
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
/// Value of [`BATCH_KIND_METADATA_KEY`] for an evaluation request.
pub const EVALUATION_REQUEST_BATCH_KIND: &str = "evaluation_request";
/// Evaluation-request metadata key containing a canonical 32-digit hexadecimal ID.
pub const EVALUATION_ID_METADATA_KEY: &str = "tenax.evaluation.id";
/// Evaluation-request metadata key containing the decimal request seed.
pub const EVALUATION_SEED_METADATA_KEY: &str = "tenax.evaluation.seed";

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

    /// A record batch has a semantic kind other than an evaluation request.
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
    #[error("Arrow field '{field}' is nullable; Tenax fields must be non-nullable")]
    NullableField {
        /// Nullable field name.
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

    /// The number of request columns differs from the expected model inputs.
    #[error(
        "Arrow evaluation request has {actual} fields, but the model schema requires {expected} inputs"
    )]
    RequestFieldCountMismatch {
        /// Number of model inputs.
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

    /// Evaluation ID metadata is not canonical hexadecimal `u128` data.
    #[error("Arrow evaluation ID metadata '{value}' is not 32 lowercase hexadecimal digits")]
    InvalidEvaluationId {
        /// Rejected metadata value.
        value: String,
    },

    /// Evaluation seed metadata is not decimal `u64` data.
    #[error("Arrow evaluation seed metadata '{value}' is not a decimal u64")]
    InvalidEvaluationSeed {
        /// Rejected metadata value.
        value: String,
    },

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

impl TryFrom<&ModelSchema> for ArrowSchema {
    type Error = ArrowConversionError;

    fn try_from(schema: &ModelSchema) -> Result<Self, Self::Error> {
        let fields = schema
            .inputs()
            .iter()
            .map(input_field)
            .chain(schema.outputs().iter().map(output_field).map(Ok))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new_with_metadata(fields, version_metadata()))
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

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut saw_output = false;
        for field in schema.fields() {
            match field_role(field)? {
                ModelFieldRole::Input => {
                    if saw_output {
                        return Err(ArrowConversionError::InputAfterOutput {
                            field: field.name().clone(),
                        });
                    }
                    inputs.push(input_schema(field)?);
                }
                ModelFieldRole::Output => {
                    saw_output = true;
                    outputs.push(output_schema(field)?);
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

impl TryFrom<EvalRequestRef<'_>> for RecordBatch {
    type Error = ArrowConversionError;

    fn try_from(value: EvalRequestRef<'_>) -> Result<Self, Self::Error> {
        let EvalRequestRef { schema, request } = value;
        request.inputs().validate_against(schema)?;

        let fields = schema
            .inputs()
            .iter()
            .map(input_field)
            .collect::<Result<Vec<_>, _>>()?;
        let columns = schema
            .inputs()
            .iter()
            .zip(request.inputs().feature_views())
            .map(|(input, values)| input_array(input, values))
            .collect::<Result<Vec<_>, _>>()?;

        let metadata = request_metadata(request);
        Self::try_new(
            Arc::new(ArrowSchema::new_with_metadata(fields, metadata)),
            columns,
        )
        .map_err(Into::into)
    }
}

impl TryFrom<(&ModelSchema, &RecordBatch)> for EvalRequest {
    type Error = ArrowConversionError;

    fn try_from((schema, batch): (&ModelSchema, &RecordBatch)) -> Result<Self, Self::Error> {
        let arrow_schema = batch.schema_ref();
        validate_version(arrow_schema.metadata())?;
        validate_batch_kind(arrow_schema.metadata())?;

        if arrow_schema.fields().len() != schema.inputs().len() {
            return Err(ArrowConversionError::RequestFieldCountMismatch {
                expected: schema.inputs().len(),
                actual: arrow_schema.fields().len(),
            });
        }
        let id = evaluation_id(arrow_schema.metadata())?;
        let seed = evaluation_seed(arrow_schema.metadata())?;

        let features = schema
            .inputs()
            .iter()
            .zip(arrow_schema.fields())
            .zip(batch.columns())
            .enumerate()
            .map(|(position, ((expected, field), array))| {
                let role = field_role(field)?;
                if role != ModelFieldRole::Input || input_schema(field)? != *expected {
                    return Err(ArrowConversionError::RequestInputMismatch {
                        position,
                        field: field.name().clone(),
                    });
                }
                feature_from_array(expected, field, array)
            })
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

fn output_field(output: &OutputSchema) -> Field {
    let data_type = match output.kind() {
        OutputKind::Boolean => DataType::Boolean,
    };
    Field::new(output.name(), data_type, false).with_metadata(HashMap::from([(
        FIELD_ROLE_METADATA_KEY.to_owned(),
        OUTPUT_FIELD_ROLE.to_owned(),
    )]))
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

fn output_schema(field: &Field) -> Result<OutputSchema, ArrowConversionError> {
    validate_non_nullable(field)?;
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

fn field_role(field: &Field) -> Result<ModelFieldRole, ArrowConversionError> {
    match required_field_metadata(field, FIELD_ROLE_METADATA_KEY)? {
        INPUT_FIELD_ROLE => Ok(ModelFieldRole::Input),
        OUTPUT_FIELD_ROLE => Ok(ModelFieldRole::Output),
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

fn validate_batch_kind(metadata: &HashMap<String, String>) -> Result<(), ArrowConversionError> {
    let actual = required_schema_metadata(metadata, BATCH_KIND_METADATA_KEY)?;
    if actual == EVALUATION_REQUEST_BATCH_KIND {
        Ok(())
    } else {
        Err(ArrowConversionError::UnexpectedBatchKind {
            expected: EVALUATION_REQUEST_BATCH_KIND,
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

fn evaluation_id(metadata: &HashMap<String, String>) -> Result<EvaluationId, ArrowConversionError> {
    let value = required_schema_metadata(metadata, EVALUATION_ID_METADATA_KEY)?;
    let canonical = value.len() == 32
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !canonical {
        return Err(ArrowConversionError::InvalidEvaluationId {
            value: value.to_owned(),
        });
    }
    u128::from_str_radix(value, 16)
        .map(EvaluationId::new)
        .map_err(|_| ArrowConversionError::InvalidEvaluationId {
            value: value.to_owned(),
        })
}

fn evaluation_seed(metadata: &HashMap<String, String>) -> Result<u64, ArrowConversionError> {
    let value = required_schema_metadata(metadata, EVALUATION_SEED_METADATA_KEY)?;
    value
        .parse()
        .map_err(|_| ArrowConversionError::InvalidEvaluationSeed {
            value: value.to_owned(),
        })
}

fn version_metadata() -> HashMap<String, String> {
    HashMap::from([(
        SCHEMA_VERSION_METADATA_KEY.to_owned(),
        SCHEMA_VERSION.to_owned(),
    )])
}

fn request_metadata(request: &EvalRequest) -> HashMap<String, String> {
    HashMap::from([
        (
            SCHEMA_VERSION_METADATA_KEY.to_owned(),
            SCHEMA_VERSION.to_owned(),
        ),
        (
            BATCH_KIND_METADATA_KEY.to_owned(),
            EVALUATION_REQUEST_BATCH_KIND.to_owned(),
        ),
        (
            EVALUATION_ID_METADATA_KEY.to_owned(),
            request.id().to_string(),
        ),
        (
            EVALUATION_SEED_METADATA_KEY.to_owned(),
            request.seed().to_string(),
        ),
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
        assert_eq!(arrow.num_columns(), 3);
        assert_eq!(
            arrow.schema().metadata().get(EVALUATION_ID_METADATA_KEY),
            Some(&native.id().to_string())
        );
        assert_eq!(
            arrow.schema().metadata().get(EVALUATION_SEED_METADATA_KEY),
            Some(&native.seed().to_string())
        );
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
        let field = input_field(&schema.inputs()[0]).unwrap();
        let metadata = HashMap::from([
            (
                SCHEMA_VERSION_METADATA_KEY.to_owned(),
                SCHEMA_VERSION.to_owned(),
            ),
            (
                BATCH_KIND_METADATA_KEY.to_owned(),
                EVALUATION_REQUEST_BATCH_KIND.to_owned(),
            ),
            (EVALUATION_ID_METADATA_KEY.to_owned(), format!("{:032x}", 1)),
            (EVALUATION_SEED_METADATA_KEY.to_owned(), "2".to_owned()),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new_with_metadata(vec![field], metadata)),
            vec![Arc::new(Float64Array::from(vec![0.5, f64::NAN]))],
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
    fn rejects_missing_request_metadata_before_copying_columns() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let mut batch = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        batch
            .schema_metadata_mut()
            .remove(EVALUATION_SEED_METADATA_KEY);

        assert!(matches!(
            EvalRequest::try_from((&schema, &batch)),
            Err(ArrowConversionError::MissingSchemaMetadata {
                key: EVALUATION_SEED_METADATA_KEY
            })
        ));
    }

    #[test]
    fn rejects_noncanonical_evaluation_id_metadata() {
        let schema = mixed_schema();
        let request = mixed_request(&schema);
        let mut batch = RecordBatch::try_from(EvalRequestRef::new(&schema, &request)).unwrap();
        batch
            .schema_metadata_mut()
            .insert(EVALUATION_ID_METADATA_KEY.to_owned(), "ABC".to_owned());

        assert!(matches!(
            EvalRequest::try_from((&schema, &batch)),
            Err(ArrowConversionError::InvalidEvaluationId { value }) if value == "ABC"
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
