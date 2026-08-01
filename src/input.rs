//! Validated columnar model-input chunks and borrowed row access.

use thiserror::Error;

use crate::data::{Feature, FeatureKind, FeatureView};
use crate::schema::{FeatureDomain, InputPosition, ModelSchema};

/// Errors raised when input columns do not satisfy a [`ModelSchema`].
#[derive(Clone, Debug, Error, PartialEq)]
pub enum InputChunkError {
    /// The number of input columns differs from the model schema.
    #[error("input chunk has {actual} features, but the model schema requires {expected}")]
    FeatureCountMismatch {
        /// Number of schema inputs.
        expected: usize,
        /// Number of supplied columns.
        actual: usize,
    },

    /// A column name differs from the corresponding schema input.
    #[error(
        "input column {position} is named '{actual}', but the model schema requires '{expected}'"
    )]
    FeatureNameMismatch {
        /// Zero-based schema position.
        position: usize,
        /// Required input name.
        expected: String,
        /// Supplied feature name.
        actual: String,
    },

    /// A column type differs from the corresponding schema input.
    #[error("input '{name}' has kind {actual:?}, but the model schema requires {expected:?}")]
    FeatureKindMismatch {
        /// Name of the affected input.
        name: String,
        /// Required storage type.
        expected: FeatureKind,
        /// Supplied storage type.
        actual: FeatureKind,
    },

    /// Input columns have different row counts.
    #[error("input '{name}' has {actual} rows, but the chunk has {expected} rows")]
    RowCountMismatch {
        /// Name of the mismatched input.
        name: String,
        /// Row count established by the first column.
        expected: usize,
        /// Row count in this column.
        actual: usize,
    },

    /// A continuous value lies outside its declared closed interval.
    #[error("input '{name}' contains value {value} at row {row}, outside [{lower}, {upper}]")]
    ContinuousOutOfBounds {
        /// Name of the affected input.
        name: String,
        /// Zero-based row containing the value.
        row: usize,
        /// Rejected value.
        value: f64,
        /// Inclusive schema lower bound.
        lower: f64,
        /// Inclusive schema upper bound.
        upper: f64,
    },

    /// An integer value lies outside its declared closed interval.
    #[error("input '{name}' contains value {value} at row {row}, outside [{lower}, {upper}]")]
    IntegerOutOfBounds {
        /// Name of the affected input.
        name: String,
        /// Zero-based row containing the value.
        row: usize,
        /// Rejected value.
        value: i64,
        /// Inclusive schema lower bound.
        lower: i64,
        /// Inclusive schema upper bound.
        upper: i64,
    },

    /// A categorical value does not belong to its declared category set.
    #[error("input '{name}' contains unknown category '{category}' at row {row}")]
    UnknownCategory {
        /// Name of the affected input.
        name: String,
        /// Zero-based row containing the value.
        row: usize,
        /// Rejected category.
        category: String,
    },

    /// A categorical dictionary code cannot be resolved.
    #[error("input '{name}' contains invalid category code {code} at row {row}")]
    InvalidCategoryCode {
        /// Name of the affected input.
        name: String,
        /// Zero-based row containing the invalid code.
        row: usize,
        /// Rejected dictionary code.
        code: i32,
    },
}

/// Errors raised when looking up a value in an [`InputRow`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum InputAccessError {
    /// The requested row position does not exist.
    #[error("row index {index} is out of bounds for a chunk with {row_count} rows")]
    RowIndexOutOfBounds {
        /// Requested zero-based row position.
        index: usize,
        /// Number of rows in the chunk.
        row_count: usize,
    },

    /// The requested feature position does not exist.
    #[error("feature index {index} is out of bounds for a row with {feature_count} features")]
    FeatureIndexOutOfBounds {
        /// Requested zero-based feature position.
        index: usize,
        /// Number of features in the row.
        feature_count: usize,
    },

    /// A categorical dictionary code cannot be resolved.
    #[error("feature '{name}' has invalid category code {code} at row {row}")]
    InvalidCategoryCode {
        /// Name of the affected feature.
        name: String,
        /// Zero-based row containing the invalid code.
        row: usize,
        /// Rejected dictionary code.
        code: i32,
    },
}

/// One typed value borrowed from an input row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputValue<'data> {
    /// A continuous value.
    Continuous(f64),
    /// An integer value.
    Integer(i64),
    /// A categorical dictionary value.
    Categorical(&'data str),
}

impl InputValue<'_> {
    /// Returns the value's storage type.
    #[must_use]
    pub const fn kind(&self) -> FeatureKind {
        match self {
            Self::Continuous(_) => FeatureKind::Continuous,
            Self::Integer(_) => FeatureKind::Integer,
            Self::Categorical(_) => FeatureKind::Categorical,
        }
    }
}

/// A non-empty set of model inputs stored as typed columns.
///
/// Construction validates names, kinds, row counts, and every value against a
/// model schema. Evaluators validate again against their own discovered schema
/// so a chunk cannot accidentally be sent to a different model.
#[derive(Clone, Debug, PartialEq)]
pub struct InputChunk {
    features: Vec<Feature>,
    row_count: usize,
}

impl InputChunk {
    /// Constructs and validates a model-input chunk.
    ///
    /// # Errors
    ///
    /// Returns [`InputChunkError`] when the columns differ from `schema`, have
    /// unequal row counts, or contain a value outside its declared domain.
    pub fn new(schema: &ModelSchema, features: Vec<Feature>) -> Result<Self, InputChunkError> {
        let row_count = validate_features(schema, &features)?;
        Ok(Self {
            features,
            row_count,
        })
    }

    /// Revalidates this chunk against an evaluator's schema.
    ///
    /// # Errors
    ///
    /// Returns [`InputChunkError`] for any schema or domain mismatch.
    pub fn validate_against(&self, schema: &ModelSchema) -> Result<(), InputChunkError> {
        validate_features(schema, &self.features).map(|_| ())
    }

    /// Returns columns in model-schema order.
    #[must_use]
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Returns zero-copy views in model-schema order.
    #[must_use]
    pub fn feature_views(&self) -> impl ExactSizeIterator<Item = FeatureView<'_>> {
        self.features.iter().map(Feature::view)
    }

    /// Returns the number of input rows.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Borrows one row by zero-based position.
    ///
    /// # Errors
    ///
    /// Returns [`InputAccessError`] when `index` is outside this chunk.
    pub const fn row(&self, index: usize) -> Result<InputRow<'_>, InputAccessError> {
        if index < self.row_count {
            Ok(self.row_in_bounds(index))
        } else {
            Err(InputAccessError::RowIndexOutOfBounds {
                index,
                row_count: self.row_count,
            })
        }
    }

    /// Iterates over borrowed rows in original order.
    #[must_use]
    pub fn rows(&self) -> impl ExactSizeIterator<Item = InputRow<'_>> {
        (0..self.row_count).map(|index| self.row_in_bounds(index))
    }

    pub(super) const fn row_in_bounds(&self, index: usize) -> InputRow<'_> {
        debug_assert!(index < self.row_count);
        InputRow { chunk: self, index }
    }

    /// Consumes the chunk and returns its owned columns.
    #[must_use]
    pub fn into_features(self) -> Vec<Feature> {
        self.features
    }
}

/// A zero-copy row facade over an [`InputChunk`].
#[derive(Clone, Copy, Debug)]
pub struct InputRow<'data> {
    chunk: &'data InputChunk,
    index: usize,
}

impl<'data> InputRow<'data> {
    /// Returns the row's zero-based position within its request chunk.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Returns the number of input features.
    #[must_use]
    pub const fn feature_count(&self) -> usize {
        self.chunk.features.len()
    }

    /// Returns one typed value by schema position.
    ///
    /// # Errors
    ///
    /// Returns [`InputAccessError`] when `position` does not belong to this
    /// chunk's schema or a categorical code violates the validated column
    /// invariant.
    pub fn value(&self, position: InputPosition) -> Result<InputValue<'data>, InputAccessError> {
        let Some(feature) = self.chunk.features.get(position.index()) else {
            return Err(InputAccessError::FeatureIndexOutOfBounds {
                index: position.index(),
                feature_count: self.chunk.features.len(),
            });
        };
        match feature.view() {
            FeatureView::Continuous(values) => Ok(InputValue::Continuous(values[self.index])),
            FeatureView::Integer(values) => Ok(InputValue::Integer(values[self.index])),
            FeatureView::Categorical(values) => {
                let code = values.codes()[self.index];
                values.category(code).map_or_else(
                    || {
                        Err(InputAccessError::InvalidCategoryCode {
                            name: feature.name().to_owned(),
                            row: self.index,
                            code,
                        })
                    },
                    |category| Ok(InputValue::Categorical(category)),
                )
            }
        }
    }
}

fn validate_features(schema: &ModelSchema, features: &[Feature]) -> Result<usize, InputChunkError> {
    if features.len() != schema.inputs().len() {
        return Err(InputChunkError::FeatureCountMismatch {
            expected: schema.inputs().len(),
            actual: features.len(),
        });
    }

    let row_count = match features.first() {
        Some(feature) => feature.len(),
        None => {
            return Err(InputChunkError::FeatureCountMismatch {
                expected: schema.inputs().len(),
                actual: 0,
            });
        }
    };

    for (position, (input, feature)) in schema.inputs().iter().zip(features).enumerate() {
        if input.name() != feature.name() {
            return Err(InputChunkError::FeatureNameMismatch {
                position,
                expected: input.name().to_owned(),
                actual: feature.name().to_owned(),
            });
        }
        if input.kind() != feature.kind() {
            return Err(InputChunkError::FeatureKindMismatch {
                name: input.name().to_owned(),
                expected: input.kind(),
                actual: feature.kind(),
            });
        }
        if feature.len() != row_count {
            return Err(InputChunkError::RowCountMismatch {
                name: input.name().to_owned(),
                expected: row_count,
                actual: feature.len(),
            });
        }
        validate_domain(input.name(), input.domain(), feature.view())?;
    }

    Ok(row_count)
}

fn validate_domain(
    name: &str,
    domain: &FeatureDomain,
    values: FeatureView<'_>,
) -> Result<(), InputChunkError> {
    match (domain, values) {
        (FeatureDomain::Continuous(domain), FeatureView::Continuous(values)) => values
            .iter()
            .position(|value| *value < domain.lower() || *value > domain.upper())
            .map_or(Ok(()), |row| {
                Err(InputChunkError::ContinuousOutOfBounds {
                    name: name.to_owned(),
                    row,
                    value: values[row],
                    lower: domain.lower(),
                    upper: domain.upper(),
                })
            }),
        (FeatureDomain::Integer(domain), FeatureView::Integer(values)) => values
            .iter()
            .position(|value| *value < domain.lower() || *value > domain.upper())
            .map_or(Ok(()), |row| {
                Err(InputChunkError::IntegerOutOfBounds {
                    name: name.to_owned(),
                    row,
                    value: values[row],
                    lower: domain.lower(),
                    upper: domain.upper(),
                })
            }),
        (FeatureDomain::Categorical(domain), FeatureView::Categorical(values)) => {
            for (row, code) in values.codes().iter().copied().enumerate() {
                let Some(category) = values.category(code) else {
                    return Err(InputChunkError::InvalidCategoryCode {
                        name: name.to_owned(),
                        row,
                        code,
                    });
                };
                if !domain.contains(category) {
                    return Err(InputChunkError::UnknownCategory {
                        name: name.to_owned(),
                        row,
                        category: category.to_owned(),
                    });
                }
            }
            Ok(())
        }
        _ => Err(InputChunkError::FeatureKindMismatch {
            name: name.to_owned(),
            expected: domain.kind(),
            actual: values.kind(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InputSchema, OutputSchema};

    fn schema() -> ModelSchema {
        ModelSchema::new(
            vec![
                InputSchema::continuous("x", 0.0, 1.0).unwrap(),
                InputSchema::categorical("mode", vec!["safe".to_owned(), "risky".to_owned()])
                    .unwrap(),
            ],
            vec![OutputSchema::boolean("failure").unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn validates_values_against_the_declared_domain() {
        let error = InputChunk::new(
            &schema(),
            vec![
                Feature::continuous("x", vec![0.5, 1.1]).unwrap(),
                Feature::categorical("mode", vec!["safe".to_owned(), "risky".to_owned()]).unwrap(),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            InputChunkError::ContinuousOutOfBounds { row: 1, .. }
        ));
    }

    #[test]
    fn rejects_unknown_categories() {
        let error = InputChunk::new(
            &schema(),
            vec![
                Feature::continuous("x", vec![0.5]).unwrap(),
                Feature::categorical("mode", vec!["unknown".to_owned()]).unwrap(),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            InputChunkError::UnknownCategory { row: 0, .. }
        ));
    }

    #[test]
    fn rows_expose_typed_values_without_copying() {
        let schema = schema();
        let x = schema.input_position("x").unwrap();
        let mode = schema.input_position("mode").unwrap();
        let chunk = InputChunk::new(
            &schema,
            vec![
                Feature::continuous("x", vec![0.25]).unwrap(),
                Feature::categorical("mode", vec!["risky".to_owned()]).unwrap(),
            ],
        )
        .unwrap();
        let row = chunk.row(0).unwrap();
        assert_eq!(row.value(x), Ok(InputValue::Continuous(0.25)));
        assert_eq!(row.value(mode), Ok(InputValue::Categorical("risky")));
    }
}
