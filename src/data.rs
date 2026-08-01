use std::collections::HashSet;

use crate::PrimError;

/// The storage type of an explanatory feature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureKind {
    /// A real-valued feature peeled at empirical quantiles.
    Continuous,
    /// An integer-valued feature peeled at observed values.
    Integer,
    /// A nominal feature peeled one category at a time.
    Categorical,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FeatureData {
    Continuous(Vec<f64>),
    Integer(Vec<i64>),
    Categorical(Vec<String>),
}

/// A named, typed input column for a static PRIM dataset.
///
/// Construct features with [`Feature::continuous`], [`Feature::integer`], or
/// [`Feature::categorical`]. Continuous values must be finite; missing values
/// are deliberately rejected rather than assigned ambiguous box semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    name: String,
    pub(crate) data: FeatureData,
}

impl Feature {
    /// Constructs a finite, real-valued feature.
    pub fn continuous(name: impl Into<String>, values: Vec<f64>) -> Result<Self, PrimError> {
        let name = validate_name(name.into())?;
        validate_not_empty(&name, &values)?;
        match values.iter().position(|value| !value.is_finite()) {
            Some(row) => Err(PrimError::NonFiniteValue {
                name,
                row,
                value: values[row],
            }),
            None => Ok(Self {
                name,
                data: FeatureData::Continuous(values),
            }),
        }
    }

    /// Constructs an integer-valued feature.
    pub fn integer(name: impl Into<String>, values: Vec<i64>) -> Result<Self, PrimError> {
        let name = validate_name(name.into())?;
        validate_not_empty(&name, &values)?;
        Ok(Self {
            name,
            data: FeatureData::Integer(values),
        })
    }

    /// Constructs a nominal feature.
    pub fn categorical(name: impl Into<String>, values: Vec<String>) -> Result<Self, PrimError> {
        let name = validate_name(name.into())?;
        validate_not_empty(&name, &values)?;
        Ok(Self {
            name,
            data: FeatureData::Categorical(values),
        })
    }

    /// Returns the feature name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the feature storage type.
    #[must_use]
    pub const fn kind(&self) -> FeatureKind {
        match &self.data {
            FeatureData::Continuous(_) => FeatureKind::Continuous,
            FeatureData::Integer(_) => FeatureKind::Integer,
            FeatureData::Categorical(_) => FeatureKind::Categorical,
        }
    }

    /// Returns the number of observations in the feature.
    #[must_use]
    pub const fn len(&self) -> usize {
        match &self.data {
            FeatureData::Continuous(values) => values.len(),
            FeatureData::Integer(values) => values.len(),
            FeatureData::Categorical(values) => values.len(),
        }
    }

    /// Returns whether this feature has no observations.
    ///
    /// Valid features are never empty; this method is provided for symmetry
    /// with other collection-like APIs.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns continuous values, or `None` for a differently typed feature.
    #[must_use]
    pub fn continuous_values(&self) -> Option<&[f64]> {
        match &self.data {
            FeatureData::Continuous(values) => Some(values),
            _ => None,
        }
    }

    /// Returns integer values, or `None` for a differently typed feature.
    #[must_use]
    pub fn integer_values(&self) -> Option<&[i64]> {
        match &self.data {
            FeatureData::Integer(values) => Some(values),
            _ => None,
        }
    }

    /// Returns categorical values, or `None` for a differently typed feature.
    #[must_use]
    pub fn categorical_values(&self) -> Option<&[String]> {
        match &self.data {
            FeatureData::Categorical(values) => Some(values),
            _ => None,
        }
    }
}

/// A validated static input/output dataset for binary scenario discovery.
///
/// `true` outcomes denote the cases of interest that PRIM should concentrate
/// inside high-density boxes.
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    features: Vec<Feature>,
    cases_of_interest: Vec<bool>,
}

impl Dataset {
    /// Constructs a dataset and verifies its schema and row counts.
    pub fn new(features: Vec<Feature>, cases_of_interest: Vec<bool>) -> Result<Self, PrimError> {
        if features.is_empty() {
            return Err(PrimError::NoFeatures);
        }
        if cases_of_interest.is_empty() {
            return Err(PrimError::NoOutcomes);
        }

        let mut names = HashSet::with_capacity(features.len());
        for feature in &features {
            if feature.len() != cases_of_interest.len() {
                return Err(PrimError::RowCountMismatch {
                    name: feature.name.clone(),
                    feature_rows: feature.len(),
                    outcome_rows: cases_of_interest.len(),
                });
            }
            if !names.insert(feature.name.clone()) {
                return Err(PrimError::DuplicateFeatureName {
                    name: feature.name.clone(),
                });
            }
        }

        Ok(Self {
            features,
            cases_of_interest,
        })
    }

    /// Returns the input features in schema order.
    #[must_use]
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Returns the binary outcome column.
    #[must_use]
    pub fn cases_of_interest(&self) -> &[bool] {
        &self.cases_of_interest
    }

    /// Returns the number of observations.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.cases_of_interest.len()
    }

    /// Returns the number of cases of interest.
    #[must_use]
    pub fn case_count(&self) -> usize {
        self.cases_of_interest
            .iter()
            .filter(|is_of_interest| **is_of_interest)
            .count()
    }
}

fn validate_name(name: String) -> Result<String, PrimError> {
    if name.trim().is_empty() {
        Err(PrimError::EmptyFeatureName)
    } else {
        Ok(name)
    }
}

fn validate_not_empty<T>(name: &str, values: &[T]) -> Result<(), PrimError> {
    if values.is_empty() {
        Err(PrimError::EmptyFeature {
            name: name.to_owned(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_finite_continuous_values() {
        let error = Feature::continuous("x", vec![0.0, f64::NAN]).unwrap_err();
        assert!(matches!(error, PrimError::NonFiniteValue { row: 1, .. }));
    }

    #[test]
    fn rejects_duplicate_names() {
        let features = vec![
            Feature::integer("x", vec![1]).unwrap(),
            Feature::categorical("x", vec!["a".to_owned()]).unwrap(),
        ];
        let error = Dataset::new(features, vec![true]).unwrap_err();
        assert_eq!(
            error,
            PrimError::DuplicateFeatureName {
                name: "x".to_owned()
            }
        );
    }
}
