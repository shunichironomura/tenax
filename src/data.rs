use std::collections::{BTreeSet, HashSet};

use crate::DataError;

/// The storage type of a model or scenario-discovery input feature.
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
struct CategoricalData {
    categories: Vec<String>,
    codes: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq)]
enum FeatureData {
    Continuous(Vec<f64>),
    Integer(Vec<i64>),
    Categorical(CategoricalData),
}

/// A zero-copy borrowed view of one categorical column.
///
/// Codes index the lexically ordered category dictionary. The `Int32`
/// representation matches the planned Arrow dictionary interchange type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CategoricalView<'data> {
    codes: &'data [i32],
    categories: &'data [String],
}

impl<'data> CategoricalView<'data> {
    /// Returns one dictionary code per row.
    #[must_use]
    pub const fn codes(&self) -> &'data [i32] {
        self.codes
    }

    /// Returns dictionary values in lexical code order.
    #[must_use]
    pub const fn categories(&self) -> &'data [String] {
        self.categories
    }

    /// Resolves a dictionary code, returning `None` when the supplied code is
    /// outside this dictionary.
    #[must_use]
    pub fn category(&self, code: i32) -> Option<&'data str> {
        usize::try_from(code)
            .ok()
            .and_then(|index| self.categories.get(index))
            .map(String::as_str)
    }

    /// Returns the code assigned to `category`, if it belongs to this column's
    /// dictionary.
    #[must_use]
    pub fn code(&self, category: &str) -> Option<i32> {
        self.categories
            .binary_search_by(|candidate| candidate.as_str().cmp(category))
            .ok()
            .and_then(|index| i32::try_from(index).ok())
    }
}

/// A zero-copy borrowed view of one typed feature column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureView<'data> {
    /// Borrowed finite real values.
    Continuous(&'data [f64]),
    /// Borrowed integer values.
    Integer(&'data [i64]),
    /// Borrowed categorical dictionary codes and values.
    Categorical(CategoricalView<'data>),
}

impl FeatureView<'_> {
    /// Returns the feature storage type.
    #[must_use]
    pub const fn kind(&self) -> FeatureKind {
        match self {
            Self::Continuous(_) => FeatureKind::Continuous,
            Self::Integer(_) => FeatureKind::Integer,
            Self::Categorical(_) => FeatureKind::Categorical,
        }
    }

    /// Returns the number of rows in the column.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Continuous(values) => values.len(),
            Self::Integer(values) => values.len(),
            Self::Categorical(values) => values.codes.len(),
        }
    }

    /// Returns whether this view contains no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A named, typed input column for a static PRIM dataset or model request.
///
/// Construct features with [`Feature::continuous`], [`Feature::integer`], or
/// [`Feature::categorical`]. Continuous values must be finite; missing and
/// empty categorical values are deliberately rejected rather than assigned
/// ambiguous model or box semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    name: String,
    data: FeatureData,
}

impl Feature {
    /// Constructs a finite, real-valued feature.
    ///
    /// # Errors
    ///
    /// Returns [`DataError`] when the name or column is empty or any value is
    /// non-finite.
    pub fn continuous(name: impl Into<String>, values: Vec<f64>) -> Result<Self, DataError> {
        let name = validate_name(name.into())?;
        validate_not_empty(&name, &values)?;
        match values.iter().position(|value| !value.is_finite()) {
            Some(row) => Err(DataError::NonFiniteValue {
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
    ///
    /// # Errors
    ///
    /// Returns [`DataError`] when the name or column is empty.
    pub fn integer(name: impl Into<String>, values: Vec<i64>) -> Result<Self, DataError> {
        let name = validate_name(name.into())?;
        validate_not_empty(&name, &values)?;
        Ok(Self {
            name,
            data: FeatureData::Integer(values),
        })
    }

    /// Constructs and dictionary-encodes a nominal feature.
    ///
    /// Codes are assigned by lexical category order, making their meaning
    /// deterministic and independent of hash iteration order.
    ///
    /// # Errors
    ///
    /// Returns [`DataError`] when the name or column is empty, a category is
    /// empty, or the dictionary cannot be represented by `Int32` codes.
    pub fn categorical(name: impl Into<String>, values: Vec<String>) -> Result<Self, DataError> {
        let name = validate_name(name.into())?;
        let data = encode_categories(&name, values)?;
        Ok(Self {
            name,
            data: FeatureData::Categorical(data),
        })
    }

    // Normalizes a boundary dictionary without expanding every row to a String.
    // Unreferenced values are omitted to preserve `Feature::categorical` semantics.
    pub(crate) fn categorical_from_dictionary(
        name: impl Into<String>,
        codes: Vec<i32>,
        dictionary: Vec<String>,
    ) -> Result<Self, DataError> {
        let name = validate_name(name.into())?;
        validate_not_empty(&name, &codes)?;
        let dictionary = encode_categories(&name, dictionary)?;
        let category_count = dictionary.codes.len();
        let category_positions = codes
            .into_iter()
            .enumerate()
            .map(|(row, code)| {
                usize::try_from(code)
                    .ok()
                    .and_then(|position| dictionary.codes.get(position))
                    .copied()
                    .and_then(|normalized| usize::try_from(normalized).ok())
                    .map(|position| (code, position))
                    .ok_or_else(|| DataError::InvalidCategoryCode {
                        name: name.clone(),
                        row,
                        code,
                        category_count,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let used_positions = category_positions
            .iter()
            .map(|(_, position)| *position)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let categories = dictionary
            .categories
            .into_iter()
            .enumerate()
            .filter_map(|(position, category)| {
                used_positions
                    .binary_search(&position)
                    .is_ok()
                    .then_some(category)
            })
            .collect();
        let codes = category_positions
            .into_iter()
            .enumerate()
            .map(|(row, (code, position))| {
                let normalized = used_positions.binary_search(&position).map_err(|_| {
                    DataError::InvalidCategoryCode {
                        name: name.clone(),
                        row,
                        code,
                        category_count,
                    }
                })?;
                i32::try_from(normalized).map_err(|_| DataError::TooManyCategories {
                    name: name.clone(),
                    count: used_positions.len(),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            name,
            data: FeatureData::Categorical(CategoricalData { categories, codes }),
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

    /// Returns a borrowed columnar view without copying values.
    #[must_use]
    pub fn view(&self) -> FeatureView<'_> {
        match &self.data {
            FeatureData::Continuous(values) => FeatureView::Continuous(values),
            FeatureData::Integer(values) => FeatureView::Integer(values),
            FeatureData::Categorical(values) => FeatureView::Categorical(CategoricalView {
                codes: &values.codes,
                categories: &values.categories,
            }),
        }
    }

    /// Returns the number of observations in the feature.
    #[must_use]
    pub fn len(&self) -> usize {
        self.view().len()
    }

    /// Returns whether this feature has no observations.
    ///
    /// Valid features are never empty; this method is provided for symmetry
    /// with other collection-like APIs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns continuous values, or `None` for a differently typed feature.
    #[must_use]
    pub fn continuous_values(&self) -> Option<&[f64]> {
        match self.view() {
            FeatureView::Continuous(values) => Some(values),
            FeatureView::Integer(_) | FeatureView::Categorical(_) => None,
        }
    }

    /// Returns integer values, or `None` for a differently typed feature.
    #[must_use]
    pub fn integer_values(&self) -> Option<&[i64]> {
        match self.view() {
            FeatureView::Integer(values) => Some(values),
            FeatureView::Continuous(_) | FeatureView::Categorical(_) => None,
        }
    }

    /// Returns categorical dictionary codes, or `None` for a differently typed
    /// feature.
    #[must_use]
    pub fn categorical_codes(&self) -> Option<&[i32]> {
        match self.view() {
            FeatureView::Categorical(values) => Some(values.codes()),
            FeatureView::Continuous(_) | FeatureView::Integer(_) => None,
        }
    }

    /// Returns categorical dictionary values in code order, or `None` for a
    /// differently typed feature.
    #[must_use]
    pub fn categories(&self) -> Option<&[String]> {
        match self.view() {
            FeatureView::Categorical(values) => Some(values.categories()),
            FeatureView::Continuous(_) | FeatureView::Integer(_) => None,
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
    ///
    /// # Errors
    ///
    /// Returns [`DataError`] when inputs or outcomes are empty, feature names
    /// are duplicated, or column lengths differ.
    pub fn new(features: Vec<Feature>, cases_of_interest: Vec<bool>) -> Result<Self, DataError> {
        if features.is_empty() {
            return Err(DataError::NoFeatures);
        }
        if cases_of_interest.is_empty() {
            return Err(DataError::NoOutcomes);
        }

        let mut names = HashSet::with_capacity(features.len());
        for feature in &features {
            if feature.len() != cases_of_interest.len() {
                return Err(DataError::RowCountMismatch {
                    name: feature.name.clone(),
                    feature_rows: feature.len(),
                    outcome_rows: cases_of_interest.len(),
                });
            }
            if !names.insert(feature.name()) {
                return Err(DataError::DuplicateFeatureName {
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

    /// Iterates over zero-copy input column views in schema order.
    #[must_use]
    pub fn feature_views(&self) -> impl ExactSizeIterator<Item = FeatureView<'_>> {
        self.features.iter().map(Feature::view)
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

fn encode_categories(name: &str, values: Vec<String>) -> Result<CategoricalData, DataError> {
    validate_not_empty(name, &values)?;
    if let Some(row) = values
        .iter()
        .position(|category| category.trim().is_empty())
    {
        return Err(DataError::EmptyCategoryValue {
            name: name.to_owned(),
            row,
        });
    }

    let categories = values
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if i32::try_from(categories.len() - 1).is_err() {
        return Err(DataError::TooManyCategories {
            name: name.to_owned(),
            count: categories.len(),
        });
    }
    let codes = values
        .into_iter()
        .map(|value| {
            let index = categories.partition_point(|candidate| candidate < &value);
            i32::try_from(index).map_err(|_| DataError::TooManyCategories {
                name: name.to_owned(),
                count: categories.len(),
            })
        })
        .collect::<Result<_, _>>()?;

    Ok(CategoricalData { categories, codes })
}

fn validate_name(name: String) -> Result<String, DataError> {
    if name.trim().is_empty() {
        Err(DataError::EmptyFeatureName)
    } else {
        Ok(name)
    }
}

fn validate_not_empty<T>(name: &str, values: &[T]) -> Result<(), DataError> {
    if values.is_empty() {
        Err(DataError::EmptyFeature {
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
        assert!(matches!(error, DataError::NonFiniteValue { row: 1, .. }));
    }

    #[test]
    fn categorical_view_has_deterministic_dictionary_codes() {
        let feature = Feature::categorical(
            "mode",
            vec!["zeta".to_owned(), "alpha".to_owned(), "zeta".to_owned()],
        )
        .unwrap();
        let FeatureView::Categorical(view) = feature.view() else {
            panic!("categorical constructor returned a different feature kind");
        };
        assert_eq!(view.categories(), ["alpha", "zeta"]);
        assert_eq!(view.codes(), [1, 0, 1]);
        assert_eq!(view.category(0), Some("alpha"));
        assert_eq!(view.category(-1), None);
    }

    #[test]
    fn categorical_dictionary_construction_normalizes_codes_without_expanding_rows() {
        let feature = Feature::categorical_from_dictionary(
            "mode",
            vec![0, 1, 0],
            vec!["zeta".to_owned(), "alpha".to_owned()],
        )
        .unwrap();
        let FeatureView::Categorical(view) = feature.view() else {
            panic!("dictionary constructor returned a different feature kind");
        };
        assert_eq!(view.categories(), ["alpha", "zeta"]);
        assert_eq!(view.codes(), [1, 0, 1]);

        assert!(matches!(
            Feature::categorical_from_dictionary(
                "mode",
                vec![2],
                vec!["zeta".to_owned(), "alpha".to_owned()]
            ),
            Err(DataError::InvalidCategoryCode {
                row: 0,
                code: 2,
                category_count: 2,
                ..
            })
        ));
    }

    #[test]
    fn rejects_empty_categorical_values() {
        let error = Feature::categorical("mode", vec![" ".to_owned()]).unwrap_err();
        assert_eq!(
            error,
            DataError::EmptyCategoryValue {
                name: "mode".to_owned(),
                row: 0
            }
        );
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
            DataError::DuplicateFeatureName {
                name: "x".to_owned()
            }
        );
    }
}
