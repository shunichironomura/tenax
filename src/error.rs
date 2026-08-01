use thiserror::Error;

/// Errors raised while constructing native feature columns or static datasets.
#[derive(Debug, Error, PartialEq)]
pub enum DataError {
    /// A feature name is empty or contains only whitespace.
    #[error("feature names must not be empty")]
    EmptyFeatureName,

    /// A feature contains no observations.
    #[error("feature '{name}' contains no observations")]
    EmptyFeature {
        /// The empty feature's name.
        name: String,
    },

    /// No input features were supplied.
    #[error("a dataset must contain at least one feature")]
    NoFeatures,

    /// No outcomes were supplied.
    #[error("a dataset must contain at least one outcome")]
    NoOutcomes,

    /// A feature and the outcome have different row counts.
    #[error("feature '{name}' has {feature_rows} rows, but the outcome has {outcome_rows} rows")]
    RowCountMismatch {
        /// The feature whose length differs.
        name: String,
        /// Number of values in the feature.
        feature_rows: usize,
        /// Number of binary outcomes in the dataset.
        outcome_rows: usize,
    },

    /// Two features have the same name.
    #[error("feature name '{name}' occurs more than once")]
    DuplicateFeatureName {
        /// The duplicated name.
        name: String,
    },

    /// A continuous feature contains NaN or infinity.
    #[error("feature '{name}' contains non-finite value {value} at row {row}")]
    NonFiniteValue {
        /// The feature containing the invalid value.
        name: String,
        /// Zero-based row containing the invalid value.
        row: usize,
        /// The NaN or infinite value.
        value: f64,
    },

    /// A categorical feature contains an empty or whitespace-only value.
    #[error("feature '{name}' contains an empty category at row {row}")]
    EmptyCategoryValue {
        /// The feature containing the invalid value.
        name: String,
        /// Zero-based row containing the invalid value.
        row: usize,
    },

    /// A categorical feature cannot be represented by `Int32` codes.
    #[error("feature '{name}' has {count} categories, exceeding the Int32 code space")]
    TooManyCategories {
        /// The feature containing too many distinct values.
        name: String,
        /// Number of distinct categories.
        count: usize,
    },
}

/// Errors raised while constructing PRIM algorithm configuration.
#[derive(Debug, Error, PartialEq)]
pub enum PrimError {
    /// A PRIM tuning parameter is outside its supported interval.
    #[error("invalid {parameter}={value}: {requirement}")]
    InvalidParameter {
        /// The configuration parameter's name.
        parameter: &'static str,
        /// The rejected value.
        value: f64,
        /// A human-readable description of the valid domain.
        requirement: &'static str,
    },
}
