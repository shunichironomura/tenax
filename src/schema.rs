//! Validated model input domains and output declarations.

use std::collections::HashSet;
use std::fmt;

use thiserror::Error;

use crate::FeatureKind;

/// Whether a named model field is an input or an output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFieldRole {
    /// A model input.
    Input,
    /// A model output.
    Output,
}

impl fmt::Display for ModelFieldRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => formatter.write_str("input"),
            Self::Output => formatter.write_str("output"),
        }
    }
}

/// A schema-validated position in a model's input list.
///
/// This role-specific type prevents row, output, and input indices from being
/// interchanged accidentally. Obtain one with [`ModelSchema::input_position`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InputPosition(usize);

impl InputPosition {
    /// Returns the zero-based boundary representation.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A schema-validated position in a model's output list.
///
/// This role-specific type prevents row, input, and output indices from being
/// interchanged accidentally. Obtain one with [`ModelSchema::output_position`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OutputPosition(usize);

impl OutputPosition {
    /// Returns the zero-based boundary representation.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Errors raised when resolving a named field in a validated schema.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SchemaLookupError {
    /// No input has the requested name.
    #[error("model schema has no input named '{name}'")]
    UnknownInput {
        /// The requested opaque input name.
        name: String,
    },

    /// No output has the requested name.
    #[error("model schema has no output named '{name}'")]
    UnknownOutput {
        /// The requested opaque output name.
        name: String,
    },
}

/// Errors raised while declaring a model schema.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SchemaError {
    /// No model inputs were declared.
    #[error("a model schema must declare at least one input")]
    NoInputs,

    /// No model outputs were declared.
    #[error("a model schema must declare at least one output")]
    NoOutputs,

    /// A field name is empty or contains only whitespace.
    #[error("{role} names must not be empty")]
    EmptyFieldName {
        /// The role of the unnamed field.
        role: ModelFieldRole,
    },

    /// An input unit is empty or contains only whitespace.
    #[error("unit for input '{name}' must not be empty")]
    EmptyUnit {
        /// The input carrying the invalid annotation.
        name: String,
    },

    /// Continuous bounds do not describe a finite, ordered interval.
    #[error(
        "continuous input '{name}' has invalid bounds [{lower}, {upper}]: bounds and their width must be finite, with lower <= upper"
    )]
    InvalidContinuousBounds {
        /// The input carrying the invalid bounds.
        name: String,
        /// The rejected lower bound.
        lower: f64,
        /// The rejected upper bound.
        upper: f64,
    },

    /// Integer bounds are reversed.
    #[error("integer input '{name}' has invalid bounds [{lower}, {upper}]: lower must be <= upper")]
    InvalidIntegerBounds {
        /// The input carrying the invalid bounds.
        name: String,
        /// The rejected lower bound.
        lower: i64,
        /// The rejected upper bound.
        upper: i64,
    },

    /// A categorical input has no permitted values.
    #[error("categorical input '{name}' must declare at least one category")]
    NoCategories {
        /// The input with an empty category set.
        name: String,
    },

    /// A category is empty or contains only whitespace.
    #[error("categorical input '{name}' has an empty category at position {position}")]
    EmptyCategory {
        /// The input carrying the invalid category.
        name: String,
        /// Zero-based position in the supplied category list.
        position: usize,
    },

    /// A category occurs more than once in one input domain.
    #[error("category '{category}' occurs more than once in input '{name}'")]
    DuplicateCategory {
        /// The input carrying the duplicate.
        name: String,
        /// The duplicated category.
        category: String,
    },

    /// A categorical domain cannot be represented by `Int32` category codes.
    #[error("categorical input '{name}' has {count} categories, exceeding the Int32 code space")]
    TooManyCategories {
        /// The input carrying too many categories.
        name: String,
        /// Number of supplied categories.
        count: usize,
    },

    /// Two input/output fields have the same name.
    #[error("model field name '{name}' occurs more than once")]
    DuplicateFieldName {
        /// The duplicated field name.
        name: String,
    },
}

/// A finite closed interval used as a continuous sampling domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuousDomain {
    lower: f64,
    upper: f64,
}

impl ContinuousDomain {
    /// Returns the inclusive lower bound.
    #[must_use]
    pub const fn lower(&self) -> f64 {
        self.lower
    }

    /// Returns the inclusive upper bound.
    #[must_use]
    pub const fn upper(&self) -> f64 {
        self.upper
    }
}

/// A closed interval used as an integer sampling domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntegerDomain {
    lower: i64,
    upper: i64,
}

impl IntegerDomain {
    /// Returns the inclusive lower bound.
    #[must_use]
    pub const fn lower(&self) -> i64 {
        self.lower
    }

    /// Returns the inclusive upper bound.
    #[must_use]
    pub const fn upper(&self) -> i64 {
        self.upper
    }
}

/// A non-empty set used as a categorical sampling domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CategoricalDomain {
    categories: Vec<String>,
}

impl CategoricalDomain {
    /// Returns permitted categories in lexical order.
    #[must_use]
    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    /// Returns whether this domain contains `category`.
    #[must_use]
    pub fn contains(&self, category: &str) -> bool {
        self.categories
            .binary_search_by(|candidate| candidate.as_str().cmp(category))
            .is_ok()
    }
}

/// The validated domain of one model input.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureDomain {
    /// A finite closed real interval.
    Continuous(ContinuousDomain),
    /// A closed integer interval.
    Integer(IntegerDomain),
    /// A non-empty nominal category set.
    Categorical(CategoricalDomain),
}

impl FeatureDomain {
    /// Returns the input storage type required by this domain.
    #[must_use]
    pub const fn kind(&self) -> FeatureKind {
        match self {
            Self::Continuous(_) => FeatureKind::Continuous,
            Self::Integer(_) => FeatureKind::Integer,
            Self::Categorical(_) => FeatureKind::Categorical,
        }
    }
}

/// The name, type, and valid domain of one model input.
#[derive(Clone, Debug, PartialEq)]
pub struct InputSchema {
    name: String,
    domain: FeatureDomain,
    unit: Option<String>,
}

impl InputSchema {
    /// Declares a continuous input over the closed interval `[lower, upper]`.
    ///
    /// Equal bounds are accepted and describe a constant input.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError`] when the name is empty, either bound or the
    /// interval width is non-finite, or the bounds are reversed.
    pub fn continuous(
        name: impl Into<String>,
        lower: f64,
        upper: f64,
    ) -> Result<Self, SchemaError> {
        let name = validate_name(name.into(), ModelFieldRole::Input)?;
        if !(lower.is_finite()
            && upper.is_finite()
            && (upper - lower).is_finite()
            && lower <= upper)
        {
            return Err(SchemaError::InvalidContinuousBounds { name, lower, upper });
        }
        Ok(Self {
            name,
            domain: FeatureDomain::Continuous(ContinuousDomain { lower, upper }),
            unit: None,
        })
    }

    /// Declares an integer input over the closed interval `[lower, upper]`.
    ///
    /// Equal bounds are accepted and describe a constant input.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError`] when the name is empty or the bounds are
    /// reversed.
    pub fn integer(name: impl Into<String>, lower: i64, upper: i64) -> Result<Self, SchemaError> {
        let name = validate_name(name.into(), ModelFieldRole::Input)?;
        if lower > upper {
            return Err(SchemaError::InvalidIntegerBounds { name, lower, upper });
        }
        Ok(Self {
            name,
            domain: FeatureDomain::Integer(IntegerDomain { lower, upper }),
            unit: None,
        })
    }

    /// Declares a categorical input.
    ///
    /// Categories are stored in lexical order so sampling and future wire
    /// encodings have deterministic dictionary codes independent of hash order.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError`] when the name is empty, no categories are
    /// supplied, a category is empty or duplicated, or the category count
    /// exceeds the `Int32` dictionary-code space.
    pub fn categorical(
        name: impl Into<String>,
        categories: Vec<String>,
    ) -> Result<Self, SchemaError> {
        let name = validate_name(name.into(), ModelFieldRole::Input)?;
        if categories.is_empty() {
            return Err(SchemaError::NoCategories { name });
        }
        if let Some(position) = categories
            .iter()
            .position(|category| category.trim().is_empty())
        {
            return Err(SchemaError::EmptyCategory { name, position });
        }
        if i32::try_from(categories.len() - 1).is_err() {
            return Err(SchemaError::TooManyCategories {
                name,
                count: categories.len(),
            });
        }

        let mut seen = HashSet::with_capacity(categories.len());
        for category in &categories {
            if !seen.insert(category.as_str()) {
                return Err(SchemaError::DuplicateCategory {
                    name,
                    category: category.clone(),
                });
            }
        }

        let mut categories = categories;
        categories.sort_unstable();
        Ok(Self {
            name,
            domain: FeatureDomain::Categorical(CategoricalDomain { categories }),
            unit: None,
        })
    }

    /// Adds a unit annotation used by typed model bindings and interchange
    /// adapters. The input's numeric bounds and values are expressed in this
    /// unit; adapters for unit-aware models must convert explicitly at their
    /// boundary.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError::EmptyUnit`] when `unit` is empty or contains only
    /// whitespace.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Result<Self, SchemaError> {
        let unit = unit.into();
        if unit.trim().is_empty() {
            return Err(SchemaError::EmptyUnit { name: self.name });
        }
        self.unit = Some(unit);
        Ok(self)
    }

    /// Returns the input name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the validated input domain.
    #[must_use]
    pub const fn domain(&self) -> &FeatureDomain {
        &self.domain
    }

    /// Returns the input storage type.
    #[must_use]
    pub const fn kind(&self) -> FeatureKind {
        self.domain.kind()
    }

    /// Returns the optional unit annotation.
    #[must_use]
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }
}

/// The value type of a model output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputKind {
    /// A binary output suitable for classification and current PRIM analysis.
    Boolean,
}

/// The name and type of one model output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputSchema {
    name: String,
    kind: OutputKind,
}

impl OutputSchema {
    /// Declares a binary model output.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError`] when the output name is empty.
    pub fn boolean(name: impl Into<String>) -> Result<Self, SchemaError> {
        Ok(Self {
            name: validate_name(name.into(), ModelFieldRole::Output)?,
            kind: OutputKind::Boolean,
        })
    }

    /// Returns the output name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the output value type.
    #[must_use]
    pub const fn kind(&self) -> OutputKind {
        self.kind
    }
}

/// A validated transport-independent model contract.
///
/// Inputs and outputs retain declaration order. Field names are unique across
/// both roles, making later Arrow and other record-oriented mappings
/// unambiguous.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSchema {
    inputs: Vec<InputSchema>,
    outputs: Vec<OutputSchema>,
}

impl ModelSchema {
    /// Constructs a schema and validates collection-level invariants.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError`] when either role is empty or any field name is
    /// duplicated across inputs and outputs.
    pub fn new(inputs: Vec<InputSchema>, outputs: Vec<OutputSchema>) -> Result<Self, SchemaError> {
        if inputs.is_empty() {
            return Err(SchemaError::NoInputs);
        }
        if outputs.is_empty() {
            return Err(SchemaError::NoOutputs);
        }

        let mut names = HashSet::with_capacity(inputs.len() + outputs.len());
        for name in inputs
            .iter()
            .map(InputSchema::name)
            .chain(outputs.iter().map(OutputSchema::name))
        {
            if !names.insert(name) {
                return Err(SchemaError::DuplicateFieldName {
                    name: name.to_owned(),
                });
            }
        }

        Ok(Self { inputs, outputs })
    }

    /// Returns inputs in declaration order.
    #[must_use]
    pub fn inputs(&self) -> &[InputSchema] {
        &self.inputs
    }

    /// Returns outputs in declaration order.
    #[must_use]
    pub fn outputs(&self) -> &[OutputSchema] {
        &self.outputs
    }

    /// Resolves an opaque input name to a role-specific position.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaLookupError::UnknownInput`] when no input has `name`.
    pub fn input_position(&self, name: &str) -> Result<InputPosition, SchemaLookupError> {
        self.inputs
            .iter()
            .position(|input| input.name() == name)
            .map(InputPosition)
            .ok_or_else(|| SchemaLookupError::UnknownInput {
                name: name.to_owned(),
            })
    }

    /// Resolves an opaque output name to a role-specific position.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaLookupError::UnknownOutput`] when no output has `name`.
    pub fn output_position(&self, name: &str) -> Result<OutputPosition, SchemaLookupError> {
        self.outputs
            .iter()
            .position(|output| output.name() == name)
            .map(OutputPosition)
            .ok_or_else(|| SchemaLookupError::UnknownOutput {
                name: name.to_owned(),
            })
    }
}

fn validate_name(name: String, role: ModelFieldRole) -> Result<String, SchemaError> {
    if name.trim().is_empty() {
        Err(SchemaError::EmptyFieldName { role })
    } else {
        Ok(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boolean_output() -> OutputSchema {
        OutputSchema::boolean("failure").unwrap()
    }

    #[test]
    fn validates_continuous_bounds() {
        assert!(InputSchema::continuous("x", 1.0, 1.0).is_ok());
        assert!(matches!(
            InputSchema::continuous("x", 2.0, 1.0),
            Err(SchemaError::InvalidContinuousBounds { .. })
        ));
        assert!(matches!(
            InputSchema::continuous("x", f64::NEG_INFINITY, 1.0),
            Err(SchemaError::InvalidContinuousBounds { .. })
        ));
        assert!(matches!(
            InputSchema::continuous("x", -f64::MAX, f64::MAX),
            Err(SchemaError::InvalidContinuousBounds { .. })
        ));
    }

    #[test]
    fn categorical_domains_are_non_empty_unique_and_sorted() {
        let input =
            InputSchema::categorical("mode", vec!["zeta".to_owned(), "alpha".to_owned()]).unwrap();
        let FeatureDomain::Categorical(domain) = input.domain() else {
            panic!("constructor returned the wrong domain kind");
        };
        assert_eq!(domain.categories(), ["alpha", "zeta"]);

        assert!(matches!(
            InputSchema::categorical("mode", Vec::new()),
            Err(SchemaError::NoCategories { .. })
        ));
        assert!(matches!(
            InputSchema::categorical("mode", vec!["a".to_owned(), "a".to_owned()]),
            Err(SchemaError::DuplicateCategory { .. })
        ));
    }

    #[test]
    fn model_field_names_are_unique_across_roles() {
        let error = ModelSchema::new(
            vec![InputSchema::integer("value", 0, 1).unwrap()],
            vec![OutputSchema::boolean("value").unwrap()],
        )
        .unwrap_err();
        assert_eq!(
            error,
            SchemaError::DuplicateFieldName {
                name: "value".to_owned()
            }
        );
    }

    #[test]
    fn model_requires_inputs_and_outputs() {
        assert_eq!(
            ModelSchema::new(Vec::new(), vec![boolean_output()]),
            Err(SchemaError::NoInputs)
        );
        assert_eq!(
            ModelSchema::new(vec![InputSchema::integer("x", 0, 1).unwrap()], Vec::new()),
            Err(SchemaError::NoOutputs)
        );
    }

    #[test]
    fn resolves_role_specific_field_positions() {
        let schema = ModelSchema::new(
            vec![InputSchema::integer("x", 0, 1).unwrap()],
            vec![boolean_output()],
        )
        .unwrap();
        assert_eq!(schema.input_position("x").unwrap().index(), 0);
        assert_eq!(schema.output_position("failure").unwrap().index(), 0);
        assert_eq!(
            schema.input_position("missing"),
            Err(SchemaLookupError::UnknownInput {
                name: "missing".to_owned()
            })
        );
    }
}
