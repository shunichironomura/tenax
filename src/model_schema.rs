//! A typed convenience builder for the session schema of a Rust model.
//!
//! Tenax evaluates models through the Graphcal batch model-evaluation
//! protocol vocabulary ([`graphcal_model_protocol`]). A configured
//! [`SessionSchema`] lists the inputs every request supplies and the outputs
//! every result returns. Graphcal models obtain theirs from the model's
//! described interface; [`ModelSchemaBuilder`] declares one directly for an
//! in-process Rust model.

use std::sync::Arc;

use graphcal_model_protocol::{
    Bounds, BoundsError, CanonicalUnitsOnly, DatetimeType, Dimension, Domain, FiniteF64, IndexName,
    InterfaceError, KeyType, KeyTypeError, LabelError, ModelInterface, NonFiniteError,
    OutputDescription, ParameterDescription, PortName, PortWire, QuantityType, SessionError,
    SessionSchema, SessionSpec, VariantName, WireType,
};
use thiserror::Error;

/// Errors raised while declaring a Rust model's session schema.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ModelSchemaError {
    /// A port, index, or variant name is not a valid protocol label.
    #[error("invalid label: {0}")]
    Label(#[from] LabelError),

    /// A quantity bound is NaN or infinite.
    #[error("input '{name}' has a non-finite bound: {source}")]
    NonFiniteBound {
        /// The input carrying the invalid bound.
        name: PortName,
        /// The rejected value.
        source: NonFiniteError,
    },

    /// Bounds are reversed.
    #[error("input '{name}' has invalid bounds: {source}")]
    InvalidBounds {
        /// The input carrying the invalid bounds.
        name: PortName,
        /// Why the bounds are invalid.
        source: BoundsError,
    },

    /// A categorical variant list is empty, repeats a variant, or is too long.
    #[error("port '{name}' has an invalid key type: {source}")]
    InvalidKey {
        /// The port carrying the invalid key type.
        name: PortName,
        /// Why the variant list is invalid.
        source: KeyTypeError,
    },

    /// Two inputs or two outputs share a name.
    #[error(transparent)]
    Interface(#[from] InterfaceError),

    /// The schema declares no input or no output.
    #[error(transparent)]
    Session(#[from] SessionError),
}

/// Declares the inputs and outputs of an in-process Rust model.
///
/// Every declared input is varied by each request and every declared output is
/// returned by each successful row, both in declaration order. Each method
/// validates its declaration immediately, so a builder chain stops at the first
/// invalid declaration.
///
/// # Example
///
/// ```
/// use tenax::ModelSchemaBuilder;
///
/// let schema = ModelSchemaBuilder::new()
///     .continuous("load", 0.0, 1.0)?
///     .integer("replicas", 1, 5)?
///     .categorical("regime", "Regime", ["stable", "fragile"])?
///     .boolean_output("failure")?
///     .build()?;
///
/// assert_eq!(schema.inputs().len(), 3);
/// assert!(schema.output_slot("failure").is_some());
/// # Ok::<(), tenax::ModelSchemaError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
#[must_use]
pub struct ModelSchemaBuilder {
    parameters: Vec<ParameterDescription>,
    outputs: Vec<OutputDescription>,
}

impl ModelSchemaBuilder {
    /// Starts an empty declaration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a dimensionless real input over the closed interval
    /// `[lower, upper]`.
    ///
    /// Equal bounds are accepted and describe a constant input.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError`] when the name is invalid, a bound is
    /// non-finite, or the bounds are reversed.
    pub fn continuous(
        self,
        name: impl Into<String>,
        lower: f64,
        upper: f64,
    ) -> Result<Self, ModelSchemaError> {
        let name = PortName::new(name)?;
        let finite = |value| {
            FiniteF64::new(value).map_err(|source| ModelSchemaError::NonFiniteBound {
                name: name.clone(),
                source,
            })
        };
        let bounds = Bounds::new(Some(finite(lower)?), Some(finite(upper)?)).map_err(|source| {
            ModelSchemaError::InvalidBounds {
                name: name.clone(),
                source,
            }
        })?;
        Ok(self.quantity_port(name, dimensionless(), bounds))
    }

    /// Declares a real input of `quantity` type whose values lie within
    /// `bounds`, expressed in the quantity's unit.
    ///
    /// Half-open bounds are accepted, but samplers require both sides.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError::Label`] when the name is invalid.
    pub fn quantity(
        self,
        name: impl Into<String>,
        quantity: QuantityType,
        bounds: Bounds<FiniteF64>,
    ) -> Result<Self, ModelSchemaError> {
        Ok(self.quantity_port(PortName::new(name)?, quantity, bounds))
    }

    /// Declares a datetime input read in `datetime`'s time scale whose values,
    /// in seconds since J2000, lie within `bounds`.
    ///
    /// Half-open bounds are accepted, but samplers require both sides.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError::Label`] when the name is invalid.
    pub fn datetime(
        mut self,
        name: impl Into<String>,
        datetime: DatetimeType,
        bounds: Bounds<FiniteF64>,
    ) -> Result<Self, ModelSchemaError> {
        self.parameters.push(parameter(
            PortName::new(name)?,
            WireType::Datetime(datetime),
            Some(Domain::Datetime(bounds)),
        ));
        Ok(self)
    }

    /// Declares an integer input over the closed interval `[lower, upper]`.
    ///
    /// Equal bounds are accepted and describe a constant input.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError`] when the name is invalid or the bounds are
    /// reversed.
    pub fn integer(
        mut self,
        name: impl Into<String>,
        lower: i64,
        upper: i64,
    ) -> Result<Self, ModelSchemaError> {
        let name = PortName::new(name)?;
        let bounds = Bounds::new(Some(lower), Some(upper)).map_err(|source| {
            ModelSchemaError::InvalidBounds {
                name: name.clone(),
                source,
            }
        })?;
        self.parameters
            .push(parameter(name, WireType::Int, Some(Domain::Int(bounds))));
        Ok(self)
    }

    /// Declares a categorical input over the `variants` of the index
    /// `index_name`, in declaration order.
    ///
    /// Samplers draw variants by their declaration-order position.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError`] when a label is invalid or the variant
    /// list is empty, repeats a variant, or is too long.
    pub fn categorical<V>(
        mut self,
        name: impl Into<String>,
        index_name: impl Into<String>,
        variants: impl IntoIterator<Item = V>,
    ) -> Result<Self, ModelSchemaError>
    where
        V: Into<String>,
    {
        let name = PortName::new(name)?;
        let key = key_type(&name, index_name, variants)?;
        self.parameters
            .push(parameter(name, WireType::Key(key), None));
        Ok(self)
    }

    /// Declares a Boolean input.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError::Label`] when the name is invalid.
    pub fn boolean(mut self, name: impl Into<String>) -> Result<Self, ModelSchemaError> {
        self.parameters
            .push(parameter(PortName::new(name)?, WireType::Bool, None));
        Ok(self)
    }

    /// Declares a Boolean output, such as a case-of-interest classification.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError::Label`] when the name is invalid.
    pub fn boolean_output(self, name: impl Into<String>) -> Result<Self, ModelSchemaError> {
        self.output(name, WireType::Bool)
    }

    /// Declares a real output of `quantity` type.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError::Label`] when the name is invalid.
    pub fn quantity_output(
        self,
        name: impl Into<String>,
        quantity: QuantityType,
    ) -> Result<Self, ModelSchemaError> {
        self.output(name, WireType::Quantity(quantity))
    }

    /// Declares an integer output.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError::Label`] when the name is invalid.
    pub fn integer_output(self, name: impl Into<String>) -> Result<Self, ModelSchemaError> {
        self.output(name, WireType::Int)
    }

    /// Declares a categorical output over the `variants` of the index
    /// `index_name`, in declaration order.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError`] when a label is invalid or the variant
    /// list is empty, repeats a variant, or is too long.
    pub fn key_output<V>(
        mut self,
        name: impl Into<String>,
        index_name: impl Into<String>,
        variants: impl IntoIterator<Item = V>,
    ) -> Result<Self, ModelSchemaError>
    where
        V: Into<String>,
    {
        let name = PortName::new(name)?;
        let key = key_type(&name, index_name, variants)?;
        self.outputs.push(output(name, WireType::Key(key)));
        Ok(self)
    }

    /// Configures the session that varies every declared input and returns
    /// every declared output.
    ///
    /// # Errors
    ///
    /// Returns [`ModelSchemaError`] when two inputs or two outputs share a
    /// name, or when no input or no output was declared.
    pub fn build(self) -> Result<Arc<SessionSchema>, ModelSchemaError> {
        let spec = SessionSpec::new(
            self.parameters
                .iter()
                .map(|parameter| parameter.name().clone())
                .collect(),
            self.outputs
                .iter()
                .map(|output| output.name().clone())
                .collect(),
        );
        let interface = ModelInterface::new(self.parameters, self.outputs)?;
        Ok(Arc::new(interface.configure(&spec, CanonicalUnitsOnly)?))
    }

    fn quantity_port(
        mut self,
        name: PortName,
        quantity: QuantityType,
        bounds: Bounds<FiniteF64>,
    ) -> Self {
        self.parameters.push(parameter(
            name,
            WireType::Quantity(quantity),
            Some(Domain::Quantity(bounds)),
        ));
        self
    }

    fn output(mut self, name: impl Into<String>, wire: WireType) -> Result<Self, ModelSchemaError> {
        self.outputs.push(output(PortName::new(name)?, wire));
        Ok(self)
    }
}

#[expect(
    clippy::expect_used,
    reason = "a dimensionless quantity never needs a unit"
)]
fn dimensionless() -> QuantityType {
    QuantityType::new(Dimension::dimensionless(), None)
        .expect("a dimensionless quantity is valid without a unit")
}

fn key_type<V>(
    name: &PortName,
    index_name: impl Into<String>,
    variants: impl IntoIterator<Item = V>,
) -> Result<KeyType, ModelSchemaError>
where
    V: Into<String>,
{
    let index = IndexName::new(index_name)?;
    let variants = variants
        .into_iter()
        .map(VariantName::new)
        .collect::<Result<Vec<_>, _>>()?;
    KeyType::new(index, variants).map_err(|source| ModelSchemaError::InvalidKey {
        name: name.clone(),
        source,
    })
}

#[expect(
    clippy::expect_used,
    reason = "callers pair quantity bounds only with quantities and integer bounds only with integers"
)]
fn parameter(name: PortName, wire: WireType, domain: Option<Domain>) -> ParameterDescription {
    let type_spelling = type_spelling(&wire);
    ParameterDescription::new(
        name,
        type_spelling,
        PortWire::Supported(wire),
        domain,
        false,
    )
    .expect("the builder pairs every domain with its matching wire type")
}

fn output(name: PortName, wire: WireType) -> OutputDescription {
    let type_spelling = type_spelling(&wire);
    OutputDescription::new(name, type_spelling, PortWire::Supported(wire))
}

/// Renders a human-readable type spelling for the interface description.
fn type_spelling(wire: &WireType) -> String {
    match wire {
        WireType::Quantity(quantity) if quantity.dimension().is_dimensionless() => {
            "Dimensionless".to_owned()
        }
        WireType::Quantity(quantity) => quantity
            .dimension()
            .exponents()
            .map(|(base, exponent)| format!("{base}^{exponent}"))
            .collect::<Vec<_>>()
            .join(" * "),
        WireType::Int => "Int".to_owned(),
        WireType::Bool => "Bool".to_owned(),
        WireType::Key(key) => format!("Key<{}>", key.index()),
        WireType::Complex(quantity) => format!(
            "Complex<{}>",
            type_spelling(&WireType::Quantity(quantity.clone()))
        ),
        WireType::Datetime(datetime) => format!("Datetime<{}>", datetime.scale()),
        WireType::Indexed(indexed) => format!(
            "{}[{}]",
            type_spelling(indexed.element()),
            indexed.axis().index()
        ),
        WireType::Algebraic(algebraic) => algebraic.name().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{BaseDimensionName, Exponent, UnitSpelling};

    use super::*;

    #[test]
    fn builds_a_session_varying_every_input_in_declaration_order() {
        let schema = ModelSchemaBuilder::new()
            .continuous("x", -1.0, 1.0)
            .unwrap()
            .integer("count", -2, 2)
            .unwrap()
            .categorical("mode", "Mode", ["safe", "risky"])
            .unwrap()
            .boolean("enabled")
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .integer_output("cost")
            .unwrap()
            .build()
            .unwrap();

        let names = schema
            .inputs()
            .iter()
            .map(|input| input.name().as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["x", "count", "mode", "enabled"]);
        assert_eq!(
            schema.inputs()[0].domain(),
            Some(&Domain::Quantity(
                Bounds::new(
                    Some(FiniteF64::new(-1.0).unwrap()),
                    Some(FiniteF64::new(1.0).unwrap())
                )
                .unwrap()
            ))
        );
        let WireType::Key(mode) = schema.inputs()[2].wire() else {
            panic!("mode must be a key input");
        };
        assert_eq!(mode.variants()[0].as_str(), "safe");
        assert_eq!(schema.inputs()[3].domain(), None);
        assert_eq!(schema.outputs()[1].wire(), &WireType::Int);
    }

    #[test]
    fn quantity_inputs_carry_their_dimension_and_unit() {
        let length = QuantityType::new(
            Dimension::from_iter([(
                BaseDimensionName::new("Length").unwrap(),
                Exponent::new(1, 1).unwrap(),
            )]),
            Some(UnitSpelling::new("m").unwrap()),
        )
        .unwrap();
        let bounds = Bounds::new(Some(FiniteF64::new(0.0).unwrap()), None).unwrap();
        let schema = ModelSchemaBuilder::new()
            .quantity("depth", length.clone(), bounds)
            .unwrap()
            .quantity_output("level", length.clone())
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(schema.inputs()[0].wire(), &WireType::Quantity(length));
        assert_eq!(schema.inputs()[0].domain(), Some(&Domain::Quantity(bounds)));
    }

    #[test]
    fn rejects_invalid_declarations_with_typed_errors() {
        assert!(matches!(
            ModelSchemaBuilder::new().continuous("", 0.0, 1.0),
            Err(ModelSchemaError::Label(_))
        ));
        assert!(matches!(
            ModelSchemaBuilder::new().continuous("x", f64::NAN, 1.0),
            Err(ModelSchemaError::NonFiniteBound { .. })
        ));
        assert!(matches!(
            ModelSchemaBuilder::new().continuous("x", 1.0, 0.0),
            Err(ModelSchemaError::InvalidBounds {
                source: BoundsError::Inverted,
                ..
            })
        ));
        assert!(matches!(
            ModelSchemaBuilder::new().integer("n", 3, 2),
            Err(ModelSchemaError::InvalidBounds { .. })
        ));
        assert!(matches!(
            ModelSchemaBuilder::new().categorical("mode", "Mode", ["a", "a"]),
            Err(ModelSchemaError::InvalidKey {
                source: KeyTypeError::DuplicateVariant { .. },
                ..
            })
        ));
        assert!(matches!(
            ModelSchemaBuilder::new().categorical("mode", "Mode", Vec::<String>::new()),
            Err(ModelSchemaError::InvalidKey {
                source: KeyTypeError::Empty { .. },
                ..
            })
        ));
    }

    #[test]
    fn build_rejects_duplicates_and_empty_sides() {
        let duplicate = ModelSchemaBuilder::new()
            .boolean("x")
            .unwrap()
            .boolean("x")
            .unwrap()
            .boolean_output("y")
            .unwrap()
            .build();
        assert!(matches!(
            duplicate,
            Err(ModelSchemaError::Interface(
                InterfaceError::DuplicateParameter { .. }
            ))
        ));
        assert_eq!(
            ModelSchemaBuilder::new()
                .boolean_output("y")
                .unwrap()
                .build(),
            Err(ModelSchemaError::Session(SessionError::NoInputs))
        );
        assert_eq!(
            ModelSchemaBuilder::new().boolean("x").unwrap().build(),
            Err(ModelSchemaError::Session(SessionError::NoOutputs))
        );
    }
}
