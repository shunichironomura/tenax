//! Choosing which parameters of a model interface a sampled session varies.

use std::fmt;

use graphcal_model_protocol::{
    Domain, ModelInterface, ParameterDescription, PortName, PortWire, UnsupportedReason, WireType,
};
use thiserror::Error;

/// How the samplers draw a parameter, by its wire type.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SampledKind {
    /// A real quantity within closed domain bounds.
    Quantity,
    /// A datetime within closed domain bounds.
    Datetime,
    /// An integer within closed domain bounds.
    Integer,
    /// A key, over every variant of its index.
    Key,
    /// A Boolean, over `false` and `true`.
    Boolean,
}

/// A parameter that a sampled session varies, with how it is drawn.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampledParameter {
    name: PortName,
    kind: SampledKind,
}

impl SampledParameter {
    /// Returns the parameter's name.
    #[must_use]
    pub const fn name(&self) -> &PortName {
        &self.name
    }

    /// Returns how the samplers draw the parameter.
    #[must_use]
    pub const fn kind(&self) -> SampledKind {
        self.kind
    }
}

/// A wire type that has no sampling interval or set of levels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UnsampledWire {
    /// A complex quantity.
    Complex,
    /// An indexed value.
    Indexed,
    /// A record or union value.
    Algebraic,
}

impl fmt::Display for UnsampledWire {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Complex => "complex",
            Self::Indexed => "indexed",
            Self::Algebraic => "record or union",
        })
    }
}

/// Why the samplers cannot draw a parameter.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum UnsampleableReason {
    /// The batch protocol cannot carry the parameter at all.
    #[error("the batch protocol cannot carry it: {message} ({reason:?})")]
    NotCarried {
        /// Graphcal's reason.
        reason: UnsupportedReason,
        /// Graphcal's diagnostic.
        message: String,
    },

    /// The parameter's type has no sampling interval or set of levels.
    #[error("the samplers cannot draw a {0} value")]
    Wire(UnsampledWire),

    /// A quantity, datetime, or integer parameter does not declare both
    /// domain bounds, so it has no finite sampling interval.
    #[error("it does not declare both a lower and an upper domain bound")]
    Unbounded,
}

/// A parameter the samplers cannot draw, and why.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("parameter '{name}' cannot be sampled: {reason}")]
pub struct UnsampleableParameter {
    name: PortName,
    reason: UnsampleableReason,
}

impl UnsampleableParameter {
    /// Returns the parameter's name.
    #[must_use]
    pub const fn name(&self) -> &PortName {
        &self.name
    }

    /// Returns why the samplers cannot draw it.
    #[must_use]
    pub const fn reason(&self) -> &UnsampleableReason {
        &self.reason
    }
}

/// Why a model interface cannot be sampled as a whole.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SampledParametersError {
    /// The interface declares no parameter, so a session has nothing to
    /// vary.
    #[error("the model declares no parameter")]
    NoParameters,

    /// Some parameters cannot be drawn.
    #[error("{} parameter(s) cannot be sampled: {}", .0.len(), Unsampleable(.0))]
    Unsampleable(Vec<UnsampleableParameter>),
}

struct Unsampleable<'a>(&'a [UnsampleableParameter]);

impl fmt::Display for Unsampleable<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, parameter) in self.0.iter().enumerate() {
            if position > 0 {
                formatter.write_str("; ")?;
            }
            write!(formatter, "'{}': {}", parameter.name, parameter.reason)?;
        }
        Ok(())
    }
}

/// Returns every parameter of `interface`, in declaration order, with how
/// the samplers draw it.
///
/// A session that varies all of them can be sampled with
/// [`crate::sample_uniform`] or [`crate::sample_latin_hypercube`].
///
/// Every parameter is varied, including those with a default: a default is
/// a modelling convenience, not a statement that the value is certain. To
/// keep some parameters at their defaults or at given values, select the
/// session's parameters explicitly instead.
///
/// The samplers draw the parameters' declared domains, so quantity,
/// datetime, and integer parameters need both bounds.
///
/// ```
/// use tenax::graphcal_model_protocol::{
///     Bounds, Domain, FiniteF64, ModelInterface, ParameterDescription, PortName, PortWire,
///     QuantityType, Dimension, WireType,
/// };
/// use tenax::{SampledKind, sampled_parameters};
///
/// let ratio = ParameterDescription::new(
///     PortName::new("ratio")?,
///     "Dimensionless",
///     PortWire::Supported(WireType::Quantity(QuantityType::new(Dimension::dimensionless(), None)?)),
///     Some(Domain::Quantity(Bounds::new(Some(FiniteF64::new(0.0)?), Some(FiniteF64::new(1.0)?))?)),
///     true,
/// )?;
/// let interface = ModelInterface::new(vec![ratio], vec![])?;
///
/// let parameters = sampled_parameters(&interface)?;
/// assert_eq!(parameters[0].name().as_str(), "ratio");
/// assert_eq!(parameters[0].kind(), SampledKind::Quantity);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`SampledParametersError::NoParameters`] for an interface
/// without parameters, and [`SampledParametersError::Unsampleable`] listing,
/// in declaration order, every parameter the samplers cannot draw.
pub fn sampled_parameters(
    interface: &ModelInterface,
) -> Result<Vec<SampledParameter>, SampledParametersError> {
    if interface.parameters().is_empty() {
        return Err(SampledParametersError::NoParameters);
    }
    let (sampled, unsampleable) = interface.parameters().iter().fold(
        (Vec::new(), Vec::new()),
        |(mut sampled, mut unsampleable), parameter| {
            match sampled_kind(parameter) {
                Ok(kind) => sampled.push(SampledParameter {
                    name: parameter.name().clone(),
                    kind,
                }),
                Err(reason) => unsampleable.push(UnsampleableParameter {
                    name: parameter.name().clone(),
                    reason,
                }),
            }
            (sampled, unsampleable)
        },
    );
    if unsampleable.is_empty() {
        Ok(sampled)
    } else {
        Err(SampledParametersError::Unsampleable(unsampleable))
    }
}

fn sampled_kind(parameter: &ParameterDescription) -> Result<SampledKind, UnsampleableReason> {
    let wire = match parameter.wire() {
        PortWire::Supported(wire) => wire,
        PortWire::Unsupported { reason, message } => {
            return Err(UnsampleableReason::NotCarried {
                reason: *reason,
                message: message.clone(),
            });
        }
    };
    let bounded = || match parameter.domain() {
        Some(Domain::Quantity(bounds) | Domain::Datetime(bounds)) => {
            bounds.min().is_some() && bounds.max().is_some()
        }
        Some(Domain::Int(bounds)) => bounds.min().is_some() && bounds.max().is_some(),
        None => false,
    };
    let kind = match wire {
        WireType::Quantity(_) => SampledKind::Quantity,
        WireType::Datetime(_) => SampledKind::Datetime,
        WireType::Int => SampledKind::Integer,
        WireType::Key(_) => return Ok(SampledKind::Key),
        WireType::Bool => return Ok(SampledKind::Boolean),
        WireType::Complex(_) => return Err(UnsampleableReason::Wire(UnsampledWire::Complex)),
        WireType::Indexed(_) => return Err(UnsampleableReason::Wire(UnsampledWire::Indexed)),
        WireType::Algebraic(_) => return Err(UnsampleableReason::Wire(UnsampledWire::Algebraic)),
    };
    if bounded() {
        Ok(kind)
    } else {
        Err(UnsampleableReason::Unbounded)
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{
        Bounds, CanonicalUnitsOnly, DatetimeType, Dimension, FiniteF64, KeyType, QuantityType,
        SessionSpec, TimeScale,
    };

    use super::*;
    use crate::sample_latin_hypercube;
    use crate::structured_fixture::named_axis;

    fn finite(value: f64) -> FiniteF64 {
        FiniteF64::new(value).unwrap()
    }

    fn parameter(name: &str, wire: PortWire, domain: Option<Domain>) -> ParameterDescription {
        ParameterDescription::new(PortName::new(name).unwrap(), "T", wire, domain, true).unwrap()
    }

    fn quantity() -> PortWire {
        PortWire::Supported(WireType::Quantity(
            QuantityType::new(Dimension::dimensionless(), None).unwrap(),
        ))
    }

    fn closed(lower: f64, upper: f64) -> Bounds<FiniteF64> {
        Bounds::new(Some(finite(lower)), Some(finite(upper))).unwrap()
    }

    /// One bounded parameter of every kind the samplers draw.
    fn sampleable() -> Vec<ParameterDescription> {
        vec![
            parameter("load", quantity(), Some(Domain::Quantity(closed(0.0, 1.0)))),
            parameter(
                "launch",
                PortWire::Supported(WireType::Datetime(DatetimeType::new(TimeScale::Utc))),
                Some(Domain::Datetime(closed(0.0, 86_400.0))),
            ),
            parameter(
                "count",
                PortWire::Supported(WireType::Int),
                Some(Domain::Int(Bounds::new(Some(1), Some(3)).unwrap())),
            ),
            parameter(
                "mode",
                PortWire::Supported(WireType::Key(KeyType::new(named_axis(
                    "Mode",
                    &["Stable", "Fragile"],
                )))),
                None,
            ),
            parameter("enabled", PortWire::Supported(WireType::Bool), None),
        ]
    }

    #[test]
    fn every_parameter_is_sampled_in_declaration_order() {
        let interface = ModelInterface::new(sampleable(), vec![]).unwrap();
        let parameters = sampled_parameters(&interface).unwrap();
        assert_eq!(
            parameters
                .iter()
                .map(|parameter| (parameter.name().as_str(), parameter.kind()))
                .collect::<Vec<_>>(),
            [
                ("load", SampledKind::Quantity),
                ("launch", SampledKind::Datetime),
                ("count", SampledKind::Integer),
                ("mode", SampledKind::Key),
                ("enabled", SampledKind::Boolean),
            ]
        );
    }

    #[test]
    fn a_session_varying_the_sampled_parameters_can_be_sampled() {
        let interface = ModelInterface::new(
            sampleable(),
            vec![graphcal_model_protocol::OutputDescription::new(
                PortName::new("failure").unwrap(),
                "Bool",
                PortWire::Supported(WireType::Bool),
            )],
        )
        .unwrap();
        let inputs = sampled_parameters(&interface)
            .unwrap()
            .into_iter()
            .map(|parameter| parameter.name().clone())
            .collect();
        let spec = SessionSpec::new(inputs, vec![PortName::new("failure").unwrap()]);
        let schema = std::sync::Arc::new(interface.configure(&spec, CanonicalUnitsOnly).unwrap());
        let request = sample_latin_hypercube(&schema, 12, 7, 0).unwrap();
        assert_eq!(request.inputs().rows(), 12);
    }

    #[test]
    fn every_unsampleable_parameter_is_reported_with_its_reason() {
        let half_open = Bounds::new(Some(finite(0.0)), None).unwrap();
        let mut parameters = sampleable();
        parameters.extend([
            parameter("open", quantity(), Some(Domain::Quantity(half_open))),
            parameter("free", PortWire::Supported(WireType::Int), None),
            parameter(
                "impedance",
                PortWire::Supported(WireType::Complex(
                    QuantityType::new(Dimension::dimensionless(), None).unwrap(),
                )),
                None,
            ),
            parameter(
                "money",
                PortWire::Unsupported {
                    reason: UnsupportedReason::NoCanonicalUnit,
                    message: "dimension `Money` has no canonical unit".to_owned(),
                },
                None,
            ),
        ]);
        let interface = ModelInterface::new(parameters, vec![]).unwrap();
        let Err(SampledParametersError::Unsampleable(unsampleable)) =
            sampled_parameters(&interface)
        else {
            panic!("four parameters cannot be sampled");
        };
        assert_eq!(
            unsampleable
                .iter()
                .map(|parameter| (parameter.name().as_str(), parameter.reason().clone()))
                .collect::<Vec<_>>(),
            [
                ("open", UnsampleableReason::Unbounded),
                ("free", UnsampleableReason::Unbounded),
                (
                    "impedance",
                    UnsampleableReason::Wire(UnsampledWire::Complex)
                ),
                (
                    "money",
                    UnsampleableReason::NotCarried {
                        reason: UnsupportedReason::NoCanonicalUnit,
                        message: "dimension `Money` has no canonical unit".to_owned(),
                    }
                ),
            ]
        );
        let message = SampledParametersError::Unsampleable(unsampleable).to_string();
        assert!(
            message.starts_with("4 parameter(s) cannot be sampled: 'open': "),
            "{message}"
        );
    }

    #[test]
    fn an_interface_without_parameters_is_rejected() {
        let interface = ModelInterface::new(vec![], vec![]).unwrap();
        assert_eq!(
            sampled_parameters(&interface),
            Err(SampledParametersError::NoParameters)
        );
    }
}
