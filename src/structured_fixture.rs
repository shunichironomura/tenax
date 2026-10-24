//! A session reading one structured output, for unit tests of paths and
//! case conditions.

use std::sync::Arc;

use graphcal_model_protocol::{
    AlgebraicType, Axis, AxisKind, BaseDimensionName, Bounds, CanonicalUnitsOnly, ConstructorName,
    ConstructorPosition, ConstructorType, Dimension, Domain, Exponent, FieldName, FieldType,
    FiniteF64, IndexName, IndexedType, KeyType, KeyValue, ModelInterface, OutputDescription,
    ParameterDescription, PortName, PortWire, QuantityType, SessionSchema, SessionSpec, TypeName,
    UnitSpelling, Value, VariantName, WireType,
};

use crate::evaluation::OutputRow;

fn quantity(dimension: &str, unit: &str) -> WireType {
    let dimension = Dimension::from_iter([(
        BaseDimensionName::new(dimension).unwrap(),
        Exponent::new(1, 1).unwrap(),
    )]);
    WireType::Quantity(
        QuantityType::new(dimension, Some(UnitSpelling::new(unit).unwrap())).unwrap(),
    )
}

fn field(name: &str, wire: WireType) -> FieldType {
    FieldType::new(FieldName::new(name).unwrap(), wire)
}

fn constructor(name: &str, fields: Vec<FieldType>) -> ConstructorType {
    ConstructorType::new(ConstructorName::new(name).unwrap(), fields)
}

fn algebraic(name: &str, constructors: Vec<ConstructorType>) -> WireType {
    WireType::Algebraic(AlgebraicType::new(TypeName::new(name).unwrap(), constructors).unwrap())
}

fn named_axis(index: &str, variants: &[&str]) -> Axis {
    Axis::new(
        IndexName::new(index).unwrap(),
        AxisKind::Named {
            variants: variants
                .iter()
                .map(|variant| VariantName::new(*variant).unwrap())
                .collect(),
        },
    )
    .unwrap()
}

/// `Report(cost: Mass, latency: Latency, checks: Bool[Check], grade:
/// Key<Grade>)` with `Latency = Available(p90: Time) | Pending`.
fn report_wire() -> WireType {
    let latency = algebraic(
        "Latency",
        vec![
            constructor("Available", vec![field("p90", quantity("Time", "s"))]),
            constructor("Pending", vec![]),
        ],
    );
    algebraic(
        "Report",
        vec![constructor(
            "Report",
            vec![
                field("cost", quantity("Mass", "kg")),
                field("latency", latency),
                field(
                    "checks",
                    WireType::Indexed(IndexedType::new(
                        named_axis("Check", &["First", "Second"]),
                        WireType::Bool,
                    )),
                ),
                field(
                    "grade",
                    WireType::Key(KeyType::new(named_axis("Grade", &["Good", "Bad"]))),
                ),
            ],
        )],
    )
}

/// A session varying `x` in `[0, 1]` and reading `report`.
pub fn report_session() -> Arc<SessionSchema> {
    let dimensionless =
        WireType::Quantity(QuantityType::new(Dimension::dimensionless(), None).unwrap());
    let x = ParameterDescription::new(
        PortName::new("x").unwrap(),
        "Dimensionless",
        PortWire::Supported(dimensionless),
        Some(Domain::Quantity(
            Bounds::new(
                Some(FiniteF64::new(0.0).unwrap()),
                Some(FiniteF64::new(1.0).unwrap()),
            )
            .unwrap(),
        )),
        false,
    )
    .unwrap();
    let report = OutputDescription::new(
        PortName::new("report").unwrap(),
        "Report",
        PortWire::Supported(report_wire()),
    );
    let interface = ModelInterface::new(vec![x], vec![report]).unwrap();
    let spec = SessionSpec::new(
        vec![PortName::new("x").unwrap()],
        vec![PortName::new("report").unwrap()],
    );
    Arc::new(interface.configure(&spec, CanonicalUnitsOnly).unwrap())
}

/// A `report` value; a `p90` of `None` uses the `Pending` variant.
pub fn report_value(cost: f64, p90: Option<f64>, checks: [bool; 2], grade: &str) -> Value {
    let latency = p90.map_or_else(
        || Value::Algebraic {
            constructor: ConstructorPosition::new(1),
            fields: vec![],
        },
        |p90| Value::Algebraic {
            constructor: ConstructorPosition::new(0),
            fields: vec![Value::Quantity(FiniteF64::new(p90).unwrap())],
        },
    );
    Value::Algebraic {
        constructor: ConstructorPosition::new(0),
        fields: vec![
            Value::Quantity(FiniteF64::new(cost).unwrap()),
            latency,
            Value::Indexed(checks.map(Value::Bool).to_vec()),
            Value::Key(KeyValue::Variant(VariantName::new(grade).unwrap())),
        ],
    }
}

/// The outputs of one row whose `report` is `value`.
pub fn report_row(value: Value) -> OutputRow {
    OutputRow::new(&report_session(), vec![value]).unwrap()
}
