//! Paths that select one value inside a structured output.

use std::borrow::Cow;
use std::fmt;

use graphcal_model_protocol::{
    AlgebraicType, ConstructorPosition, ConstructorType, KeyRef, OutputSlot, PortName,
    SessionSchema, Value, WireType,
};
use thiserror::Error;

use crate::evaluation::OutputRow;
use crate::evaluation_dataset::key_label;

/// Why a path does not select a value of a session's outputs.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OutputPathError {
    /// The session reads no output of this name.
    #[error("the session reads no output named '{name}'")]
    UnknownOutput {
        /// The requested output.
        name: String,
    },
    /// The record or variant has no field of this name.
    #[error("{path}: {constructor} has no field '{field}'")]
    UnknownField {
        /// The path so far.
        path: String,
        /// The record or variant.
        constructor: String,
        /// The requested field.
        field: String,
    },
    /// The union has no variant of this name.
    #[error("{path}: {union} has no variant '{variant}'")]
    UnknownVariant {
        /// The path so far.
        path: String,
        /// The union type.
        union: String,
        /// The requested variant.
        variant: String,
    },
    /// The indexed value has no entry with this key.
    #[error("{path}: index {index} has no entry '{key}'")]
    UnknownKey {
        /// The path so far.
        path: String,
        /// The axis's index.
        index: String,
        /// The requested key.
        key: String,
    },
    /// A union value is selected into without naming its variant first.
    #[error("{path}: {union} is a union; name a variant before its field '{field}'")]
    FieldOfUnion {
        /// The path so far.
        path: String,
        /// The union type.
        union: String,
        /// The requested field.
        field: String,
    },
    /// The step does not apply to the value the path has reached.
    #[error("{path}: cannot select '{step}' inside a {kind} value")]
    NotSelectable {
        /// The path so far.
        path: String,
        /// The requested step.
        step: String,
        /// What the path has reached.
        kind: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq)]
enum Step {
    /// The field at this position of the current constructor.
    Field(usize),
    /// The current union value must use this constructor.
    Variant(ConstructorPosition),
    /// The entry at this position of the current indexed value.
    Entry(usize),
}

/// Where a path has arrived.
#[derive(Clone, Debug, PartialEq)]
enum Cursor {
    /// A value of this wire type.
    Value(WireType),
    /// A union value known to use this constructor; reading it there reads
    /// whether it does.
    Variant(ConstructorType),
}

/// A path from a session output to one value inside it, checked against the
/// output's wire type.
///
/// A path selects a field of a record (a one-constructor algebraic type), a
/// variant of a union and then one of its fields, or an entry of an indexed
/// value by key. A path that ends at a variant reads whether the union value
/// uses that variant, so its target is Boolean. A value that uses another
/// variant than a step names has no value at the path: reading it gives
/// `None`, never a placeholder.
///
/// ```
/// use tenax::OutputPath;
/// # use tenax::graphcal_model_protocol::{SessionSchema};
/// # fn demo(schema: &SessionSchema) -> Result<(), tenax::OutputPathError> {
/// // `result` is a record with a `latency` union of `Available(p90: Time)`
/// // and `NotEvaluated`.
/// let p90 = OutputPath::new(schema, "result")?
///     .field("latency")?
///     .variant("Available")?
///     .field("p90")?;
/// let same = OutputPath::parse(schema, "result.latency.Available.p90")?;
/// assert_eq!(p90, same);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct OutputPath {
    slot: OutputSlot,
    output: PortName,
    output_wire: WireType,
    steps: Vec<Step>,
    labels: Vec<String>,
    cursor: Cursor,
}

impl OutputPath {
    /// Starts a path at the output `name` of `schema`.
    ///
    /// # Errors
    ///
    /// Returns [`OutputPathError::UnknownOutput`] when the session does not
    /// read the output.
    pub fn new(schema: &SessionSchema, name: &str) -> Result<Self, OutputPathError> {
        let unknown = || OutputPathError::UnknownOutput {
            name: name.to_owned(),
        };
        let slot = schema.output_slot(name).ok_or_else(unknown)?;
        let output = schema.output(slot).ok_or_else(unknown)?;
        Ok(Self {
            slot,
            output: output.name().clone(),
            output_wire: output.wire().clone(),
            steps: Vec::new(),
            labels: Vec::new(),
            cursor: Cursor::Value(output.wire().clone()),
        })
    }

    /// Parses a dot-separated path: the output name, then one step per
    /// segment, interpreted by the type the path has reached: a field of a
    /// record or of the variant just named, a variant of a union, or the key
    /// of an indexed value, written as its variant name or `Fin` position.
    ///
    /// This is a convenience for paths written in configuration files.
    /// Coordinate keys contain dots, so select them with
    /// [`OutputPath::entry`].
    ///
    /// # Errors
    ///
    /// Returns [`OutputPathError`] for the first segment that does not apply.
    pub fn parse(schema: &SessionSchema, path: &str) -> Result<Self, OutputPathError> {
        let mut segments = path.split('.');
        let output = segments.next().unwrap_or_default();
        segments.try_fold(Self::new(schema, output)?, |path, segment| {
            match &path.cursor {
                Cursor::Value(WireType::Algebraic(algebraic))
                    if algebraic.constructors().len() > 1 =>
                {
                    path.variant(segment)
                }
                Cursor::Value(WireType::Indexed(indexed)) => {
                    let axis = indexed.axis();
                    let key = (0..axis.len())
                        .filter_map(|position| axis.key(position))
                        .find(|key| key_label(*key) == segment)
                        .map(KeyRef::to_key_value);
                    match key {
                        Some(key) => path.entry(key.as_key_ref()),
                        None => Err(OutputPathError::UnknownKey {
                            path: path.to_string(),
                            index: axis.index().to_string(),
                            key: segment.to_owned(),
                        }),
                    }
                }
                // A record, or the variant just named.
                Cursor::Value(_) | Cursor::Variant(_) => path.field(segment),
            }
        })
    }

    /// Selects the field `name` of the record the path has reached, or of
    /// the variant it has just named.
    ///
    /// # Errors
    ///
    /// Returns [`OutputPathError`] when the path is not at a record or a
    /// named variant, or the field does not exist.
    pub fn field(mut self, name: &str) -> Result<Self, OutputPathError> {
        let constructor = match &self.cursor {
            Cursor::Variant(constructor) => constructor.clone(),
            Cursor::Value(WireType::Algebraic(algebraic)) => match algebraic.constructors() {
                [record] => record.clone(),
                _ => {
                    return Err(OutputPathError::FieldOfUnion {
                        path: self.to_string(),
                        union: algebraic.name().to_string(),
                        field: name.to_owned(),
                    });
                }
            },
            Cursor::Value(wire) => return Err(self.not_selectable(name, wire)),
        };
        let position = constructor
            .fields()
            .iter()
            .position(|field| field.name().as_str() == name)
            .ok_or_else(|| OutputPathError::UnknownField {
                path: self.to_string(),
                constructor: constructor.name().to_string(),
                field: name.to_owned(),
            })?;
        self.steps.push(Step::Field(position));
        self.labels.push(name.to_owned());
        self.cursor = Cursor::Value(constructor.fields()[position].wire().clone());
        Ok(self)
    }

    /// Requires the union the path has reached to use the variant `name`.
    /// A path that ends here reads whether it does.
    ///
    /// # Errors
    ///
    /// Returns [`OutputPathError`] when the path is not at an algebraic value
    /// or the variant does not exist.
    pub fn variant(mut self, name: &str) -> Result<Self, OutputPathError> {
        let Cursor::Value(WireType::Algebraic(algebraic)) = &self.cursor else {
            let kind = match &self.cursor {
                Cursor::Value(wire) => wire_kind(wire),
                Cursor::Variant(_) => "variant",
            };
            return Err(OutputPathError::NotSelectable {
                path: self.to_string(),
                step: name.to_owned(),
                kind,
            });
        };
        let (position, constructor) =
            variant(algebraic, name).ok_or_else(|| OutputPathError::UnknownVariant {
                path: self.to_string(),
                union: algebraic.name().to_string(),
                variant: name.to_owned(),
            })?;
        self.steps.push(Step::Variant(position));
        self.labels.push(name.to_owned());
        self.cursor = Cursor::Variant(constructor);
        Ok(self)
    }

    /// Selects the entry with `key` of the indexed value the path has
    /// reached.
    ///
    /// # Errors
    ///
    /// Returns [`OutputPathError`] when the path is not at an indexed value
    /// or the axis has no such entry.
    pub fn entry(mut self, key: KeyRef<'_>) -> Result<Self, OutputPathError> {
        let Cursor::Value(WireType::Indexed(indexed)) = &self.cursor else {
            let kind = match &self.cursor {
                Cursor::Value(wire) => wire_kind(wire),
                Cursor::Variant(_) => "variant",
            };
            return Err(OutputPathError::NotSelectable {
                path: self.to_string(),
                step: key_label(key),
                kind,
            });
        };
        let position = indexed
            .axis()
            .position(key)
            .ok_or_else(|| OutputPathError::UnknownKey {
                path: self.to_string(),
                index: indexed.axis().index().to_string(),
                key: key_label(key),
            })?;
        let element = indexed.element().clone();
        self.steps.push(Step::Entry(position));
        self.labels.push(key_label(key));
        self.cursor = Cursor::Value(element);
        Ok(self)
    }

    /// Returns the output the path starts at.
    #[must_use]
    pub const fn output(&self) -> OutputSlot {
        self.slot
    }

    /// Returns the wire type of the selected value: Boolean for a path that
    /// ends at a variant.
    #[must_use]
    pub const fn target(&self) -> &WireType {
        match &self.cursor {
            Cursor::Value(wire) => wire,
            Cursor::Variant(_) => &WireType::Bool,
        }
    }

    /// Returns whether `schema` reads the path's output with the wire type
    /// the path was checked against.
    #[must_use]
    pub fn belongs_to(&self, schema: &SessionSchema) -> bool {
        schema.output(self.slot).is_some_and(|output| {
            output.name() == &self.output && output.wire() == &self.output_wire
        })
    }

    /// Reads the selected value from the outputs of one successful row, or
    /// `None` when the row's value uses another variant than the path names
    /// or the row has no such output.
    #[must_use]
    pub fn get<'v>(&self, outputs: &'v OutputRow) -> Option<Cow<'v, Value>> {
        self.get_in(outputs.value(self.slot)?)
    }

    /// Reads the selected value from a value of the path's output; see
    /// [`OutputPath::get`].
    #[must_use]
    pub fn get_in<'v>(&self, output: &'v Value) -> Option<Cow<'v, Value>> {
        let mut current = output;
        for (position, step) in self.steps.iter().enumerate() {
            match (step, current) {
                (Step::Field(field), Value::Algebraic { fields, .. }) => {
                    current = fields.get(*field)?;
                }
                (Step::Variant(expected), Value::Algebraic { constructor, .. }) => {
                    if position + 1 == self.steps.len() {
                        return Some(Cow::Owned(Value::Bool(constructor == expected)));
                    }
                    if constructor != expected {
                        return None;
                    }
                }
                (Step::Entry(entry), Value::Indexed(entries)) => current = entries.get(*entry)?,
                _ => return None,
            }
        }
        Some(Cow::Borrowed(current))
    }

    fn not_selectable(&self, step: &str, wire: &WireType) -> OutputPathError {
        OutputPathError::NotSelectable {
            path: self.to_string(),
            step: step.to_owned(),
            kind: wire_kind(wire),
        }
    }
}

impl fmt::Display for OutputPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.output.as_str())?;
        self.labels
            .iter()
            .try_for_each(|label| write!(formatter, ".{label}"))
    }
}

fn variant(
    algebraic: &AlgebraicType,
    name: &str,
) -> Option<(ConstructorPosition, ConstructorType)> {
    let position = algebraic.position(name)?;
    let constructor = algebraic.constructor(position)?.clone();
    Some((position, constructor))
}

const fn wire_kind(wire: &WireType) -> &'static str {
    match wire {
        WireType::Quantity(_) => "quantity",
        WireType::Complex(_) => "complex",
        WireType::Int => "integer",
        WireType::Bool => "Boolean",
        WireType::Key(_) => "key",
        WireType::Datetime(_) => "datetime",
        WireType::Indexed(_) => "indexed",
        WireType::Algebraic(_) => "algebraic",
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{FiniteF64, KeyValue, VariantName};

    use super::*;
    use crate::structured_fixture::{report_row, report_session, report_value};

    fn quantity(value: f64) -> Value {
        Value::Quantity(FiniteF64::new(value).unwrap())
    }

    #[test]
    fn built_and_parsed_paths_agree() {
        let schema = report_session();
        let built = OutputPath::new(&schema, "report")
            .unwrap()
            .field("latency")
            .unwrap()
            .variant("Available")
            .unwrap()
            .field("p90")
            .unwrap();
        assert_eq!(
            OutputPath::parse(&schema, "report.latency.Available.p90").unwrap(),
            built
        );
        assert_eq!(built.to_string(), "report.latency.Available.p90");
        assert!(matches!(built.target(), WireType::Quantity(_)));
        assert!(built.belongs_to(&schema));
    }

    #[test]
    fn reads_only_the_named_variant() {
        let schema = report_session();
        let p90 = OutputPath::parse(&schema, "report.latency.Available.p90").unwrap();
        let available = report_row(report_value(1.0, Some(60.0), [true, false], "Good"));
        let pending = report_row(report_value(1.0, None, [true, false], "Good"));
        assert_eq!(p90.get(&available).as_deref(), Some(&quantity(60.0)));
        assert_eq!(p90.get(&pending), None);

        let is_available = OutputPath::parse(&schema, "report.latency.Available").unwrap();
        assert_eq!(is_available.target(), &WireType::Bool);
        assert_eq!(
            is_available.get(&available).as_deref(),
            Some(&Value::Bool(true))
        );
        assert_eq!(
            is_available.get(&pending).as_deref(),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn reads_record_fields_entries_and_keys() {
        let schema = report_session();
        let row = report_row(report_value(124.0, None, [true, false], "Bad"));
        let read = |path: &str| {
            OutputPath::parse(&schema, path)
                .unwrap()
                .get(&row)
                .map(std::borrow::Cow::into_owned)
        };
        assert_eq!(read("report.cost"), Some(quantity(124.0)));
        assert_eq!(read("report.checks.Second"), Some(Value::Bool(false)));
        assert_eq!(
            read("report.grade"),
            Some(Value::Key(KeyValue::Variant(
                VariantName::new("Bad").unwrap()
            )))
        );
        let entry = OutputPath::new(&schema, "report")
            .unwrap()
            .field("checks")
            .unwrap()
            .entry(KeyValue::Variant(VariantName::new("First").unwrap()).as_key_ref())
            .unwrap();
        assert_eq!(entry.get(&row).as_deref(), Some(&Value::Bool(true)));
    }

    #[test]
    fn rejects_steps_that_do_not_apply() {
        let schema = report_session();
        let cases = [
            ("missing", "the session reads no output named 'missing'"),
            ("report.nope", "report: Report has no field 'nope'"),
            (
                "report.latency.p90",
                "report.latency: Latency has no variant 'p90'",
            ),
            (
                "report.latency.Late",
                "report.latency: Latency has no variant 'Late'",
            ),
            (
                "report.latency.Available.p50",
                "report.latency.Available: Available has no field 'p50'",
            ),
            (
                "report.checks.Third",
                "report.checks: index Check has no entry 'Third'",
            ),
            (
                "report.cost.more",
                "report.cost: cannot select 'more' inside a quantity value",
            ),
        ];
        for (path, message) in cases {
            assert_eq!(
                OutputPath::parse(&schema, path).unwrap_err().to_string(),
                message,
                "{path}"
            );
        }
        assert!(matches!(
            OutputPath::new(&schema, "report")
                .unwrap()
                .field("latency")
                .unwrap()
                .field("p90"),
            Err(OutputPathError::FieldOfUnion { .. })
        ));
    }
}
