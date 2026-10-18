//! Zero-copy, typed access to one row of a protocol request batch.

use std::fmt;

use graphcal_model_protocol::{
    Bounds, Domain, InputColumn, InputSlot, KeyRef, PortName, RequestBatch,
    RowFailure as ProtocolRowFailure, ValueRef,
};
use thiserror::Error;

/// The kind of one protocol value, used to report access mismatches.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ValueKind {
    /// A finite real quantity.
    Quantity,
    /// A signed integer.
    Int,
    /// A Boolean.
    Bool,
    /// A categorical key variant.
    Key,
    /// A complex quantity.
    Complex,
    /// A datetime, as seconds since J2000.
    Datetime,
    /// An indexed value.
    Indexed,
    /// A record or tagged union.
    Algebraic,
}

impl ValueKind {
    /// Returns the kind of a borrowed protocol value.
    #[must_use]
    pub const fn of(value: &ValueRef<'_>) -> Self {
        match value {
            ValueRef::Quantity(_) => Self::Quantity,
            ValueRef::Int(_) => Self::Int,
            ValueRef::Bool(_) => Self::Bool,
            ValueRef::Key(_) => Self::Key,
            ValueRef::Complex(_) => Self::Complex,
            ValueRef::Datetime(_) => Self::Datetime,
            ValueRef::Indexed(_) => Self::Indexed,
            ValueRef::Algebraic { .. } => Self::Algebraic,
        }
    }
}

impl fmt::Display for ValueKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Quantity => "quantity",
            Self::Int => "integer",
            Self::Bool => "Boolean",
            Self::Key => "key",
            Self::Complex => "complex",
            Self::Datetime => "datetime",
            Self::Indexed => "indexed",
            Self::Algebraic => "algebraic",
        })
    }
}

/// Errors raised when reading a value from an [`InputRow`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum InputAccessError {
    /// The requested row does not exist in the batch.
    #[error("row index {index} is out of bounds for a batch with {row_count} rows")]
    RowIndexOutOfBounds {
        /// Requested zero-based row position.
        index: usize,
        /// Number of rows in the batch.
        row_count: usize,
    },

    /// The slot does not belong to the batch's session schema.
    #[error("input slot {slot} is out of bounds for a session with {input_count} inputs")]
    UnknownSlot {
        /// Requested zero-based input slot.
        slot: usize,
        /// Number of inputs the session varies.
        input_count: usize,
    },

    /// The input has a different kind from the one requested.
    #[error("input '{input}' is a {actual} input, not a {expected} input")]
    KindMismatch {
        /// The accessed input.
        input: PortName,
        /// The kind the caller requested.
        expected: ValueKind,
        /// The input's wire kind.
        actual: ValueKind,
    },
}

/// Reports an input access error as a model evaluation error, so an in-process
/// model can propagate it with `?`.
impl From<InputAccessError> for ProtocolRowFailure {
    fn from(error: InputAccessError) -> Self {
        Self::EvaluationError {
            message: error.to_string(),
        }
    }
}

/// A zero-copy view of one row of a [`RequestBatch`].
///
/// Values are addressed by the [`InputSlot`]s of the batch's session schema,
/// obtained once with [`graphcal_model_protocol::SessionSchema::input_slot`].
#[derive(Clone, Copy, Debug)]
pub struct InputRow<'a> {
    batch: &'a RequestBatch,
    index: usize,
}

impl<'a> InputRow<'a> {
    /// Borrows row `index` of `batch`.
    ///
    /// # Errors
    ///
    /// Returns [`InputAccessError::RowIndexOutOfBounds`] when `index` is
    /// outside the batch.
    pub const fn new(batch: &'a RequestBatch, index: usize) -> Result<Self, InputAccessError> {
        if index < batch.rows() {
            Ok(Self { batch, index })
        } else {
            Err(InputAccessError::RowIndexOutOfBounds {
                index,
                row_count: batch.rows(),
            })
        }
    }

    /// Iterates over every row of `batch` in order.
    pub fn all(batch: &'a RequestBatch) -> impl ExactSizeIterator<Item = Self> {
        (0..batch.rows()).map(move |index| Self { batch, index })
    }

    pub(crate) fn in_bounds(batch: &'a RequestBatch, index: usize) -> Self {
        debug_assert!(index < batch.rows());
        Self { batch, index }
    }

    /// Returns the row's zero-based position within its batch.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Returns the borrowed request batch.
    #[must_use]
    pub const fn batch(&self) -> &'a RequestBatch {
        self.batch
    }

    /// Returns the value of the input at `slot`.
    ///
    /// # Errors
    ///
    /// Returns [`InputAccessError::UnknownSlot`] when `slot` does not belong to
    /// the batch's session schema.
    pub fn value(&self, slot: InputSlot) -> Result<ValueRef<'a>, InputAccessError> {
        self.batch
            .value(slot, self.index)
            .ok_or_else(|| self.unknown_slot(slot))
    }

    /// Returns the quantity at `slot`, in the input's wire unit.
    ///
    /// # Errors
    ///
    /// Returns [`InputAccessError`] for an unknown slot or a non-quantity
    /// input.
    pub fn quantity(&self, slot: InputSlot) -> Result<f64, InputAccessError> {
        match self.value(slot)? {
            ValueRef::Quantity(value) => Ok(value.get()),
            value => Err(self.mismatch(slot, ValueKind::Quantity, &value)),
        }
    }

    /// Returns the integer at `slot`.
    ///
    /// # Errors
    ///
    /// Returns [`InputAccessError`] for an unknown slot or a non-integer input.
    pub fn int(&self, slot: InputSlot) -> Result<i64, InputAccessError> {
        match self.value(slot)? {
            ValueRef::Int(value) => Ok(value),
            value => Err(self.mismatch(slot, ValueKind::Int, &value)),
        }
    }

    /// Returns the Boolean at `slot`.
    ///
    /// # Errors
    ///
    /// Returns [`InputAccessError`] for an unknown slot or a non-Boolean input.
    pub fn bool(&self, slot: InputSlot) -> Result<bool, InputAccessError> {
        match self.value(slot)? {
            ValueRef::Bool(value) => Ok(value),
            value => Err(self.mismatch(slot, ValueKind::Bool, &value)),
        }
    }

    /// Returns the key at `slot`: a variant name, a `Fin` position, or an SI
    /// coordinate, depending on the key's index.
    ///
    /// # Errors
    ///
    /// Returns [`InputAccessError`] for an unknown slot or a non-key input.
    pub fn key(&self, slot: InputSlot) -> Result<KeyRef<'a>, InputAccessError> {
        match self.value(slot)? {
            ValueRef::Key(key) => Ok(key),
            value => Err(self.mismatch(slot, ValueKind::Key, &value)),
        }
    }

    fn unknown_slot(&self, slot: InputSlot) -> InputAccessError {
        InputAccessError::UnknownSlot {
            slot: slot.get(),
            input_count: self.batch.schema().inputs().len(),
        }
    }

    fn mismatch(
        &self,
        slot: InputSlot,
        expected: ValueKind,
        actual: &ValueRef<'_>,
    ) -> InputAccessError {
        self.batch.schema().input(slot).map_or_else(
            || self.unknown_slot(slot),
            |input| InputAccessError::KindMismatch {
                input: input.name().clone(),
                expected,
                actual: ValueKind::of(actual),
            },
        )
    }

    /// Returns the first input of this row, in session order, whose value lies
    /// outside its declared domain, reported with the protocol's
    /// [`ProtocolRowFailure::InputOutOfDomain`] vocabulary.
    pub(crate) fn domain_violation(&self) -> Option<ProtocolRowFailure> {
        self.batch
            .schema()
            .inputs()
            .iter()
            .zip(self.batch.columns())
            .find_map(|(input, column)| {
                let message = match (input.domain()?, column) {
                    (Domain::Quantity(bounds), InputColumn::Quantity(values)) => {
                        violation(bounds, values.get(self.index)?)
                    }
                    (Domain::Int(bounds), InputColumn::Int(values)) => {
                        violation(bounds, values.get(self.index)?)
                    }
                    _ => None,
                }?;
                Some(ProtocolRowFailure::InputOutOfDomain {
                    parameter: input.name().clone(),
                    message,
                })
            })
    }
}

fn violation<T: PartialOrd + fmt::Display>(bounds: &Bounds<T>, value: &T) -> Option<String> {
    if bounds.contains(value) {
        return None;
    }
    Some(match (bounds.min(), bounds.max()) {
        (Some(min), Some(max)) => format!("value {value} lies outside [{min}, {max}]"),
        (Some(min), None) => format!("value {value} is less than the minimum {min}"),
        (None, Some(max)) => format!("value {value} is greater than the maximum {max}"),
        (None, None) => format!("value {value} lies outside its domain"),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graphcal_model_protocol::{
        FiniteF64, KeyPosition, KeyRef, SessionSchema, VariantName, WireType,
    };

    use super::*;
    use crate::ModelSchemaBuilder;

    fn schema() -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .categorical("mode", "Mode", ["safe", "risky"])
            .unwrap()
            .integer("n", 0, 3)
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap()
    }

    fn batch(schema: &Arc<SessionSchema>, x: f64, n: i64) -> RequestBatch {
        let WireType::Key(mode) = schema.inputs()[1].wire() else {
            panic!("mode must be a key input");
        };
        let risky = VariantName::new("risky").unwrap();
        let risky: KeyPosition = mode.position(KeyRef::Variant(&risky)).unwrap();
        RequestBatch::new(
            Arc::clone(schema),
            vec![
                InputColumn::Quantity(vec![FiniteF64::new(x).unwrap()]),
                InputColumn::Key(vec![risky]),
                InputColumn::Int(vec![n]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn rows_expose_typed_values_without_copying() {
        let schema = schema();
        let x = schema.input_slot("x").unwrap();
        let mode = schema.input_slot("mode").unwrap();
        let n = schema.input_slot("n").unwrap();
        let batch = batch(&schema, 0.25, 2);
        let row = InputRow::new(&batch, 0).unwrap();
        assert_eq!(row.quantity(x), Ok(0.25));
        assert!(matches!(
            row.key(mode).unwrap(),
            KeyRef::Variant(variant) if variant.as_str() == "risky"
        ));
        assert_eq!(row.int(n), Ok(2));
        assert_eq!(InputRow::all(&batch).len(), 1);
    }

    #[test]
    fn kind_mismatches_and_bad_rows_are_typed_errors() {
        let schema = schema();
        let x = schema.input_slot("x").unwrap();
        let batch = batch(&schema, 0.25, 2);
        let row = InputRow::new(&batch, 0).unwrap();
        assert_eq!(
            row.bool(x),
            Err(InputAccessError::KindMismatch {
                input: PortName::new("x").unwrap(),
                expected: ValueKind::Bool,
                actual: ValueKind::Quantity,
            })
        );
        assert_eq!(
            InputRow::new(&batch, 1).unwrap_err(),
            InputAccessError::RowIndexOutOfBounds {
                index: 1,
                row_count: 1
            }
        );
        assert!(matches!(
            ProtocolRowFailure::from(row.int(x).unwrap_err()),
            ProtocolRowFailure::EvaluationError { .. }
        ));
    }

    #[test]
    fn reports_the_first_out_of_domain_input() {
        let schema = schema();
        let inside = batch(&schema, 1.0, 3);
        assert_eq!(InputRow::new(&inside, 0).unwrap().domain_violation(), None);

        let outside = batch(&schema, 1.5, 4);
        assert!(matches!(
            InputRow::new(&outside, 0).unwrap().domain_violation(),
            Some(ProtocolRowFailure::InputOutOfDomain { parameter, .. })
                if parameter.as_str() == "x"
        ));
    }
}
