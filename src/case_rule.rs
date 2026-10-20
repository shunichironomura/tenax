//! Rules that classify one output value as a case of interest.

use graphcal_model_protocol::{FiniteF64, KeyValue, Value, WireType};

/// How a value compares with a threshold, with the boundary stated
/// explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Comparison {
    /// The value is strictly below the threshold.
    Below,
    /// The value is below or equal to the threshold.
    AtMost,
    /// The value is above or equal to the threshold.
    AtLeast,
    /// The value is strictly above the threshold.
    Above,
}

impl Comparison {
    fn holds<T: PartialOrd>(self, value: &T, threshold: &T) -> bool {
        match self {
            Self::Below => value < threshold,
            Self::AtMost => value <= threshold,
            Self::AtLeast => value >= threshold,
            Self::Above => value > threshold,
        }
    }
}

/// A rule that classifies the value of one output as a case of interest.
///
/// Thresholds are stated in the output's wire unit: the canonical SI unit
/// unless the session chose another unit for the output, and seconds since
/// J2000 in the output's time scale for datetimes.
#[derive(Clone, Debug, PartialEq)]
pub enum CaseRule {
    /// A Boolean output is `true`.
    IsTrue,
    /// A quantity output compares with `threshold`.
    Quantity {
        /// How the value compares with the threshold.
        comparison: Comparison,
        /// The threshold, in the output's wire unit.
        threshold: FiniteF64,
    },
    /// A datetime output compares with `threshold`.
    Datetime {
        /// How the value compares with the threshold.
        comparison: Comparison,
        /// The threshold, in seconds since J2000 in the output's time scale.
        threshold: FiniteF64,
    },
    /// An integer output compares with `threshold`.
    Integer {
        /// How the value compares with the threshold.
        comparison: Comparison,
        /// The threshold.
        threshold: i64,
    },
    /// A key output is one of `keys`.
    KeyIn {
        /// The keys that classify a case of interest.
        keys: Vec<KeyValue>,
    },
}

impl CaseRule {
    /// Whether the rule can classify values of `wire`: its kinds agree and
    /// every listed key is an entry of a key output's axis.
    pub(crate) fn applies_to(&self, wire: &WireType) -> bool {
        match (self, wire) {
            (Self::IsTrue, WireType::Bool)
            | (Self::Quantity { .. }, WireType::Quantity(_))
            | (Self::Datetime { .. }, WireType::Datetime(_))
            | (Self::Integer { .. }, WireType::Int) => true,
            (Self::KeyIn { keys }, WireType::Key(key)) => keys
                .iter()
                .all(|value| key.position(value.as_key_ref()).is_some()),
            _ => false,
        }
    }

    /// Classifies one value of an output the rule applies to; `None` when
    /// the value's kind does not match the rule.
    pub(crate) fn classify(&self, value: &Value) -> Option<bool> {
        Some(match (self, value) {
            (Self::IsTrue, Value::Bool(value)) => *value,
            (
                Self::Quantity {
                    comparison,
                    threshold,
                },
                Value::Quantity(value),
            )
            | (
                Self::Datetime {
                    comparison,
                    threshold,
                },
                Value::Datetime(value),
            ) => comparison.holds(&value.get(), &threshold.get()),
            (
                Self::Integer {
                    comparison,
                    threshold,
                },
                Value::Int(value),
            ) => comparison.holds(value, threshold),
            (Self::KeyIn { keys }, Value::Key(value)) => keys.contains(value),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::VariantName;

    use super::*;

    fn finite(value: f64) -> FiniteF64 {
        FiniteF64::new(value).unwrap()
    }

    #[test]
    fn comparisons_state_their_boundary() {
        let at = |comparison| CaseRule::Quantity {
            comparison,
            threshold: finite(1.0),
        };
        let on_boundary = Value::Quantity(finite(1.0));
        assert_eq!(at(Comparison::Below).classify(&on_boundary), Some(false));
        assert_eq!(at(Comparison::AtMost).classify(&on_boundary), Some(true));
        assert_eq!(at(Comparison::AtLeast).classify(&on_boundary), Some(true));
        assert_eq!(at(Comparison::Above).classify(&on_boundary), Some(false));
        assert_eq!(
            at(Comparison::Below).classify(&Value::Quantity(finite(0.5))),
            Some(true)
        );
        assert_eq!(at(Comparison::Below).classify(&Value::Int(0)), None);
    }

    #[test]
    fn integer_and_key_rules_classify_their_kinds() {
        let rule = CaseRule::Integer {
            comparison: Comparison::AtLeast,
            threshold: 3,
        };
        assert_eq!(rule.classify(&Value::Int(3)), Some(true));
        assert_eq!(rule.classify(&Value::Int(2)), Some(false));

        let variant = |name| KeyValue::Variant(VariantName::new(name).unwrap());
        let rule = CaseRule::KeyIn {
            keys: vec![variant("risky")],
        };
        assert_eq!(rule.classify(&Value::Key(variant("risky"))), Some(true));
        assert_eq!(rule.classify(&Value::Key(variant("safe"))), Some(false));
    }
}
