//! Conditions that classify a row by the values at several output paths.

use graphcal_model_protocol::{SessionSchema, WireType};
use thiserror::Error;

use crate::case_rule::CaseRule;
use crate::evaluation::OutputRow;
use crate::output_path::OutputPath;

/// What a condition at a path decides for a row that has no value there,
/// because its value uses another variant than the path names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbsentValue {
    /// The condition does not hold. For example, "the P90 latency is at
    /// most 12 h" does not hold for a row whose latency was not evaluated.
    NotACase,
    /// Classification fails, for a value every row is expected to have.
    Reject,
}

/// Errors raised while building or applying a [`CaseCondition`].
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CaseConditionError {
    /// The rule cannot classify values of the path's target.
    #[error("the case rule does not apply to '{path}', which has wire type {wire:?}")]
    InapplicableRule {
        /// The path.
        path: String,
        /// The wire type the path selects.
        wire: Box<WireType>,
    },
    /// A conjunction or disjunction has no condition.
    #[error("a combination of conditions needs at least one condition")]
    NoConditions,
    /// A row has no value at a path whose condition rejects absent values.
    #[error("the row has no value at '{path}'")]
    AbsentValue {
        /// The path.
        path: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
struct At {
    path: OutputPath,
    rule: CaseRule,
    absent: AbsentValue,
}

#[derive(Clone, Debug, PartialEq)]
enum Node {
    At(Box<At>),
    All(Vec<CaseCondition>),
    Any(Vec<CaseCondition>),
    Not(Box<CaseCondition>),
}

/// The negation of a condition: it holds when the condition does not. A row
/// without a value at a path whose condition decides
/// [`AbsentValue::NotACase`] satisfies the negation.
impl std::ops::Not for CaseCondition {
    type Output = Self;

    fn not(self) -> Self {
        Self(Node::Not(Box::new(self)))
    }
}

/// A classification of a row as a case of interest from the values at one
/// or more output paths.
///
/// A condition applies a [`CaseRule`] to the value at an [`OutputPath`], with
/// an explicit decision for rows that have no value there, and combines such
/// conditions by conjunction ([`CaseCondition::all`]), disjunction
/// ([`CaseCondition::any`]), and negation (`!`). Thresholds are in
/// the wire unit of the selected value, as for [`CaseRule`].
///
/// ```
/// use tenax::graphcal_model_protocol::{FiniteF64, SessionSchema};
/// use tenax::{AbsentValue, CaseCondition, CaseRule, Comparison, OutputPath};
///
/// # fn demo(schema: &SessionSchema) -> Result<(), Box<dyn std::error::Error>> {
/// // The P90 latency is available and at most 12 h, and the design is cheap.
/// let fast = CaseCondition::at(
///     OutputPath::parse(schema, "result.latency.Available.p90")?,
///     CaseRule::Quantity {
///         comparison: Comparison::AtMost,
///         threshold: FiniteF64::new(12.0 * 3600.0)?,
///     },
///     AbsentValue::NotACase,
/// )?;
/// let cheap = CaseCondition::at(
///     OutputPath::parse(schema, "result.cost")?,
///     CaseRule::Quantity {
///         comparison: Comparison::Below,
///         threshold: FiniteF64::new(100.0)?,
///     },
///     AbsentValue::Reject,
/// )?;
/// let condition = CaseCondition::all(vec![fast, cheap])?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CaseCondition(Node);

impl CaseCondition {
    /// Applies `rule` to the value at `path`; `absent` decides rows without
    /// a value there.
    ///
    /// # Errors
    ///
    /// Returns [`CaseConditionError::InapplicableRule`] when `rule` cannot
    /// classify values of the path's target, including a key that is not an
    /// entry of the target's axis.
    pub fn at(
        path: OutputPath,
        rule: CaseRule,
        absent: AbsentValue,
    ) -> Result<Self, CaseConditionError> {
        if !rule.applies_to(path.target()) {
            return Err(CaseConditionError::InapplicableRule {
                path: path.to_string(),
                wire: Box::new(path.target().clone()),
            });
        }
        Ok(Self(Node::At(Box::new(At { path, rule, absent }))))
    }

    /// Holds when every condition holds.
    ///
    /// # Errors
    ///
    /// Returns [`CaseConditionError::NoConditions`] for an empty list, which
    /// would otherwise hold vacuously.
    pub fn all(conditions: Vec<Self>) -> Result<Self, CaseConditionError> {
        if conditions.is_empty() {
            return Err(CaseConditionError::NoConditions);
        }
        Ok(Self(Node::All(conditions)))
    }

    /// Holds when at least one condition holds.
    ///
    /// # Errors
    ///
    /// Returns [`CaseConditionError::NoConditions`] for an empty list.
    pub fn any(conditions: Vec<Self>) -> Result<Self, CaseConditionError> {
        if conditions.is_empty() {
            return Err(CaseConditionError::NoConditions);
        }
        Ok(Self(Node::Any(conditions)))
    }

    /// Whether every path of the condition belongs to `schema` (see
    /// [`OutputPath::belongs_to`]).
    #[must_use]
    pub fn belongs_to(&self, schema: &SessionSchema) -> bool {
        self.first_foreign_path(schema).is_none()
    }

    pub(crate) fn first_foreign_path(&self, schema: &SessionSchema) -> Option<&OutputPath> {
        match &self.0 {
            Node::At(at) => (!at.path.belongs_to(schema)).then_some(&at.path),
            Node::All(conditions) | Node::Any(conditions) => conditions
                .iter()
                .find_map(|condition| condition.first_foreign_path(schema)),
            Node::Not(condition) => condition.first_foreign_path(schema),
        }
    }

    /// Classifies the outputs of one successful row. Combinations evaluate
    /// every condition, so a rejected absent value is reported even when
    /// another condition already decides the result.
    ///
    /// The outputs must belong to the session the paths were resolved
    /// against (see [`CaseCondition::belongs_to`]).
    ///
    /// # Errors
    ///
    /// Returns [`CaseConditionError::AbsentValue`] when the row has no value
    /// at a path whose condition rejects absent values.
    pub fn classify(&self, outputs: &OutputRow) -> Result<bool, CaseConditionError> {
        match &self.0 {
            Node::At(at) => match (at.path.get(outputs), at.absent) {
                // The path's target is the wire type the rule applies to,
                // and the row's outputs conform to the session.
                (Some(value), _) => Ok(at.rule.classify(&value).unwrap_or(false)),
                (None, AbsentValue::NotACase) => Ok(false),
                (None, AbsentValue::Reject) => Err(CaseConditionError::AbsentValue {
                    path: at.path.to_string(),
                }),
            },
            Node::All(conditions) => conditions
                .iter()
                .map(|condition| condition.classify(outputs))
                .try_fold(true, |all, holds| Ok(all & holds?)),
            Node::Any(conditions) => conditions
                .iter()
                .map(|condition| condition.classify(outputs))
                .try_fold(false, |any, holds| Ok(any | holds?)),
            Node::Not(condition) => Ok(!condition.classify(outputs)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{FiniteF64, KeyValue, VariantName};

    use super::*;
    use crate::case_rule::Comparison;
    use crate::structured_fixture::{report_row, report_session, report_value};

    fn path(path: &str) -> OutputPath {
        OutputPath::parse(&report_session(), path).unwrap()
    }

    fn p90_at_most(seconds: f64, absent: AbsentValue) -> CaseCondition {
        CaseCondition::at(
            path("report.latency.Available.p90"),
            CaseRule::Quantity {
                comparison: Comparison::AtMost,
                threshold: FiniteF64::new(seconds).unwrap(),
            },
            absent,
        )
        .unwrap()
    }

    fn good() -> CaseCondition {
        CaseCondition::at(
            path("report.grade"),
            CaseRule::KeyIn {
                keys: vec![KeyValue::Variant(VariantName::new("Good").unwrap())],
            },
            AbsentValue::Reject,
        )
        .unwrap()
    }

    #[test]
    fn absent_values_are_decided_explicitly() {
        let fast = report_row(report_value(1.0, Some(30.0), [true, true], "Good"));
        let pending = report_row(report_value(1.0, None, [true, true], "Good"));
        let lenient = p90_at_most(60.0, AbsentValue::NotACase);
        assert_eq!(lenient.classify(&fast), Ok(true));
        assert_eq!(lenient.classify(&pending), Ok(false));
        assert_eq!((!lenient).classify(&pending), Ok(true));
        let not_available = CaseCondition::at(
            path("report.latency.Available"),
            CaseRule::IsFalse,
            AbsentValue::NotACase,
        )
        .unwrap();
        assert_eq!(not_available.classify(&pending), Ok(true));
        assert_eq!(not_available.classify(&fast), Ok(false));
        let strict = p90_at_most(60.0, AbsentValue::Reject);
        assert_eq!(
            strict.classify(&pending),
            Err(CaseConditionError::AbsentValue {
                path: "report.latency.Available.p90".to_owned()
            })
        );
    }

    #[test]
    fn combinations_follow_boolean_logic() {
        let row = report_row(report_value(1.0, Some(90.0), [true, false], "Good"));
        let slow = p90_at_most(60.0, AbsentValue::NotACase);
        assert_eq!(
            CaseCondition::all(vec![good(), slow.clone()])
                .unwrap()
                .classify(&row),
            Ok(false)
        );
        assert_eq!(
            CaseCondition::any(vec![good(), slow])
                .unwrap()
                .classify(&row),
            Ok(true)
        );
        let second = CaseCondition::at(
            path("report.checks.Second"),
            CaseRule::IsTrue,
            AbsentValue::Reject,
        )
        .unwrap();
        assert_eq!((!second).classify(&row), Ok(true));
        assert_eq!(
            CaseCondition::all(Vec::new()),
            Err(CaseConditionError::NoConditions)
        );
        assert_eq!(
            CaseCondition::any(Vec::new()),
            Err(CaseConditionError::NoConditions)
        );
    }

    #[test]
    fn rules_must_apply_to_the_selected_value() {
        assert!(matches!(
            CaseCondition::at(path("report.cost"), CaseRule::IsTrue, AbsentValue::Reject),
            Err(CaseConditionError::InapplicableRule { .. })
        ));
        assert!(matches!(
            CaseCondition::at(
                path("report.grade"),
                CaseRule::KeyIn {
                    keys: vec![KeyValue::Variant(VariantName::new("Unknown").unwrap())]
                },
                AbsentValue::Reject
            ),
            Err(CaseConditionError::InapplicableRule { .. })
        ));
        assert!(
            CaseCondition::at(
                path("report.latency.Available"),
                CaseRule::IsTrue,
                AbsentValue::Reject
            )
            .is_ok()
        );
    }
}
