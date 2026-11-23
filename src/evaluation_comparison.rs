//! Comparing two evaluation tables of the same requests output by output.

use std::collections::{BTreeMap, BTreeSet};

use graphcal_model_protocol::{
    Axis, ConstructorType, FieldType, OutputSlot, SessionOutput, Value, WireType,
};
use thiserror::Error;

use crate::evaluation::{EvaluationId, OutputRow, RowOutcome};
use crate::evaluation_dataset::key_label;
use crate::evaluation_table::{EvaluatedRequest, EvaluationTable};
use crate::output_path::{OutputPath, Step};

/// Why two tables cannot be compared.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum TableComparisonError {
    /// The tables belong to different session schemas.
    #[error("the tables belong to different sessions")]
    SchemaMismatch,

    /// Only one of the tables holds an evaluation with this identifier.
    #[error("evaluation {id} is in only one of the tables")]
    UnmatchedEvaluation {
        /// The smallest such identifier.
        id: EvaluationId,
    },

    /// The tables hold different requests under the same identifier.
    #[error("evaluation {id} has different requests in the two tables")]
    RequestMismatch {
        /// The evaluation's identifier.
        id: EvaluationId,
    },
}

/// How far the values at one output path drift between two tables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Deviation {
    /// Quantity or complex values that differ in their bits, with the largest
    /// relative difference `|a - b| / max(|a|, |b|)` over the differing rows.
    ///
    /// The difference lies in `[0, 2]`. It is zero only for `0.0` against
    /// `-0.0`, whose bits differ although the values are equal.
    Relative(f64),

    /// Values without a numeric distance that differ: integers, Booleans,
    /// keys, datetimes, or unions that use different variants.
    Unequal,
}

impl Deviation {
    const fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Relative(a), Self::Relative(b)) => Self::Relative(a.max(b)),
            _ => Self::Unequal,
        }
    }
}

/// The rows in which the value at one output path differs.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueDifference {
    path: OutputPath,
    rows: usize,
    deviation: Deviation,
}

impl ValueDifference {
    /// Returns the path of the differing value. Entries of indexed values
    /// and fields of the variant both tables use are compared separately; a
    /// union whose variant differs is reported at the union itself.
    #[must_use]
    pub const fn path(&self) -> &OutputPath {
        &self.path
    }

    /// Returns the number of rows in which the value differs.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns how far the value drifts over those rows.
    #[must_use]
    pub const fn deviation(&self) -> Deviation {
        self.deviation
    }
}

/// How the outcomes of two evaluation tables of the same requests differ.
///
/// Values are compared bit for bit, so a table compared with itself, or
/// with a bit-identical reevaluation, has no difference. Produced by
/// [`EvaluationTable::compare`].
#[derive(Clone, Debug, PartialEq)]
pub struct TableComparison {
    rows: usize,
    identical_rows: usize,
    outcome_mismatches: usize,
    differences: Vec<ValueDifference>,
}

impl TableComparison {
    /// Returns the number of rows compared.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of rows whose outcomes are bit-for-bit identical.
    #[must_use]
    pub const fn identical_rows(&self) -> usize {
        self.identical_rows
    }

    /// Returns the number of rows that succeeded in both tables with
    /// different output values.
    #[must_use]
    pub const fn value_mismatches(&self) -> usize {
        self.rows - self.identical_rows - self.outcome_mismatches
    }

    /// Returns the number of rows that failed in at least one table and
    /// whose outcomes differ: success against failure, or different
    /// failures.
    #[must_use]
    pub const fn outcome_mismatches(&self) -> usize {
        self.outcome_mismatches
    }

    /// Returns the differing values of rows that succeeded in both tables,
    /// one entry per path, ordered by output and then by position within it.
    #[must_use]
    pub fn differences(&self) -> &[ValueDifference] {
        &self.differences
    }

    /// Returns whether every row is bit-for-bit identical.
    #[must_use]
    pub const fn is_identical(&self) -> bool {
        self.identical_rows == self.rows
    }

    /// Returns the largest relative difference of any quantity or complex
    /// value, or `None` when no such value differs.
    #[must_use]
    pub fn max_relative_difference(&self) -> Option<f64> {
        self.differences
            .iter()
            .filter_map(|difference| match difference.deviation {
                Deviation::Relative(relative) => Some(relative),
                Deviation::Unequal => None,
            })
            .reduce(f64::max)
    }
}

impl EvaluationTable {
    /// Compares every output of every row of `other` with this table.
    ///
    /// Evaluations are paired by identifier, so the tables may hold them in
    /// different orders, as tables filled in completion order do. This is
    /// meant for checking that two evaluations of the same requests agree,
    /// such as a remote evaluation against a local one, or a run against a
    /// stored reference.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenax::graphcal_model_protocol::{Dimension, FiniteF64, QuantityType, Value};
    /// use tenax::{
    ///     EvaluationTable, Evaluator, InProcessEvaluator, InputRow, ModelSchemaBuilder,
    ///     RowContext, sample_uniform,
    /// };
    ///
    /// let schema = ModelSchemaBuilder::new()
    ///     .continuous("x", 1.0, 2.0)?
    ///     .quantity_output("y", QuantityType::new(Dimension::dimensionless(), None)?)?
    ///     .build()?;
    /// let x = schema.input_slot("x").expect("x is an input");
    /// let request = sample_uniform(&schema, 10, 7, 0)?;
    /// let table = |scale: f64| -> Result<EvaluationTable, Box<dyn std::error::Error>> {
    ///     let evaluator = InProcessEvaluator::new(
    ///         Arc::clone(&schema),
    ///         move |row: InputRow<'_>, _: RowContext| {
    ///             let y = FiniteF64::new(row.quantity(x)? * scale).expect("finite");
    ///             Ok(vec![Value::Quantity(y)])
    ///         },
    ///     );
    ///     let result = evaluator
    ///         .evaluate(vec![request.clone()])
    ///         .next()
    ///         .expect("one request produces one result")?;
    ///     let mut table = EvaluationTable::new(Arc::clone(&schema));
    ///     table.push(request.clone(), result)?;
    ///     Ok(table)
    /// };
    ///
    /// let reference = table(1.0)?;
    /// assert!(reference.compare(&table(1.0)?)?.is_identical());
    /// let drifted = reference.compare(&table(1.0 + 1e-12)?)?;
    /// assert_eq!(drifted.value_mismatches(), 10);
    /// assert!(drifted.max_relative_difference().expect("y differs") < 1e-11);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TableComparisonError`] when the tables belong to different
    /// sessions, do not hold the same identifiers, or hold different requests
    /// under one identifier.
    pub fn compare(&self, other: &Self) -> Result<TableComparison, TableComparisonError> {
        if self.schema() != other.schema() {
            return Err(TableComparisonError::SchemaMismatch);
        }
        let expected = by_id(self);
        let actual = by_id(other);
        let expected_ids = expected.keys().copied().collect::<BTreeSet<_>>();
        let actual_ids = actual.keys().copied().collect::<BTreeSet<_>>();
        if let Some(&id) = expected_ids.symmetric_difference(&actual_ids).next() {
            return Err(TableComparisonError::UnmatchedEvaluation { id });
        }

        let outputs = self
            .schema()
            .output_slots()
            .filter_map(|slot| Some((slot, self.schema().output(slot)?)))
            .collect::<Vec<_>>();
        let mut comparison = TableComparison {
            rows: 0,
            identical_rows: 0,
            outcome_mismatches: 0,
            differences: Vec::new(),
        };
        let mut differences = BTreeMap::<PathKey, ValueDifference>::new();
        for (id, left) in &expected {
            let right = actual[id];
            if left.request() != right.request() {
                return Err(TableComparisonError::RequestMismatch { id: *id });
            }
            for (a, b) in left.rows().iter().zip(right.rows()) {
                comparison.rows += 1;
                match (a, b) {
                    (RowOutcome::Success(a), RowOutcome::Success(b)) => {
                        let row = row_differences(&outputs, a, b);
                        if row.is_empty() {
                            comparison.identical_rows += 1;
                        }
                        for (key, (path, deviation)) in row {
                            differences
                                .entry(key)
                                .and_modify(|difference| {
                                    difference.rows += 1;
                                    difference.deviation = difference.deviation.merge(deviation);
                                })
                                .or_insert(ValueDifference {
                                    path,
                                    rows: 1,
                                    deviation,
                                });
                        }
                    }
                    _ if a == b => comparison.identical_rows += 1,
                    _ => comparison.outcome_mismatches += 1,
                }
            }
        }
        comparison.differences = differences.into_values().collect();
        Ok(comparison)
    }
}

fn by_id(table: &EvaluationTable) -> BTreeMap<EvaluationId, &EvaluatedRequest> {
    table
        .evaluations()
        .iter()
        .map(|evaluation| (evaluation.id(), evaluation))
        .collect()
}

/// Orders paths by output, then by the positions of their steps.
type PathKey = (OutputSlot, Vec<Step>);

/// The differing values of one row, by path.
type RowDifferences = BTreeMap<PathKey, (OutputPath, Deviation)>;

fn row_differences(
    outputs: &[(OutputSlot, &SessionOutput)],
    a: &OutputRow,
    b: &OutputRow,
) -> RowDifferences {
    let mut found = BTreeMap::new();
    for ((slot, output), (a, b)) in outputs.iter().zip(a.values().iter().zip(b.values())) {
        let mut trace = Trace {
            slot: *slot,
            output,
            steps: Vec::new(),
            labels: Vec::new(),
            found: &mut found,
        };
        trace.compare(output.wire(), a, b);
    }
    found
}

/// What spells one step of a path.
enum Label<'s> {
    Field(&'s FieldType),
    Variant(&'s ConstructorType),
    Entry(&'s Axis, usize),
}

impl Label<'_> {
    fn spell(&self) -> String {
        match self {
            Self::Field(field) => field.name().as_str().to_owned(),
            Self::Variant(constructor) => constructor.name().as_str().to_owned(),
            Self::Entry(axis, position) => axis
                .key(*position)
                .map_or_else(|| position.to_string(), key_label),
        }
    }
}

/// A walk through one output value of two rows, recording each leaf whose
/// bits differ.
struct Trace<'s, 'f> {
    slot: OutputSlot,
    output: &'s SessionOutput,
    steps: Vec<Step>,
    labels: Vec<Label<'s>>,
    found: &'f mut RowDifferences,
}

impl<'s> Trace<'s, '_> {
    fn compare(&mut self, wire: &'s WireType, a: &Value, b: &Value) {
        match (wire, a, b) {
            (_, Value::Quantity(x), Value::Quantity(y)) => {
                if x.get().to_bits() != y.get().to_bits() {
                    let deviation = relative_difference((x.get(), 0.0), (y.get(), 0.0));
                    self.record(wire, Deviation::Relative(deviation));
                }
            }
            (_, Value::Complex(x), Value::Complex(y)) => {
                let (x, y) = ((x.re.get(), x.im.get()), (y.re.get(), y.im.get()));
                if x.0.to_bits() != y.0.to_bits() || x.1.to_bits() != y.1.to_bits() {
                    self.record(wire, Deviation::Relative(relative_difference(x, y)));
                }
            }
            (_, Value::Datetime(x), Value::Datetime(y)) => {
                if x.get().to_bits() != y.get().to_bits() {
                    self.record(wire, Deviation::Unequal);
                }
            }
            (WireType::Indexed(indexed), Value::Indexed(xs), Value::Indexed(ys))
                if xs.len() == ys.len() =>
            {
                for (position, (x, y)) in xs.iter().zip(ys).enumerate() {
                    self.steps.push(Step::Entry(position));
                    self.labels.push(Label::Entry(indexed.axis(), position));
                    self.compare(indexed.element(), x, y);
                    self.steps.pop();
                    self.labels.pop();
                }
            }
            (
                WireType::Algebraic(algebraic),
                Value::Algebraic {
                    constructor: cx,
                    fields: fx,
                },
                Value::Algebraic {
                    constructor: cy,
                    fields: fy,
                },
            ) if cx == cy && fx.len() == fy.len() => {
                let Some(constructor) = algebraic.constructor(*cx) else {
                    self.record(wire, Deviation::Unequal);
                    return;
                };
                // A record is a one-constructor type; its paths name no variant.
                let union = algebraic.constructors().len() > 1;
                if union {
                    self.steps.push(Step::Variant(*cx));
                    self.labels.push(Label::Variant(constructor));
                }
                for (position, (field, (x, y))) in constructor
                    .fields()
                    .iter()
                    .zip(fx.iter().zip(fy))
                    .enumerate()
                {
                    self.steps.push(Step::Field(position));
                    self.labels.push(Label::Field(field));
                    self.compare(field.wire(), x, y);
                    self.steps.pop();
                    self.labels.pop();
                }
                if union {
                    self.steps.pop();
                    self.labels.pop();
                }
            }
            _ if a == b => {}
            _ => self.record(wire, Deviation::Unequal),
        }
    }

    fn record(&mut self, wire: &WireType, deviation: Deviation) {
        let key = (self.slot, self.steps.clone());
        let path = OutputPath::from_checked_steps(
            self.slot,
            self.output,
            self.steps.clone(),
            self.labels.iter().map(Label::spell).collect(),
            wire.clone(),
        );
        // Each step names a distinct position, so a row reaches a path once.
        self.found.insert(key, (path, deviation));
    }
}

/// `|a - b| / max(|a|, |b|)` of two complex numbers, zero when both are
/// zero.
///
/// Both are scaled by their largest component first, so the result is finite
/// (at most 2) even where `a - b` would overflow.
fn relative_difference(a: (f64, f64), b: (f64, f64)) -> f64 {
    let scale = [a.0, a.1, b.0, b.1]
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let (a, b) = ((a.0 / scale, a.1 / scale), (b.0 / scale, b.1 / scale));
    let magnitude = a.0.hypot(a.1).max(b.0.hypot(b.1));
    (a.0 - b.0).hypot(a.1 - b.1) / magnitude
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graphcal_model_protocol::{FiniteF64, InputColumn, RequestBatch, SessionSchema};

    use super::*;
    use crate::evaluation::{ChunkResult, EvalRequest, RowFailure, SessionMismatch};
    use crate::structured_fixture::{report_row, report_session, report_value};

    /// A request of `rows` rows with identifier `id`.
    fn request(schema: &Arc<SessionSchema>, id: u128, rows: usize) -> EvalRequest {
        let xs = (0..rows)
            .map(|row| FiniteF64::new(0.1 * f64::from(u32::try_from(row).unwrap())).unwrap())
            .collect();
        let inputs =
            RequestBatch::new(Arc::clone(schema), vec![InputColumn::Quantity(xs)]).unwrap();
        EvalRequest::new(EvaluationId::new(id), 7, inputs)
    }

    /// A table of one evaluation per entry of `evaluations`, each pairing an
    /// identifier with its row outcomes.
    fn table(evaluations: Vec<(u128, Vec<RowOutcome>)>) -> EvaluationTable {
        let schema = report_session();
        let mut table = EvaluationTable::new(Arc::clone(&schema));
        for (id, rows) in evaluations {
            let request = request(&schema, id, rows.len());
            let result = ChunkResult::new(EvaluationId::new(id), rows).unwrap();
            table.push(request, result).unwrap();
        }
        table
    }

    fn success(cost: f64, p90: Option<f64>, checks: [bool; 2], grade: &str) -> RowOutcome {
        RowOutcome::Success(report_row(report_value(cost, p90, checks, grade)))
    }

    fn baseline() -> RowOutcome {
        success(10.0, Some(2.0), [true, false], "Good")
    }

    fn failure() -> RowOutcome {
        RowOutcome::Failure(RowFailure::InvalidInput(SessionMismatch))
    }

    fn paths(comparison: &TableComparison) -> Vec<(String, usize, Deviation)> {
        comparison
            .differences()
            .iter()
            .map(|difference| {
                (
                    difference.path().to_string(),
                    difference.rows(),
                    difference.deviation(),
                )
            })
            .collect()
    }

    #[test]
    fn identical_tables_have_no_difference_in_any_order() {
        let left = table(vec![
            (1, vec![baseline(), failure()]),
            (2, vec![success(1.0, None, [false, false], "Bad")]),
        ]);
        let mut right = left.clone();
        right.sort_by_id();
        let reversed = table(vec![
            (2, vec![success(1.0, None, [false, false], "Bad")]),
            (1, vec![baseline(), failure()]),
        ]);
        for other in [&right, &reversed] {
            let comparison = left.compare(other).unwrap();
            assert_eq!(comparison.rows(), 3);
            assert_eq!(comparison.identical_rows(), 3);
            assert!(comparison.is_identical());
            assert_eq!(comparison.differences(), []);
            assert_eq!(comparison.max_relative_difference(), None);
        }
    }

    #[test]
    fn differences_are_reported_by_leaf_path_in_schema_order() {
        let left = table(vec![(
            1,
            vec![
                baseline(),
                baseline(),
                baseline(),
                success(-4.0, Some(2.0), [true, false], "Good"),
            ],
        )]);
        let right = table(vec![(
            1,
            vec![
                // Cost drifts in the last bits; the second check flips.
                success(10.0 * (1.0 + 1e-12), Some(2.0), [true, true], "Good"),
                // The latency changes variant; the grade changes.
                success(10.0, None, [true, false], "Bad"),
                baseline(),
                // A larger drift of cost, through zero's other side.
                success(4.0, Some(2.0), [true, false], "Good"),
            ],
        )]);
        let comparison = left.compare(&right).unwrap();
        assert_eq!(comparison.rows(), 4);
        assert_eq!(comparison.identical_rows(), 1);
        assert_eq!(comparison.value_mismatches(), 3);
        assert_eq!(comparison.outcome_mismatches(), 0);
        let found = paths(&comparison);
        assert_eq!(
            found
                .iter()
                .map(|(path, rows, _)| (path.as_str(), *rows))
                .collect::<Vec<_>>(),
            [
                ("report.cost", 2),
                ("report.latency", 1),
                ("report.checks.Second", 1),
                ("report.grade", 1),
            ]
        );
        for difference in comparison.differences() {
            let spelled = difference.path().to_string();
            assert_eq!(
                difference.path(),
                &OutputPath::parse(left.schema(), &spelled).unwrap(),
                "{spelled}"
            );
        }
        // |-4 - 4| / 4 is the largest possible relative difference.
        assert_eq!(found[0].2, Deviation::Relative(2.0));
        assert!(found[1..].iter().all(|(_, _, d)| *d == Deviation::Unequal));
        assert_eq!(comparison.max_relative_difference(), Some(2.0));
    }

    #[test]
    fn fields_of_a_shared_variant_are_compared_under_it() {
        let left = table(vec![(1, vec![baseline()])]);
        let right = table(vec![(
            1,
            vec![success(10.0, Some(2.5), [true, false], "Good")],
        )]);
        let comparison = left.compare(&right).unwrap();
        let found = paths(&comparison);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, "report.latency.Available.p90");
        let Deviation::Relative(relative) = found[0].2 else {
            panic!("p90 is a quantity");
        };
        assert!((relative - 0.2).abs() < 1e-15, "{relative}");
        // The path is the one a configuration would spell, and reads p90.
        let path = comparison.differences()[0].path();
        assert_eq!(
            path,
            &OutputPath::parse(right.schema(), "report.latency.Available.p90").unwrap()
        );
        let RowOutcome::Success(outputs) = &right.evaluations()[0].rows()[0] else {
            panic!("the row succeeded");
        };
        assert_eq!(
            path.get(outputs).as_deref(),
            Some(&Value::Quantity(FiniteF64::new(2.5).unwrap()))
        );
    }

    #[test]
    fn outcome_changes_are_counted_apart_from_values() {
        let left = table(vec![(1, vec![baseline(), failure(), failure()])]);
        let right = table(vec![(1, vec![failure(), baseline(), failure()])]);
        let comparison = left.compare(&right).unwrap();
        assert_eq!(comparison.identical_rows(), 1);
        assert_eq!(comparison.outcome_mismatches(), 2);
        assert_eq!(comparison.value_mismatches(), 0);
        assert_eq!(comparison.differences(), []);
    }

    #[test]
    fn signed_zeros_differ_in_bits_by_nothing() {
        let left = table(vec![(1, vec![success(0.0, None, [true, true], "Good")])]);
        let right = table(vec![(1, vec![success(-0.0, None, [true, true], "Good")])]);
        let comparison = left.compare(&right).unwrap();
        assert!(!comparison.is_identical());
        assert_eq!(comparison.max_relative_difference(), Some(0.0));
    }

    #[test]
    fn rejects_tables_of_other_evaluations() {
        let left = table(vec![(1, vec![baseline()]), (3, vec![baseline()])]);
        assert_eq!(
            left.compare(&table(vec![(1, vec![baseline()]), (2, vec![baseline()])])),
            Err(TableComparisonError::UnmatchedEvaluation {
                id: EvaluationId::new(2)
            })
        );
        assert_eq!(
            left.compare(&table(vec![(1, vec![baseline()])])),
            Err(TableComparisonError::UnmatchedEvaluation {
                id: EvaluationId::new(3)
            })
        );
        let longer = table(vec![
            (1, vec![baseline(), baseline()]),
            (3, vec![baseline()]),
        ]);
        assert_eq!(
            left.compare(&longer),
            Err(TableComparisonError::RequestMismatch {
                id: EvaluationId::new(1)
            })
        );
        let other_session = EvaluationTable::new(
            crate::ModelSchemaBuilder::new()
                .continuous("x", 0.0, 1.0)
                .unwrap()
                .boolean_output("b")
                .unwrap()
                .build()
                .unwrap(),
        );
        assert_eq!(
            left.compare(&other_session),
            Err(TableComparisonError::SchemaMismatch)
        );
    }

    #[test]
    fn relative_differences_stay_finite_and_symmetric() {
        let huge = f64::MAX;
        assert_eq!(relative_difference((huge, 0.0), (-huge, 0.0)), 2.0);
        assert_eq!(relative_difference((0.0, 0.0), (-0.0, 0.0)), 0.0);
        assert_eq!(relative_difference((0.0, 1.0), (0.0, 0.0)), 1.0);
        for (a, b) in [((3.0, 4.0), (3.0, 4.5)), ((1.0, 0.0), (1.0 + 1e-9, 0.0))] {
            assert_eq!(relative_difference(a, b), relative_difference(b, a));
        }
    }
}
