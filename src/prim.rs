use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::PrimError;
use crate::data::{Dataset, FeatureData, FeatureKind};

/// The score used to rank candidate peels and pastes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Objective {
    /// Increase in mean divided by the number of observations added or removed.
    ///
    /// This is EMA Workbench's default and is less eager to remove categorical
    /// levels than the original objective.
    #[default]
    Lenient1,
    /// Friedman and Fisher's mass-weighted improvement score.
    Lenient2,
    /// The mean outcome inside the candidate box.
    Original,
}

/// Validated controls for PRIM peeling and pasting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimConfig {
    peel_alpha: f64,
    paste_alpha: f64,
    mass_min: f64,
    objective: Objective,
}

impl PrimConfig {
    /// Constructs a configuration.
    ///
    /// Peel and paste alpha must be in `(0, 1)`. Minimum mass must be in
    /// `(0, 1]` and is measured against the entire dataset.
    pub fn new(
        peel_alpha: f64,
        paste_alpha: f64,
        mass_min: f64,
        objective: Objective,
    ) -> Result<Self, PrimError> {
        validate_alpha("peel_alpha", peel_alpha)?;
        validate_alpha("paste_alpha", paste_alpha)?;
        if !(mass_min.is_finite() && 0.0 < mass_min && mass_min <= 1.0) {
            return Err(PrimError::InvalidParameter {
                parameter: "mass_min",
                value: mass_min,
                requirement: "must be finite and in (0, 1]",
            });
        }
        Ok(Self {
            peel_alpha,
            paste_alpha,
            mass_min,
            objective,
        })
    }

    /// Returns the fraction targeted by each peel.
    #[must_use]
    pub const fn peel_alpha(&self) -> f64 {
        self.peel_alpha
    }

    /// Returns the fraction targeted by each paste.
    #[must_use]
    pub const fn paste_alpha(&self) -> f64 {
        self.paste_alpha
    }

    /// Returns the minimum permitted fraction of all observations in a box.
    #[must_use]
    pub const fn mass_min(&self) -> f64 {
        self.mass_min
    }

    /// Returns the candidate ranking objective.
    #[must_use]
    pub const fn objective(&self) -> Objective {
        self.objective
    }
}

impl Default for PrimConfig {
    fn default() -> Self {
        Self {
            peel_alpha: 0.05,
            paste_alpha: 0.05,
            mass_min: 0.05,
            objective: Objective::Lenient1,
        }
    }
}

fn validate_alpha(parameter: &'static str, value: f64) -> Result<(), PrimError> {
    if value.is_finite() && 0.0 < value && value < 1.0 {
        Ok(())
    } else {
        Err(PrimError::InvalidParameter {
            parameter,
            value,
            requirement: "must be finite and in (0, 1)",
        })
    }
}

/// Inclusive bounds for a continuous feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuousRange {
    lower: f64,
    upper: f64,
}

impl ContinuousRange {
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

/// Inclusive bounds for an integer feature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntegerRange {
    lower: i64,
    upper: i64,
}

impl IntegerRange {
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

/// The allowed values of a categorical feature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CategorySet {
    values: BTreeSet<String>,
}

impl CategorySet {
    /// Returns the allowed categories in lexical order.
    #[must_use]
    pub const fn values(&self) -> &BTreeSet<String> {
        &self.values
    }
}

/// A typed restriction on one feature.
#[derive(Clone, Debug, PartialEq)]
pub enum Restriction {
    /// Inclusive continuous bounds.
    Continuous(ContinuousRange),
    /// Inclusive integer bounds.
    Integer(IntegerRange),
    /// A set of allowed nominal values.
    Categorical(CategorySet),
}

impl Restriction {
    /// Returns the feature type represented by this restriction.
    #[must_use]
    pub const fn kind(&self) -> FeatureKind {
        match self {
            Self::Continuous(_) => FeatureKind::Continuous,
            Self::Integer(_) => FeatureKind::Integer,
            Self::Categorical(_) => FeatureKind::Categorical,
        }
    }
}

/// A named feature restriction in a PRIM box.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureLimit {
    name: String,
    restriction: Restriction,
}

impl FeatureLimit {
    /// Returns the feature name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the typed restriction.
    #[must_use]
    pub const fn restriction(&self) -> &Restriction {
        &self.restriction
    }
}

/// An axis-aligned box over all input features.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxLimits {
    limits: Vec<FeatureLimit>,
}

impl BoxLimits {
    /// Returns restrictions in dataset schema order.
    #[must_use]
    pub fn limits(&self) -> &[FeatureLimit] {
        &self.limits
    }

    /// Looks up a restriction by feature name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Restriction> {
        self.limits
            .iter()
            .find(|limit| limit.name == name)
            .map(|limit| &limit.restriction)
    }
}

/// Summary measures for one point on a peeling and pasting trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxStatistics {
    coverage: f64,
    density: f64,
    mean: f64,
    mass: f64,
    restricted_dimensions: usize,
    points: usize,
    cases_of_interest: usize,
}

impl BoxStatistics {
    /// Fraction of all cases of interest captured by the box.
    #[must_use]
    pub const fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Fraction of observations in the box that are cases of interest.
    #[must_use]
    pub const fn density(&self) -> f64 {
        self.density
    }

    /// Mean of the binary outcome inside the box; equal to density.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Fraction of all observations captured by the box.
    #[must_use]
    pub const fn mass(&self) -> f64 {
        self.mass
    }

    /// Number of features restricted relative to the initial box.
    #[must_use]
    pub const fn restricted_dimensions(&self) -> usize {
        self.restricted_dimensions
    }

    /// Number of observations captured by the box.
    #[must_use]
    pub const fn points(&self) -> usize {
        self.points
    }

    /// Number of cases of interest captured by the box.
    #[must_use]
    pub const fn cases_of_interest(&self) -> usize {
        self.cases_of_interest
    }
}

/// Whether a trajectory entry was created by initialization, peeling, or pasting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimPhase {
    /// The unrestricted starting box.
    Initial,
    /// A restriction added during peeling.
    Peel,
    /// A restriction relaxed during pasting.
    Paste,
}

/// One-sided quasi-p values for a restricted feature.
///
/// A missing side means that the corresponding bound is unrestricted. For a
/// categorical feature, `lower` contains the test result and `upper` is absent.
#[derive(Clone, Debug, PartialEq)]
pub struct QuasiPValue {
    name: String,
    lower: Option<f64>,
    upper: Option<f64>,
}

impl QuasiPValue {
    /// Returns the feature name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the lower-bound (or categorical) quasi-p value.
    #[must_use]
    pub const fn lower(&self) -> Option<f64> {
        self.lower
    }

    /// Returns the upper-bound quasi-p value.
    #[must_use]
    pub const fn upper(&self) -> Option<f64> {
        self.upper
    }
}

/// One candidate box on the complete PRIM trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxStep {
    limits: BoxLimits,
    indices: Vec<usize>,
    statistics: BoxStatistics,
    quasi_p_values: Vec<QuasiPValue>,
    phase: PrimPhase,
}

impl BoxStep {
    /// Returns this candidate's limits.
    #[must_use]
    pub const fn limits(&self) -> &BoxLimits {
        &self.limits
    }

    /// Returns original dataset row indices captured by this candidate.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns scenario-discovery summary measures.
    #[must_use]
    pub const fn statistics(&self) -> BoxStatistics {
        self.statistics
    }

    /// Returns quasi-p values for restricted features in schema order.
    #[must_use]
    pub fn quasi_p_values(&self) -> &[QuasiPValue] {
        &self.quasi_p_values
    }

    /// Returns the phase that produced this candidate.
    #[must_use]
    pub const fn phase(&self) -> PrimPhase {
        self.phase
    }
}

/// The complete peeling and pasting trajectory for one discovered box.
#[derive(Clone, Debug, PartialEq)]
pub struct PrimBox {
    trajectory: Vec<BoxStep>,
}

impl PrimBox {
    /// Returns all nested candidate boxes, including the unrestricted start.
    #[must_use]
    pub fn trajectory(&self) -> &[BoxStep] {
        &self.trajectory
    }

    /// Returns the final candidate after peeling and pasting.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "`fit_box` is the only constructor and always pushes the initial box"
    )]
    pub fn final_step(&self) -> &BoxStep {
        self.trajectory
            .last()
            .expect("a PRIM trajectory always has an initial box")
    }
}

/// Stateful conventional PRIM over a validated static dataset.
///
/// Each call to [`Prim::find_box`] finds one box and removes its final members
/// from consideration, matching EMA Workbench's default covering behavior.
/// The first returned box contains the full trajectory normally inspected for
/// the coverage/density trade-off.
pub struct Prim<'data> {
    dataset: &'data Dataset,
    config: PrimConfig,
    initial_limits: BoxLimits,
    remaining: Vec<usize>,
}

impl<'data> Prim<'data> {
    /// Creates a PRIM analysis with all rows initially available.
    #[must_use]
    pub fn new(dataset: &'data Dataset, config: PrimConfig) -> Self {
        Self {
            dataset,
            config,
            initial_limits: make_initial_limits(dataset),
            remaining: (0..dataset.row_count()).collect(),
        }
    }

    /// Returns row indices not covered by previously returned final boxes.
    #[must_use]
    pub fn remaining_rows(&self) -> &[usize] {
        &self.remaining
    }

    /// Finds the next box, or returns `None` after every row has been covered.
    pub fn find_box(&mut self) -> Option<PrimBox> {
        if self.remaining.is_empty() {
            return None;
        }

        let found = self.fit_box();
        let mut removed = vec![false; self.dataset.row_count()];
        found
            .final_step()
            .indices()
            .iter()
            .for_each(|index| removed[*index] = true);
        self.remaining.retain(|index| !removed[*index]);
        Some(found)
    }

    /// Peels then pastes from the unrestricted box, recording every accepted step.
    ///
    /// The candidate under consideration is held in `current` rather than read
    /// back from `trajectory`, so the trajectory is provably non-empty without
    /// any fallible lookup.
    fn fit_box(&self) -> PrimBox {
        let mut trajectory = Vec::new();
        let mut current = self.make_step(
            self.initial_limits.clone(),
            self.remaining.clone(),
            PrimPhase::Initial,
        );

        while let Some(candidate) = self.best_peel(&current) {
            let mass_old = self.mass_of(&current.indices);
            let mass_new = self.mass_of(&candidate.indices);

            if mass_new >= self.config.mass_min && mass_new < mass_old && candidate.score > 0.0 {
                let next = self.make_step(candidate.limits, candidate.indices, PrimPhase::Peel);
                trajectory.push(current);
                current = next;
            } else {
                break;
            }
        }

        while let Some(candidate) = self.best_paste(&current) {
            let mass_old = self.mass_of(&current.indices);
            let mass_new = self.mass_of(&candidate.indices);
            let mean_old = mean(self.dataset, &current.indices);
            let mean_new = mean(self.dataset, &candidate.indices);

            if mass_new >= self.config.mass_min
                && mass_new > mass_old
                && candidate.score > 0.0
                && mean_new > mean_old
            {
                let next = self.make_step(candidate.limits, candidate.indices, PrimPhase::Paste);
                trajectory.push(current);
                current = next;
            } else {
                break;
            }
        }

        trajectory.push(current);
        PrimBox { trajectory }
    }

    /// Returns the fraction of all dataset rows covered by `indices`.
    fn mass_of(&self, indices: &[usize]) -> f64 {
        count_as_f64(indices.len()) / count_as_f64(self.dataset.row_count())
    }

    fn best_peel(&self, current: &BoxStep) -> Option<Candidate> {
        let mut candidates = Vec::new();
        for kind in [
            FeatureKind::Continuous,
            FeatureKind::Integer,
            FeatureKind::Categorical,
        ] {
            for feature_index in self.feature_indices(kind) {
                match &self.dataset.features()[feature_index].data {
                    FeatureData::Continuous(values) => {
                        self.continuous_peels(current, feature_index, values, &mut candidates);
                    }
                    FeatureData::Integer(values) => {
                        self.integer_peels(current, feature_index, values, &mut candidates);
                    }
                    FeatureData::Categorical(values) => {
                        self.categorical_peels(current, feature_index, values, &mut candidates);
                    }
                }
            }
        }
        pick_best(candidates)
    }

    fn continuous_peels(
        &self,
        current: &BoxStep,
        feature_index: usize,
        values: &[f64],
        candidates: &mut Vec<Candidate>,
    ) {
        let in_box = current
            .indices
            .iter()
            .map(|index| values[*index])
            .collect::<Vec<_>>();

        // EMA Workbench evaluates the upper peel before the lower peel.
        let upper = quantile(&in_box, 1.0 - self.config.peel_alpha);
        let upper_indices = current
            .indices
            .iter()
            .copied()
            .filter(|index| values[*index] <= upper)
            .collect();
        let mut upper_limits = current.limits.clone();
        continuous_range_mut(&mut upper_limits, feature_index).upper = upper;
        candidates.push(self.candidate(current, upper_limits, upper_indices));

        let lower = quantile(&in_box, self.config.peel_alpha);
        let lower_indices = current
            .indices
            .iter()
            .copied()
            .filter(|index| values[*index] >= lower)
            .collect();
        let mut lower_limits = current.limits.clone();
        continuous_range_mut(&mut lower_limits, feature_index).lower = lower;
        candidates.push(self.candidate(current, lower_limits, lower_indices));
    }

    fn integer_peels(
        &self,
        current: &BoxStep,
        feature_index: usize,
        values: &[i64],
        candidates: &mut Vec<Candidate>,
    ) {
        let in_box = current
            .indices
            .iter()
            .map(|index| values[*index])
            .collect::<Vec<_>>();
        let bounds = *integer_range(&current.limits, feature_index);

        let upper_quantile = integer_quantile(&in_box, 1.0 - self.config.peel_alpha);
        let upper_indices = current
            .indices
            .iter()
            .copied()
            .filter(|index| {
                let value = values[*index];
                if upper_quantile == bounds.upper {
                    value < bounds.upper && value >= bounds.lower
                } else {
                    value <= upper_quantile && value >= bounds.lower
                }
            })
            .collect::<Vec<_>>();
        let upper = upper_indices
            .iter()
            .map(|index| values[*index])
            .max()
            .or_else(|| in_box.iter().copied().max())
            .unwrap_or(bounds.upper);
        let mut upper_limits = current.limits.clone();
        integer_range_mut(&mut upper_limits, feature_index).upper = upper;
        candidates.push(self.candidate(current, upper_limits, upper_indices));

        let lower_quantile = integer_quantile(&in_box, self.config.peel_alpha);
        let lower_indices = current
            .indices
            .iter()
            .copied()
            .filter(|index| {
                let value = values[*index];
                if lower_quantile == bounds.lower {
                    value > bounds.lower && value <= bounds.upper
                } else {
                    value >= lower_quantile && value <= bounds.upper
                }
            })
            .collect::<Vec<_>>();
        let lower = lower_indices
            .iter()
            .map(|index| values[*index])
            .min()
            .or_else(|| in_box.iter().copied().min())
            .unwrap_or(bounds.lower);
        let mut lower_limits = current.limits.clone();
        integer_range_mut(&mut lower_limits, feature_index).lower = lower;
        candidates.push(self.candidate(current, lower_limits, lower_indices));
    }

    fn categorical_peels(
        &self,
        current: &BoxStep,
        feature_index: usize,
        values: &[String],
        candidates: &mut Vec<Candidate>,
    ) {
        let categories = category_set(&current.limits, feature_index)
            .values
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        if categories.len() <= 1 {
            return;
        }

        candidates.extend(categories.into_iter().map(|removed| {
            let indices = current
                .indices
                .iter()
                .copied()
                .filter(|index| values[*index] != removed)
                .collect();
            let mut limits = current.limits.clone();
            category_set_mut(&mut limits, feature_index)
                .values
                .remove(&removed);
            self.candidate(current, limits, indices)
        }));
    }

    fn best_paste(&self, current: &BoxStep) -> Option<Candidate> {
        let mut candidates = Vec::new();
        for kind in [
            FeatureKind::Continuous,
            FeatureKind::Integer,
            FeatureKind::Categorical,
        ] {
            for feature_index in self.feature_indices(kind).filter(|index| {
                current.limits.limits[*index].restriction
                    != self.initial_limits.limits[*index].restriction
            }) {
                match &self.dataset.features()[feature_index].data {
                    FeatureData::Continuous(values) => {
                        self.continuous_pastes(current, feature_index, values, &mut candidates);
                    }
                    FeatureData::Integer(values) => {
                        self.integer_pastes(current, feature_index, values, &mut candidates);
                    }
                    FeatureData::Categorical(values) => {
                        self.categorical_pastes(current, feature_index, values, &mut candidates);
                    }
                }
            }
        }
        pick_best(candidates)
    }

    fn continuous_pastes(
        &self,
        current: &BoxStep,
        feature_index: usize,
        values: &[f64],
        candidates: &mut Vec<Candidate>,
    ) {
        let current_range = *continuous_range(&current.limits, feature_index);
        let initial_range = *continuous_range(&self.initial_limits, feature_index);

        let mut lower_region = current.limits.clone();
        let region = continuous_range_mut(&mut lower_region, feature_index);
        region.lower = initial_range.lower;
        region.upper = current_range.lower;
        let lower_data = rows_in_box(self.dataset, &self.remaining, &lower_region)
            .into_iter()
            .map(|index| values[index])
            .collect::<Vec<_>>();
        let lower = if lower_data.is_empty() {
            initial_range.lower
        } else {
            quantile(&lower_data, 1.0 - self.config.paste_alpha)
        };
        let mut lower_limits = current.limits.clone();
        continuous_range_mut(&mut lower_limits, feature_index).lower = lower;
        let lower_indices = rows_in_box(self.dataset, &self.remaining, &lower_limits);
        candidates.push(self.candidate(current, lower_limits, lower_indices));

        let mut upper_region = current.limits.clone();
        let region = continuous_range_mut(&mut upper_region, feature_index);
        region.lower = current_range.upper;
        region.upper = initial_range.upper;
        let upper_data = rows_in_box(self.dataset, &self.remaining, &upper_region)
            .into_iter()
            .map(|index| values[index])
            .collect::<Vec<_>>();
        let upper = if upper_data.is_empty() {
            initial_range.upper
        } else {
            quantile(&upper_data, self.config.paste_alpha)
        };
        let mut upper_limits = current.limits.clone();
        continuous_range_mut(&mut upper_limits, feature_index).upper = upper;
        let upper_indices = rows_in_box(self.dataset, &self.remaining, &upper_limits);
        candidates.push(self.candidate(current, upper_limits, upper_indices));
    }

    fn integer_pastes(
        &self,
        current: &BoxStep,
        feature_index: usize,
        values: &[i64],
        candidates: &mut Vec<Candidate>,
    ) {
        let current_range = *integer_range(&current.limits, feature_index);
        let initial_range = *integer_range(&self.initial_limits, feature_index);

        let mut lower_region = current.limits.clone();
        let region = integer_range_mut(&mut lower_region, feature_index);
        region.lower = initial_range.lower;
        region.upper = current_range.lower;
        let lower_data = rows_in_box(self.dataset, &self.remaining, &lower_region)
            .into_iter()
            .map(|index| values[index])
            .collect::<Vec<_>>();
        let lower = if lower_data.is_empty() {
            initial_range.lower
        } else {
            integer_quantile(&lower_data, 1.0 - self.config.paste_alpha)
        };
        let mut lower_limits = current.limits.clone();
        integer_range_mut(&mut lower_limits, feature_index).lower = lower;
        let lower_indices = rows_in_box(self.dataset, &self.remaining, &lower_limits);
        candidates.push(self.candidate(current, lower_limits, lower_indices));

        let mut upper_region = current.limits.clone();
        let region = integer_range_mut(&mut upper_region, feature_index);
        region.lower = current_range.upper;
        region.upper = initial_range.upper;
        let upper_data = rows_in_box(self.dataset, &self.remaining, &upper_region)
            .into_iter()
            .map(|index| values[index])
            .collect::<Vec<_>>();
        let upper = if upper_data.is_empty() {
            initial_range.upper
        } else {
            integer_quantile(&upper_data, self.config.paste_alpha)
        };
        let mut upper_limits = current.limits.clone();
        integer_range_mut(&mut upper_limits, feature_index).upper = upper;
        let upper_indices = rows_in_box(self.dataset, &self.remaining, &upper_limits);
        candidates.push(self.candidate(current, upper_limits, upper_indices));
    }

    fn categorical_pastes(
        &self,
        current: &BoxStep,
        feature_index: usize,
        _values: &[String],
        candidates: &mut Vec<Candidate>,
    ) {
        let current_categories = &category_set(&current.limits, feature_index).values;
        let missing = category_set(&self.initial_limits, feature_index)
            .values
            .difference(current_categories)
            .cloned();

        candidates.extend(missing.map(|added| {
            let mut limits = current.limits.clone();
            category_set_mut(&mut limits, feature_index)
                .values
                .insert(added);
            let indices = rows_in_box(self.dataset, &self.remaining, &limits);
            self.candidate(current, limits, indices)
        }));
    }

    fn candidate(&self, current: &BoxStep, limits: BoxLimits, indices: Vec<usize>) -> Candidate {
        let score = objective_score(
            self.dataset,
            &current.indices,
            &indices,
            self.config.objective,
        );
        let non_restricted =
            self.dataset.features().len() - restricted_count(&limits, &self.initial_limits);
        Candidate {
            limits,
            indices,
            score,
            non_restricted,
        }
    }

    fn feature_indices(&self, kind: FeatureKind) -> impl Iterator<Item = usize> + '_ {
        self.dataset
            .features()
            .iter()
            .enumerate()
            .filter(move |(_, feature)| feature.kind() == kind)
            .map(|(index, _)| index)
    }

    fn make_step(&self, limits: BoxLimits, indices: Vec<usize>, phase: PrimPhase) -> BoxStep {
        let cases = count_cases(self.dataset, &indices);
        let points = indices.len();
        let total_cases = self.dataset.case_count().max(1);
        let statistics = BoxStatistics {
            coverage: count_as_f64(cases) / count_as_f64(total_cases),
            density: count_as_f64(cases) / count_as_f64(points),
            mean: count_as_f64(cases) / count_as_f64(points),
            mass: self.mass_of(&indices),
            restricted_dimensions: restricted_count(&limits, &self.initial_limits),
            points,
            cases_of_interest: cases,
        };
        let quasi_p_values = self.calculate_quasi_p_values(&limits, &statistics);
        BoxStep {
            limits,
            indices,
            statistics,
            quasi_p_values,
            phase,
        }
    }

    #[expect(
        clippy::float_cmp,
        reason = "bounds are compared for identity with the initial box, not for numeric closeness"
    )]
    fn calculate_quasi_p_values(
        &self,
        limits: &BoxLimits,
        statistics: &BoxStatistics,
    ) -> Vec<QuasiPValue> {
        limits
            .limits
            .iter()
            .enumerate()
            .filter(|(index, limit)| {
                limit.restriction != self.initial_limits.limits[*index].restriction
            })
            .map(|(index, limit)| {
                let (lower, upper) = match &limit.restriction {
                    Restriction::Continuous(current) => {
                        let initial = continuous_range(&self.initial_limits, index);
                        (
                            (current.lower != initial.lower).then(|| {
                                self.quasi_p_with_relaxed_bound(
                                    limits,
                                    index,
                                    Side::Lower,
                                    statistics,
                                )
                            }),
                            (current.upper != initial.upper).then(|| {
                                self.quasi_p_with_relaxed_bound(
                                    limits,
                                    index,
                                    Side::Upper,
                                    statistics,
                                )
                            }),
                        )
                    }
                    Restriction::Integer(current) => {
                        let initial = integer_range(&self.initial_limits, index);
                        (
                            (current.lower != initial.lower).then(|| {
                                self.quasi_p_with_relaxed_bound(
                                    limits,
                                    index,
                                    Side::Lower,
                                    statistics,
                                )
                            }),
                            (current.upper != initial.upper).then(|| {
                                self.quasi_p_with_relaxed_bound(
                                    limits,
                                    index,
                                    Side::Upper,
                                    statistics,
                                )
                            }),
                        )
                    }
                    Restriction::Categorical(_) => (
                        Some(self.quasi_p_with_relaxed_bound(
                            limits,
                            index,
                            Side::Categorical,
                            statistics,
                        )),
                        None,
                    ),
                };
                QuasiPValue {
                    name: limit.name.clone(),
                    lower,
                    upper,
                }
            })
            .collect()
    }

    fn quasi_p_with_relaxed_bound(
        &self,
        limits: &BoxLimits,
        feature_index: usize,
        side: Side,
        statistics: &BoxStatistics,
    ) -> f64 {
        let kind = self.dataset.features()[feature_index].kind();
        let mut relaxed = limits.clone();
        match side {
            Side::Lower => match (
                &mut relaxed.limits[feature_index].restriction,
                &self.initial_limits.limits[feature_index].restriction,
            ) {
                (Restriction::Continuous(current), Restriction::Continuous(initial)) => {
                    current.lower = initial.lower;
                }
                (Restriction::Integer(current), Restriction::Integer(initial)) => {
                    current.lower = initial.lower;
                }
                _ => schema_invariant_violated(kind),
            },
            Side::Upper => match (
                &mut relaxed.limits[feature_index].restriction,
                &self.initial_limits.limits[feature_index].restriction,
            ) {
                (Restriction::Continuous(current), Restriction::Continuous(initial)) => {
                    current.upper = initial.upper;
                }
                (Restriction::Integer(current), Restriction::Integer(initial)) => {
                    current.upper = initial.upper;
                }
                _ => schema_invariant_violated(kind),
            },
            Side::Categorical => {
                relaxed.limits[feature_index].restriction = self.initial_limits.limits
                    [feature_index]
                    .restriction
                    .clone();
            }
        }

        let comparison = rows_in_box(self.dataset, &self.remaining, &relaxed);
        let comparison_cases = count_cases(self.dataset, &comparison);
        let probability = count_as_f64(comparison_cases) / count_as_f64(comparison.len());
        binomial_greater(statistics.cases_of_interest, statistics.points, probability)
    }
}

#[derive(Clone, Copy)]
enum Side {
    Lower,
    Upper,
    Categorical,
}

struct Candidate {
    limits: BoxLimits,
    indices: Vec<usize>,
    score: f64,
    non_restricted: usize,
}

fn pick_best(candidates: Vec<Candidate>) -> Option<Candidate> {
    candidates.into_iter().reduce(|best, candidate| {
        let ordering = candidate
            .score
            .total_cmp(&best.score)
            .then(candidate.non_restricted.cmp(&best.non_restricted));
        if ordering == Ordering::Greater {
            candidate
        } else {
            best
        }
    })
}

/// Builds the unrestricted box spanning every observed value of every feature.
#[expect(
    clippy::expect_used,
    reason = "`Feature` constructors reject empty columns, so every feature has a minimum and maximum"
)]
fn make_initial_limits(dataset: &Dataset) -> BoxLimits {
    let limits = dataset
        .features()
        .iter()
        .map(|feature| {
            let restriction = match &feature.data {
                FeatureData::Continuous(values) => Restriction::Continuous(ContinuousRange {
                    lower: values
                        .iter()
                        .copied()
                        .reduce(f64::min)
                        .expect("non-empty feature"),
                    upper: values
                        .iter()
                        .copied()
                        .reduce(f64::max)
                        .expect("non-empty feature"),
                }),
                FeatureData::Integer(values) => Restriction::Integer(IntegerRange {
                    lower: *values.iter().min().expect("non-empty feature"),
                    upper: *values.iter().max().expect("non-empty feature"),
                }),
                FeatureData::Categorical(values) => Restriction::Categorical(CategorySet {
                    values: values.iter().cloned().collect(),
                }),
            };
            FeatureLimit {
                name: feature.name().to_owned(),
                restriction,
            }
        })
        .collect();
    BoxLimits { limits }
}

fn rows_in_box(dataset: &Dataset, population: &[usize], limits: &BoxLimits) -> Vec<usize> {
    population
        .iter()
        .copied()
        .filter(|row| {
            dataset
                .features()
                .iter()
                .zip(&limits.limits)
                .all(
                    |(feature, limit)| match (&feature.data, &limit.restriction) {
                        (FeatureData::Continuous(values), Restriction::Continuous(range)) => {
                            range.lower <= values[*row] && values[*row] <= range.upper
                        }
                        (FeatureData::Integer(values), Restriction::Integer(range)) => {
                            range.lower <= values[*row] && values[*row] <= range.upper
                        }
                        (
                            FeatureData::Categorical(values),
                            Restriction::Categorical(categories),
                        ) => categories.values.contains(&values[*row]),
                        _ => schema_invariant_violated(feature.kind()),
                    },
                )
        })
        .collect()
}

fn restricted_count(limits: &BoxLimits, initial: &BoxLimits) -> usize {
    limits
        .limits
        .iter()
        .zip(&initial.limits)
        .filter(|(current, original)| current.restriction != original.restriction)
        .count()
}

/// Reports a box whose restrictions disagree with the dataset schema.
///
/// [`make_initial_limits`] derives one restriction per feature from that
/// feature's own data, and peeling and pasting only ever narrow or widen a
/// restriction in place. A limit can therefore never change variant, so the
/// mismatched arms this backs are dead code rather than unhandled input.
#[expect(
    clippy::unreachable,
    reason = "box limits mirror the dataset schema by construction; see the invariant above"
)]
fn schema_invariant_violated(kind: FeatureKind) -> ! {
    unreachable!("a {kind:?} feature must carry a {kind:?} restriction");
}

/// Converts a row or case count to `f64` for scenario-discovery statistics.
///
/// Datasets are indexed by `usize`, but a count large enough to lose precision
/// here would need more than 2^53 rows in memory.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts stay far below 2^53, where the conversion is exact"
)]
const fn count_as_f64(count: usize) -> f64 {
    count as f64
}

fn continuous_range(limits: &BoxLimits, index: usize) -> &ContinuousRange {
    match &limits.limits[index].restriction {
        Restriction::Continuous(range) => range,
        _ => schema_invariant_violated(FeatureKind::Continuous),
    }
}

fn continuous_range_mut(limits: &mut BoxLimits, index: usize) -> &mut ContinuousRange {
    match &mut limits.limits[index].restriction {
        Restriction::Continuous(range) => range,
        _ => schema_invariant_violated(FeatureKind::Continuous),
    }
}

fn integer_range(limits: &BoxLimits, index: usize) -> &IntegerRange {
    match &limits.limits[index].restriction {
        Restriction::Integer(range) => range,
        _ => schema_invariant_violated(FeatureKind::Integer),
    }
}

fn integer_range_mut(limits: &mut BoxLimits, index: usize) -> &mut IntegerRange {
    match &mut limits.limits[index].restriction {
        Restriction::Integer(range) => range,
        _ => schema_invariant_violated(FeatureKind::Integer),
    }
}

fn category_set(limits: &BoxLimits, index: usize) -> &CategorySet {
    match &limits.limits[index].restriction {
        Restriction::Categorical(categories) => categories,
        _ => schema_invariant_violated(FeatureKind::Categorical),
    }
}

fn category_set_mut(limits: &mut BoxLimits, index: usize) -> &mut CategorySet {
    match &mut limits.limits[index].restriction {
        Restriction::Categorical(categories) => categories,
        _ => schema_invariant_violated(FeatureKind::Categorical),
    }
}

#[expect(
    clippy::float_cmp,
    reason = "EMA Workbench scores a candidate as neutral only when the mean is bit-identical"
)]
fn objective_score(
    dataset: &Dataset,
    old_indices: &[usize],
    new_indices: &[usize],
    objective: Objective,
) -> f64 {
    let old_mean = mean(dataset, old_indices);
    let new_mean = mean(dataset, new_indices);
    let delta = new_mean - old_mean;
    let changed = old_indices.len().abs_diff(new_indices.len());

    match objective {
        Objective::Lenient1 if old_mean != new_mean && changed > 0 => delta / count_as_f64(changed),
        Objective::Lenient2 if old_mean != new_mean && changed > 0 => {
            count_as_f64(new_indices.len()) * delta / count_as_f64(changed)
        }
        Objective::Original if new_indices.is_empty() => -1.0,
        Objective::Original => new_mean,
        Objective::Lenient1 | Objective::Lenient2 => 0.0,
    }
}

fn count_cases(dataset: &Dataset, indices: &[usize]) -> usize {
    indices
        .iter()
        .filter(|index| dataset.cases_of_interest()[**index])
        .count()
}

fn mean(dataset: &Dataset, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        0.0
    } else {
        count_as_f64(count_cases(dataset, indices)) / count_as_f64(indices.len())
    }
}

/// EMA Workbench's tie-aware empirical quantile rather than a library
/// interpolator. Moving across equal values prevents a no-op peel.
#[expect(
    clippy::float_cmp,
    reason = "equal neighbours are exact ties in the sorted sample, not approximate matches"
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "`position` lies in [0, len - 1], so both bounds are in-range indices"
)]
fn quantile(values: &[f64], probability: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = count_as_f64(sorted.len() - 1) * probability;
    let mut lower = position.floor() as usize;
    let mut upper = position.ceil() as usize;

    if probability > 0.5 {
        while sorted[lower] == sorted[upper] && lower > 0 {
            lower -= 1;
        }
    } else {
        while sorted[lower] == sorted[upper] && upper < sorted.len() - 1 {
            upper += 1;
        }
    }
    f64::midpoint(sorted[lower], sorted[upper])
}

#[expect(
    clippy::cast_precision_loss,
    reason = "integer features hold sample values, not counts near i64's extremes"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "truncation back to i64 is the intended floor-toward-zero of the quantile"
)]
fn integer_quantile(values: &[i64], probability: f64) -> i64 {
    let as_float = values.iter().map(|value| *value as f64).collect::<Vec<_>>();
    quantile(&as_float, probability).trunc() as i64
}

/// Exact one-sided binomial tail P(X >= observed), evaluated with log-sum-exp.
fn binomial_greater(observed: usize, trials: usize, probability: f64) -> f64 {
    if observed == 0 {
        return 1.0;
    }
    if probability <= 0.0 {
        return 0.0;
    }
    if probability >= 1.0 {
        return 1.0;
    }

    let log_p = probability.ln();
    let log_q = (-probability).ln_1p();
    let successes = count_as_f64(observed);
    let failures = count_as_f64(trials - observed);
    let first = successes.mul_add(
        log_p,
        failures.mul_add(log_q, log_binomial_coefficient(trials, observed)),
    );
    let log_terms = std::iter::successors(Some((observed, first)), |(successes, term)| {
        (*successes < trials).then(|| {
            let next = *term + count_as_f64(trials - successes).ln()
                - count_as_f64(successes + 1).ln()
                + log_p
                - log_q;
            (successes + 1, next)
        })
    })
    .map(|(_, term)| term)
    .collect::<Vec<_>>();
    let maximum = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (maximum.exp()
        * log_terms
            .iter()
            .map(|term| (*term - maximum).exp())
            .sum::<f64>())
    .min(1.0)
}

fn log_binomial_coefficient(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (1..=k)
        .map(|i| count_as_f64(n - k + i).ln() - count_as_f64(i).ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Feature;

    fn simple_dataset() -> Dataset {
        Dataset::new(
            vec![Feature::continuous("x", (0..100).map(f64::from).collect()).unwrap()],
            (0..100).map(|value| value >= 70).collect(),
        )
        .unwrap()
    }

    #[test]
    fn configuration_rejects_out_of_range_values() {
        assert!(PrimConfig::new(0.0, 0.1, 0.1, Objective::Lenient1).is_err());
        assert!(PrimConfig::new(0.1, 1.0, 0.1, Objective::Lenient1).is_err());
        assert!(PrimConfig::new(0.1, 0.1, 0.0, Objective::Lenient1).is_err());
        assert!(PrimConfig::new(0.8, 0.8, 1.0, Objective::Lenient1).is_ok());
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "these quantiles are exactly representable, so the reference values must match bit for bit"
    )]
    fn quantile_matches_ema_workbench_examples() {
        let values = (0..10).map(f64::from).collect::<Vec<_>>();
        assert_eq!(quantile(&values, 0.9), 8.5);
        assert_eq!(quantile(&values, 0.95), 8.5);
        assert_eq!(quantile(&values, 0.1), 0.5);
        assert_eq!(quantile(&values, 0.05), 0.5);

        let repeated = vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0];
        assert_eq!(quantile(&repeated, 0.9), 8.5);
        assert_eq!(quantile(&repeated, 0.1), 1.5);
    }

    #[test]
    fn trajectory_preserves_core_invariants() {
        let dataset = simple_dataset();
        let mut prim = Prim::new(
            &dataset,
            PrimConfig::new(0.1, 0.1, 0.1, Objective::Lenient1).unwrap(),
        );
        let found = prim.find_box().unwrap();

        for pair in found.trajectory().windows(2) {
            if pair[1].phase() == PrimPhase::Peel {
                assert!(pair[1].statistics().mass() < pair[0].statistics().mass());
                assert!(pair[1].statistics().mean() > pair[0].statistics().mean());
            }
            assert!(pair[1].statistics().mass() >= 0.1);
            assert!((0.0..=1.0).contains(&pair[1].statistics().coverage()));
            assert!((0.0..=1.0).contains(&pair[1].statistics().density()));
        }
        assert!(found.final_step().statistics().density() >= 0.99);
    }

    #[test]
    fn successive_boxes_partition_rows() {
        let dataset = simple_dataset();
        let mut prim = Prim::new(&dataset, PrimConfig::default());
        let mut seen = BTreeSet::new();
        while let Some(found) = prim.find_box() {
            for index in found.final_step().indices() {
                assert!(seen.insert(*index));
            }
        }
        assert_eq!(seen.len(), dataset.row_count());
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the degenerate tails return exactly 0.0 or 1.0 by early return"
    )]
    fn binomial_tail_handles_boundaries() {
        assert_eq!(binomial_greater(0, 5, 0.2), 1.0);
        assert_eq!(binomial_greater(2, 5, 0.0), 0.0);
        assert_eq!(binomial_greater(5, 5, 1.0), 1.0);
        assert!((binomial_greater(3, 5, 0.5) - 0.5).abs() < 1e-14);
    }
}
