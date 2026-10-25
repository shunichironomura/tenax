//! Choosing one candidate box from a PRIM trajectory by an explicit
//! coverage-density criterion.

use thiserror::Error;

use crate::prim::{BoxStatistics, PrimBox};

/// Errors raised while configuring candidate selection.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum CandidateSelectionError {
    /// A minimum density must be a finite fraction in `(0, 1]`.
    #[error("minimum density must be finite and in (0, 1], got {value}")]
    InvalidMinimumDensity {
        /// The rejected value.
        value: f64,
    },
}

/// What a selected candidate box maximizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionObjective {
    /// The harmonic mean of coverage and density, `2cd / (c + d)`, which is
    /// zero when both are zero: the box that best describes where the cases
    /// of interest are.
    Balanced,
    /// The coverage among candidates whose density is at least the given
    /// minimum: the largest box in which the cases of interest are reliably
    /// found.
    HighDensity(MinimumDensity),
}

/// A density threshold in `(0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MinimumDensity(f64);

impl MinimumDensity {
    /// Validates a density threshold.
    ///
    /// # Errors
    ///
    /// Returns [`CandidateSelectionError::InvalidMinimumDensity`] unless
    /// `value` is finite and in `(0, 1]`.
    pub fn new(value: f64) -> Result<Self, CandidateSelectionError> {
        if value.is_finite() && value > 0.0 && value <= 1.0 {
            Ok(Self(value))
        } else {
            Err(CandidateSelectionError::InvalidMinimumDensity { value })
        }
    }

    /// Returns the threshold.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// A rule for choosing one candidate of a PRIM trajectory.
///
/// Peeling trades coverage for density, so no single trajectory step is best
/// for every purpose. A selection considers the candidates that restrict at
/// most `max_restricted_dimensions` features, so that the chosen box stays
/// interpretable, and picks the one that maximizes its [`SelectionObjective`].
///
/// The unrestricted initial box is a candidate too: when a property already
/// holds almost everywhere, "everywhere" is its best description. Use
/// [`CandidateSelection::restricting_at_least_one`] to choose only among boxes
/// that restrict some feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateSelection {
    max_restricted_dimensions: usize,
    requires_restriction: bool,
    objective: SelectionObjective,
}

impl CandidateSelection {
    /// Constructs a selection rule over the candidates that restrict at most
    /// `max_restricted_dimensions` features, the unrestricted box included.
    #[must_use]
    pub const fn new(max_restricted_dimensions: usize, objective: SelectionObjective) -> Self {
        Self {
            max_restricted_dimensions,
            requires_restriction: false,
            objective,
        }
    }

    /// Excludes the unrestricted box: the selected candidate restricts at
    /// least one feature.
    #[must_use]
    pub const fn restricting_at_least_one(mut self) -> Self {
        self.requires_restriction = true;
        self
    }

    /// Returns the largest number of restricted features a selected
    /// candidate may have.
    #[must_use]
    pub const fn max_restricted_dimensions(&self) -> usize {
        self.max_restricted_dimensions
    }

    /// Returns whether the selected candidate must restrict a feature.
    #[must_use]
    pub const fn requires_restriction(&self) -> bool {
        self.requires_restriction
    }

    /// Returns what the selected candidate maximizes.
    #[must_use]
    pub const fn objective(&self) -> SelectionObjective {
        self.objective
    }

    fn eligible(&self, statistics: BoxStatistics) -> bool {
        let restricted = statistics.restricted_dimensions();
        let minimum = usize::from(self.requires_restriction);
        let interpretable = (minimum..=self.max_restricted_dimensions).contains(&restricted);
        interpretable
            && match self.objective {
                SelectionObjective::Balanced => true,
                SelectionObjective::HighDensity(minimum) => statistics.density() >= minimum.get(),
            }
    }

    fn score(&self, statistics: BoxStatistics) -> f64 {
        let (coverage, density) = (statistics.coverage(), statistics.density());
        match self.objective {
            SelectionObjective::Balanced if coverage + density > 0.0 => {
                2.0 * coverage * density / (coverage + density)
            }
            SelectionObjective::Balanced => 0.0,
            SelectionObjective::HighDensity(_) => coverage,
        }
    }
}

impl PrimBox {
    /// Returns the trajectory position of the candidate `selection` chooses,
    /// or `None` when no candidate is eligible.
    ///
    /// Among equal scores the earliest trajectory position wins.
    ///
    /// ```
    /// use tenax::{
    ///     CandidateSelection, Dataset, Feature, MinimumDensity, Prim, PrimConfig,
    ///     SelectionObjective,
    /// };
    ///
    /// let load = (0..100).map(|i| f64::from(i) / 100.0).collect::<Vec<_>>();
    /// let cases = load.iter().map(|load| *load >= 0.7).collect();
    /// let dataset = Dataset::new(vec![Feature::continuous("load", load)?], cases)?;
    /// let found = Prim::new(&dataset, PrimConfig::default())
    ///     .find_box()
    ///     .expect("the dataset has rows");
    ///
    /// let selection = CandidateSelection::new(
    ///     3,
    ///     SelectionObjective::HighDensity(MinimumDensity::new(0.8)?),
    /// );
    /// let position = found
    ///     .select_candidate(selection)
    ///     .expect("a dense candidate exists");
    /// assert!(found.trajectory()[position].statistics().density() >= 0.8);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn select_candidate(&self, selection: CandidateSelection) -> Option<usize> {
        self.trajectory()
            .iter()
            .enumerate()
            .filter(|(_, step)| selection.eligible(step.statistics()))
            .fold(None::<(usize, f64)>, |best, (position, step)| {
                let score = selection.score(step.statistics());
                match best {
                    Some((_, best_score)) if best_score >= score => best,
                    _ => Some((position, score)),
                }
            })
            .map(|(position, _)| position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dataset, Feature, Prim, PrimConfig};

    /// Cases where `x >= 0.6` and `c == "b"`; `n` is irrelevant.
    fn dataset() -> Dataset {
        let x = (0..400)
            .map(|i| f64::from(i % 100) / 100.0)
            .collect::<Vec<_>>();
        let c = (0..400)
            .map(|i| if i / 100 % 2 == 0 { "a" } else { "b" }.to_owned())
            .collect();
        let n = (0..400).map(|i| i64::from(i % 7)).collect();
        let cases = (0..400)
            .map(|i| i % 100 >= 60 && i / 100 % 2 == 1)
            .collect();
        Dataset::new(
            vec![
                Feature::continuous("x", x).unwrap(),
                Feature::categorical("c", c).unwrap(),
                Feature::integer("n", n).unwrap(),
            ],
            cases,
        )
        .unwrap()
    }

    fn found() -> PrimBox {
        Prim::new(&dataset(), PrimConfig::default())
            .find_box()
            .unwrap()
    }

    fn selection(max: usize, objective: SelectionObjective) -> CandidateSelection {
        CandidateSelection::new(max, objective).restricting_at_least_one()
    }

    fn high_density(minimum: f64) -> SelectionObjective {
        SelectionObjective::HighDensity(MinimumDensity::new(minimum).unwrap())
    }

    #[test]
    fn minimum_density_must_be_a_positive_fraction() {
        for value in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(MinimumDensity::new(value).is_err(), "{value}");
        }
        assert_eq!(MinimumDensity::new(1.0).unwrap().get(), 1.0);
    }

    #[test]
    fn balanced_maximizes_the_harmonic_mean_of_eligible_candidates() {
        let found = found();
        let chosen = selection(6, SelectionObjective::Balanced);
        let position = found.select_candidate(chosen).unwrap();
        let harmonic = |statistics: BoxStatistics| {
            let (c, d) = (statistics.coverage(), statistics.density());
            2.0 * c * d / (c + d)
        };
        let best = harmonic(found.trajectory()[position].statistics());
        for (other, step) in found.trajectory().iter().enumerate() {
            let statistics = step.statistics();
            if statistics.restricted_dimensions() >= 1 {
                assert!(harmonic(statistics) <= best);
                if other < position {
                    assert!(harmonic(statistics) < best, "ties go to the earliest");
                }
            }
        }
    }

    #[test]
    fn high_density_maximizes_coverage_above_the_minimum() {
        let found = found();
        let position = found
            .select_candidate(selection(6, high_density(0.8)))
            .unwrap();
        let chosen = found.trajectory()[position].statistics();
        assert!(chosen.density() >= 0.8);
        assert!(found.trajectory().iter().all(|step| {
            let statistics = step.statistics();
            statistics.density() < 0.8
                || statistics.restricted_dimensions() == 0
                || statistics.coverage() <= chosen.coverage()
        }));
    }

    #[test]
    fn the_unrestricted_box_and_wide_boxes_are_not_eligible() {
        let found = found();
        assert_eq!(
            found.trajectory()[0].statistics().restricted_dimensions(),
            0
        );
        let one = found
            .select_candidate(selection(1, SelectionObjective::Balanced))
            .unwrap();
        assert_eq!(
            found.trajectory()[one].statistics().restricted_dimensions(),
            1
        );
    }

    #[test]
    fn the_unrestricted_box_is_eligible_unless_excluded() {
        // The property holds in 95 of 100 rows, so the unrestricted box is
        // already dense and has full coverage.
        let dataset = Dataset::new(
            vec![Feature::continuous("x", (0..100).map(f64::from).collect()).unwrap()],
            (0..100).map(|i| i >= 5).collect(),
        )
        .unwrap();
        let found = Prim::new(&dataset, PrimConfig::default())
            .find_box()
            .unwrap();
        let dense = CandidateSelection::new(6, high_density(0.9));
        assert_eq!(found.select_candidate(dense), Some(0));
        assert_ne!(
            found.select_candidate(dense.restricting_at_least_one()),
            Some(0)
        );
        assert_eq!(
            found.select_candidate(CandidateSelection::new(0, SelectionObjective::Balanced)),
            Some(0)
        );
    }

    #[test]
    fn no_candidate_without_the_density() {
        let dataset = Dataset::new(
            vec![Feature::continuous("x", (0..50).map(f64::from).collect()).unwrap()],
            (0..50).map(|i| i % 2 == 0).collect(),
        )
        .unwrap();
        let found = Prim::new(&dataset, PrimConfig::default())
            .find_box()
            .unwrap();
        assert_eq!(
            found.select_candidate(selection(3, high_density(1.0))),
            None
        );
    }
}
