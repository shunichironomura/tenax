//! What each restriction of a PRIM candidate box contributes: the box
//! re-measured with that one restriction removed.

use thiserror::Error;

use crate::data::Dataset;
use crate::prim::{BoxStatistics, FeatureLimit, PrimBox, restricted_count, rows_in_box};

/// Errors raised while measuring the restrictions of a candidate box.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RestrictionEffectError {
    /// The trajectory has no step at this position.
    #[error("trajectory position {position} is out of range for {steps} steps")]
    PositionOutOfRange {
        /// The requested position.
        position: usize,
        /// The number of trajectory steps.
        steps: usize,
    },

    /// The dataset is not the one PRIM searched: its features, rows, or cases
    /// disagree with the trajectory.
    #[error("the dataset is not the one this PRIM box was found in: {reason}")]
    DatasetMismatch {
        /// What disagrees.
        reason: String,
    },
}

/// One restriction of a candidate box and the box measured without it.
#[derive(Clone, Debug, PartialEq)]
pub struct RestrictionEffect {
    limit: FeatureLimit,
    with: BoxStatistics,
    without: BoxStatistics,
}

impl RestrictionEffect {
    /// Returns the candidate's restriction on the feature.
    #[must_use]
    pub const fn limit(&self) -> &FeatureLimit {
        &self.limit
    }

    /// Returns the measures of the box with every restriction.
    #[must_use]
    pub const fn with(&self) -> BoxStatistics {
        self.with
    }

    /// Returns the measures of the box without this restriction, over the
    /// rows the PRIM search started from.
    #[must_use]
    pub const fn without(&self) -> BoxStatistics {
        self.without
    }

    /// Returns how much this restriction raises the density: the box's
    /// density minus the density without the restriction. A restriction that
    /// barely raises the density adds little to the description.
    #[must_use]
    pub fn density_gain(&self) -> f64 {
        self.with.density() - self.without.density()
    }
}

impl PrimBox {
    /// Returns the restrictions of the candidate at trajectory `position`:
    /// the limits that differ from the unrestricted initial box, in feature
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`RestrictionEffectError::PositionOutOfRange`] when the
    /// trajectory has no such step.
    pub fn restricted_limits(
        &self,
        position: usize,
    ) -> Result<Vec<&FeatureLimit>, RestrictionEffectError> {
        let trajectory = self.trajectory();
        let step = trajectory
            .get(position)
            .ok_or(RestrictionEffectError::PositionOutOfRange {
                position,
                steps: trajectory.len(),
            })?;
        Ok(step
            .limits()
            .limits()
            .iter()
            .zip(trajectory[0].limits().limits())
            .filter(|(current, initial)| current.restriction() != initial.restriction())
            .map(|(current, _)| current)
            .collect())
    }

    /// Measures each restriction of the candidate at trajectory `position`
    /// by removing it alone, in feature order.
    ///
    /// `dataset` must be the dataset PRIM searched. The box without a
    /// restriction is counted over the rows the search started from (the
    /// initial step's rows) and measured like the trajectory: coverage and
    /// mass relative to the whole dataset. Bounds are inclusive.
    ///
    /// ```
    /// use tenax::{Dataset, Feature, Prim, PrimConfig};
    ///
    /// let load = (0..200).map(|i| f64::from(i % 100) / 100.0).collect::<Vec<_>>();
    /// let noise = (0..200).map(|i| f64::from((i * 37) % 200)).collect::<Vec<_>>();
    /// let cases = load.iter().map(|load| *load >= 0.7).collect();
    /// let dataset = Dataset::new(
    ///     vec![
    ///         Feature::continuous("load", load)?,
    ///         Feature::continuous("noise", noise)?,
    ///     ],
    ///     cases,
    /// )?;
    /// let found = Prim::new(&dataset, PrimConfig::default())
    ///     .find_box()
    ///     .expect("the dataset has rows");
    /// let last = found.trajectory().len() - 1;
    ///
    /// for effect in found.restriction_effects(&dataset, last)? {
    ///     println!("{}: +{:.2} density", effect.limit().name(), effect.density_gain());
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RestrictionEffectError::PositionOutOfRange`] when the
    /// trajectory has no such step, and
    /// [`RestrictionEffectError::DatasetMismatch`] when `dataset`'s features
    /// differ from the box's or recounting the candidate in `dataset` does
    /// not reproduce its trajectory measures.
    pub fn restriction_effects(
        &self,
        dataset: &Dataset,
        position: usize,
    ) -> Result<Vec<RestrictionEffect>, RestrictionEffectError> {
        let trajectory = self.trajectory();
        let step = trajectory
            .get(position)
            .ok_or(RestrictionEffectError::PositionOutOfRange {
                position,
                steps: trajectory.len(),
            })?;
        let initial = trajectory[0].limits();
        let population = trajectory[0].indices();
        check_dataset(self, dataset)?;
        let with = step.statistics();
        let recounted = BoxStatistics::of_rows(
            dataset,
            &rows_in_box(dataset, population, step.limits()),
            with.restricted_dimensions(),
        );
        if recounted != with {
            return Err(mismatch(format!(
                "the candidate holds {} rows and {} cases of interest in the dataset, \
                 but the trajectory reports {} and {}",
                recounted.points(),
                recounted.cases_of_interest(),
                with.points(),
                with.cases_of_interest()
            )));
        }
        Ok(step
            .limits()
            .limits()
            .iter()
            .zip(initial.limits())
            .enumerate()
            .filter(|(_, (current, original))| current.restriction() != original.restriction())
            .map(|(feature, (current, _))| {
                let relaxed = step.limits().with_restriction_of(initial, feature);
                RestrictionEffect {
                    limit: current.clone(),
                    with,
                    without: BoxStatistics::of_rows(
                        dataset,
                        &rows_in_box(dataset, population, &relaxed),
                        restricted_count(&relaxed, initial),
                    ),
                }
            })
            .collect())
    }
}

const fn mismatch(reason: String) -> RestrictionEffectError {
    RestrictionEffectError::DatasetMismatch { reason }
}

/// Checks what box membership relies on: one feature of the same name and
/// kind per limit, and every starting row in range.
fn check_dataset(found: &PrimBox, dataset: &Dataset) -> Result<(), RestrictionEffectError> {
    let initial = &found.trajectory()[0];
    let limits = initial.limits().limits();
    if limits.len() != dataset.features().len() {
        return Err(mismatch(format!(
            "the box spans {} features, the dataset has {}",
            limits.len(),
            dataset.features().len()
        )));
    }
    if let Some((feature, limit)) =
        dataset
            .features()
            .iter()
            .zip(limits)
            .find(|(feature, limit)| {
                feature.name() != limit.name() || feature.kind() != limit.restriction().kind()
            })
    {
        return Err(mismatch(format!(
            "feature '{}' ({:?}) is limited as '{}' ({:?})",
            feature.name(),
            feature.kind(),
            limit.name(),
            limit.restriction().kind()
        )));
    }
    if let Some(row) = initial
        .indices()
        .iter()
        .find(|row| **row >= dataset.row_count())
    {
        return Err(mismatch(format!(
            "the search started from row {row}, but the dataset has {} rows",
            dataset.row_count()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Feature, Prim, PrimConfig};

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

    fn last(found: &PrimBox) -> usize {
        found.trajectory().len() - 1
    }

    #[test]
    fn removing_a_needed_restriction_lowers_the_density() {
        let dataset = dataset();
        let found = Prim::new(&dataset, PrimConfig::default())
            .find_box()
            .unwrap();
        let position = last(&found);
        let effects = found.restriction_effects(&dataset, position).unwrap();
        let names = effects
            .iter()
            .map(|effect| effect.limit().name())
            .collect::<Vec<_>>();
        let restricted = found
            .restricted_limits(position)
            .unwrap()
            .into_iter()
            .map(FeatureLimit::name)
            .collect::<Vec<_>>();

        assert_eq!(names, restricted);
        assert!(names.contains(&"x") && names.contains(&"c"));
        assert!(!names.contains(&"n"));
        for effect in &effects {
            assert!(effect.density_gain() > 0.0, "{}", effect.limit().name());
            assert!(effect.without().points() > effect.with().points());
            assert!(effect.without().coverage() >= effect.with().coverage());
            assert_eq!(
                effect.without().restricted_dimensions() + 1,
                effect.with().restricted_dimensions()
            );
        }
    }

    #[test]
    fn the_initial_box_has_no_restrictions() {
        let dataset = dataset();
        let found = Prim::new(&dataset, PrimConfig::default())
            .find_box()
            .unwrap();
        assert_eq!(found.restricted_limits(0).unwrap().len(), 0);
        assert_eq!(found.restriction_effects(&dataset, 0).unwrap().len(), 0);
    }

    #[test]
    fn a_later_box_is_measured_over_its_remaining_rows() {
        let dataset = dataset();
        let mut prim = Prim::new(&dataset, PrimConfig::default());
        prim.find_box().unwrap();
        let second = prim.find_box().unwrap();
        let position = last(&second);

        for effect in second.restriction_effects(&dataset, position).unwrap() {
            assert!(effect.without().points() <= second.trajectory()[0].indices().len());
        }
    }

    #[test]
    fn rejects_other_datasets_and_positions() {
        let dataset = dataset();
        let found = Prim::new(&dataset, PrimConfig::default())
            .find_box()
            .unwrap();
        let steps = found.trajectory().len();
        assert_eq!(
            found.restriction_effects(&dataset, steps),
            Err(RestrictionEffectError::PositionOutOfRange {
                position: steps,
                steps
            })
        );

        let renamed = Dataset::new(
            vec![
                Feature::continuous("y", vec![0.0; 400]).unwrap(),
                dataset.features()[1].clone(),
                dataset.features()[2].clone(),
            ],
            dataset.cases_of_interest().to_vec(),
        )
        .unwrap();
        assert!(matches!(
            found.restriction_effects(&renamed, last(&found)),
            Err(RestrictionEffectError::DatasetMismatch { .. })
        ));

        let other_cases = Dataset::new(
            dataset.features().to_vec(),
            dataset
                .cases_of_interest()
                .iter()
                .map(|case| !case)
                .collect(),
        )
        .unwrap();
        assert!(matches!(
            found.restriction_effects(&other_cases, last(&found)),
            Err(RestrictionEffectError::DatasetMismatch { .. })
        ));
    }
}
