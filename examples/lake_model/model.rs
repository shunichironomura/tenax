//! Direct Policy Search lake model used by the EMA Workbench tutorials.
//!
//! The equations follow EMA Workbench's `dps_lake_model.py`, while random
//! draws use Tenax's explicitly seeded `ChaCha12` convention rather than
//! `NumPy`'s legacy global generator.

use std::fmt;

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, LogNormal, NormalError};
use thiserror::Error;

const MIN_RELEASE: f64 = 0.01;
const MAX_RELEASE: f64 = 0.1;
const ROOT_LOWER: f64 = 0.01;
const ROOT_UPPER: f64 = 1.5;
const ROOT_ITERATIONS: usize = 100;

/// Deeply uncertain lake parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LakeUncertainties {
    decay_rate: f64,
    recycling_exponent: f64,
    inflow_mean: f64,
    inflow_standard_deviation: f64,
    discount_rate: f64,
}

impl LakeUncertainties {
    pub fn new(
        decay_rate: f64,
        recycling_exponent: f64,
        inflow_mean: f64,
        inflow_standard_deviation: f64,
        discount_rate: f64,
    ) -> Result<Self, LakeModelError> {
        validate_positive(LakeParameter::DecayRate, decay_rate)?;
        validate_positive(LakeParameter::RecyclingExponent, recycling_exponent)?;
        validate_positive(LakeParameter::InflowMean, inflow_mean)?;
        validate_non_negative(
            LakeParameter::InflowStandardDeviation,
            inflow_standard_deviation,
        )?;
        validate_unit_interval(LakeParameter::DiscountRate, discount_rate)?;

        Ok(Self {
            decay_rate,
            recycling_exponent,
            inflow_mean,
            inflow_standard_deviation,
            discount_rate,
        })
    }
}

/// Five-parameter radial-basis-function pollution-release policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpsPolicy {
    centers: [f64; 2],
    radii: [f64; 2],
    first_weight: f64,
}

impl DpsPolicy {
    pub fn new(
        first_center: f64,
        second_center: f64,
        first_radius: f64,
        second_radius: f64,
        first_weight: f64,
    ) -> Result<Self, LakeModelError> {
        validate_finite(LakeParameter::FirstCenter, first_center)?;
        validate_finite(LakeParameter::SecondCenter, second_center)?;
        validate_positive(LakeParameter::FirstRadius, first_radius)?;
        validate_positive(LakeParameter::SecondRadius, second_radius)?;
        validate_closed_unit_interval(LakeParameter::FirstWeight, first_weight)?;

        Ok(Self {
            centers: [first_center, second_center],
            radii: [first_radius, second_radius],
            first_weight,
        })
    }

    fn anthropogenic_release(self, lake_phosphorus: f64) -> f64 {
        let first_response = ((lake_phosphorus - self.centers[0]).abs() / self.radii[0]).powi(3);
        let second_response = ((lake_phosphorus - self.centers[1]).abs() / self.radii[1]).powi(3);
        let rule = self
            .first_weight
            .mul_add(first_response, (1.0 - self.first_weight) * second_response);
        rule.clamp(MIN_RELEASE, MAX_RELEASE)
    }
}

/// Fixed controls for a stochastic lake-model evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LakeModel {
    utility_from_pollution: f64,
    realizations: u32,
    years: u32,
}

impl LakeModel {
    pub fn new(
        utility_from_pollution: f64,
        realizations: u32,
        years: u32,
    ) -> Result<Self, LakeModelError> {
        validate_finite(LakeParameter::UtilityFromPollution, utility_from_pollution)?;
        if realizations == 0 {
            return Err(LakeModelError::NoRealizations);
        }
        if years < 2 {
            return Err(LakeModelError::TooFewYears { years });
        }
        Ok(Self {
            utility_from_pollution,
            realizations,
            years,
        })
    }

    /// Runs all stochastic realizations with a local deterministic RNG.
    pub fn evaluate(
        self,
        uncertainties: LakeUncertainties,
        policy: DpsPolicy,
        seed: u64,
    ) -> Result<LakeOutcomes, LakeModelError> {
        let critical_phosphorus =
            critical_phosphorus(uncertainties.decay_rate, uncertainties.recycling_exponent)?;
        let inflow_distribution = LogNormal::from_mean_cv(
            uncertainties.inflow_mean,
            uncertainties.inflow_standard_deviation / uncertainties.inflow_mean,
        )
        .map_err(LakeModelError::InvalidInflowDistribution)?;
        let years = usize::try_from(self.years)
            .map_err(|_| LakeModelError::UnsupportedYearCount { years: self.years })?;
        let realization_denominator = f64::from(self.realizations);
        let observation_denominator = realization_denominator * f64::from(self.years);
        let mut average_phosphorus = vec![0.0; years];
        let mut reliability = 0.0;
        let mut inertia = 0.0;
        let mut utility = 0.0;
        let mut rng = ChaCha12Rng::seed_from_u64(seed);

        for realization in 0..self.realizations {
            // The upstream model draws `years` values and intentionally leaves
            // the final one unused. Retaining that draw keeps each realization
            // on an independent, fixed-width portion of the RNG stream.
            let natural_inflows = (0..years)
                .map(|_| inflow_distribution.sample(&mut rng))
                .collect::<Vec<f64>>();
            let mut lake_phosphorus = 0.0;
            let mut decision = MAX_RELEASE;
            let mut discount = 1.0;
            let mut reliable_years = u32::from(lake_phosphorus < critical_phosphorus);
            let mut inertial_changes = 0_u32;
            utility += self.utility_from_pollution * decision / realization_denominator;

            for year in 1..years {
                let next_decision = policy.anthropogenic_release(lake_phosphorus);
                let phosphorus_power = lake_phosphorus.powf(uncertainties.recycling_exponent);
                lake_phosphorus = (1.0 - uncertainties.decay_rate)
                    .mul_add(lake_phosphorus, phosphorus_power / (1.0 + phosphorus_power))
                    + next_decision
                    + natural_inflows[year - 1];
                if !lake_phosphorus.is_finite() {
                    return Err(LakeModelError::NonFiniteState { realization, year });
                }

                average_phosphorus[year] += lake_phosphorus / realization_denominator;
                reliable_years += u32::from(lake_phosphorus < critical_phosphorus);
                inertial_changes += u32::from((next_decision - decision).abs() < 0.02);
                discount *= uncertainties.discount_rate;
                utility += self.utility_from_pollution * next_decision * discount
                    / realization_denominator;
                decision = next_decision;
            }

            reliability += f64::from(reliable_years) / observation_denominator;
            inertia += f64::from(inertial_changes) / observation_denominator;
        }

        let max_phosphorus = average_phosphorus.into_iter().fold(0.0_f64, f64::max);
        let outcomes = LakeOutcomes {
            max_phosphorus,
            utility,
            inertia,
            reliability,
        };
        outcomes.validate()?;
        Ok(outcomes)
    }
}

/// Scalar outcomes calculated by the tutorial model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LakeOutcomes {
    max_phosphorus: f64,
    utility: f64,
    inertia: f64,
    reliability: f64,
}

impl LakeOutcomes {
    /// Returns the maximum of mean annual lake phosphorus.
    pub const fn max_phosphorus(self) -> f64 {
        self.max_phosphorus
    }

    fn validate(self) -> Result<(), LakeModelError> {
        [
            (LakeOutcome::MaxPhosphorus, self.max_phosphorus),
            (LakeOutcome::Utility, self.utility),
            (LakeOutcome::Inertia, self.inertia),
            (LakeOutcome::Reliability, self.reliability),
        ]
        .into_iter()
        .find(|(_, value)| !value.is_finite())
        .map_or(Ok(()), |(outcome, value)| {
            Err(LakeModelError::NonFiniteOutcome { outcome, value })
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LakeParameter {
    DecayRate,
    RecyclingExponent,
    InflowMean,
    InflowStandardDeviation,
    DiscountRate,
    FirstCenter,
    SecondCenter,
    FirstRadius,
    SecondRadius,
    FirstWeight,
    UtilityFromPollution,
}

impl fmt::Display for LakeParameter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DecayRate => "decay rate (b)",
            Self::RecyclingExponent => "recycling exponent (q)",
            Self::InflowMean => "natural-inflow mean",
            Self::InflowStandardDeviation => "natural-inflow standard deviation",
            Self::DiscountRate => "discount rate (delta)",
            Self::FirstCenter => "first RBF center (c1)",
            Self::SecondCenter => "second RBF center (c2)",
            Self::FirstRadius => "first RBF radius (r1)",
            Self::SecondRadius => "second RBF radius (r2)",
            Self::FirstWeight => "first RBF weight (w1)",
            Self::UtilityFromPollution => "utility from pollution (alpha)",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LakeOutcome {
    MaxPhosphorus,
    Utility,
    Inertia,
    Reliability,
}

impl fmt::Display for LakeOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::MaxPhosphorus => "maximum phosphorus",
            Self::Utility => "utility",
            Self::Inertia => "inertia",
            Self::Reliability => "reliability",
        })
    }
}

/// Explicit model-domain and numerical failures.
#[derive(Debug, Error)]
pub enum LakeModelError {
    #[error("{parameter} must be finite, got {value}")]
    NonFiniteParameter {
        parameter: LakeParameter,
        value: f64,
    },

    #[error("{parameter} must be greater than zero, got {value}")]
    NonPositiveParameter {
        parameter: LakeParameter,
        value: f64,
    },

    #[error("{parameter} must be non-negative, got {value}")]
    NegativeParameter {
        parameter: LakeParameter,
        value: f64,
    },

    #[error("{parameter} must be in (0, 1], got {value}")]
    NotInUnitInterval {
        parameter: LakeParameter,
        value: f64,
    },

    #[error("{parameter} must be in [0, 1], got {value}")]
    NotInClosedUnitInterval {
        parameter: LakeParameter,
        value: f64,
    },

    #[error("the lake model requires at least one stochastic realization")]
    NoRealizations,

    #[error("the lake model requires at least two years, got {years}")]
    TooFewYears { years: u32 },

    #[error("year count {years} is not supported on this platform")]
    UnsupportedYearCount { years: u32 },

    #[error("natural-inflow parameters do not define a log-normal distribution: {0}")]
    InvalidInflowDistribution(#[source] NormalError),

    #[error(
        "the critical-phosphorus root is not bracketed for b={decay_rate}, q={recycling_exponent}"
    )]
    CriticalPhosphorusNotBracketed {
        decay_rate: f64,
        recycling_exponent: f64,
    },

    #[error("lake phosphorus became non-finite in realization {realization}, year {year}")]
    NonFiniteState { realization: u32, year: usize },

    #[error("{outcome} outcome is non-finite: {value}")]
    NonFiniteOutcome { outcome: LakeOutcome, value: f64 },
}

fn critical_phosphorus(decay_rate: f64, recycling_exponent: f64) -> Result<f64, LakeModelError> {
    let balance = |phosphorus: f64| {
        let power = phosphorus.powf(recycling_exponent);
        decay_rate.mul_add(-phosphorus, power / (1.0 + power))
    };
    let mut lower = ROOT_LOWER;
    let mut upper = ROOT_UPPER;
    let mut lower_value = balance(lower);
    let upper_value = balance(upper);
    if lower_value.is_sign_negative() == upper_value.is_sign_negative() {
        return Err(LakeModelError::CriticalPhosphorusNotBracketed {
            decay_rate,
            recycling_exponent,
        });
    }

    for _ in 0..ROOT_ITERATIONS {
        let midpoint = f64::midpoint(lower, upper);
        let midpoint_value = balance(midpoint);
        if lower_value.is_sign_negative() == midpoint_value.is_sign_negative() {
            lower = midpoint;
            lower_value = midpoint_value;
        } else {
            upper = midpoint;
        }
    }
    Ok(f64::midpoint(lower, upper))
}

const fn validate_finite(parameter: LakeParameter, value: f64) -> Result<(), LakeModelError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(LakeModelError::NonFiniteParameter { parameter, value })
    }
}

fn validate_positive(parameter: LakeParameter, value: f64) -> Result<(), LakeModelError> {
    validate_finite(parameter, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(LakeModelError::NonPositiveParameter { parameter, value })
    }
}

fn validate_non_negative(parameter: LakeParameter, value: f64) -> Result<(), LakeModelError> {
    validate_finite(parameter, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(LakeModelError::NegativeParameter { parameter, value })
    }
}

fn validate_unit_interval(parameter: LakeParameter, value: f64) -> Result<(), LakeModelError> {
    validate_finite(parameter, value)?;
    if 0.0 < value && value <= 1.0 {
        Ok(())
    } else {
        Err(LakeModelError::NotInUnitInterval { parameter, value })
    }
}

fn validate_closed_unit_interval(
    parameter: LakeParameter,
    value: f64,
) -> Result<(), LakeModelError> {
    validate_finite(parameter, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(LakeModelError::NotInClosedUnitInterval { parameter, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uncertainties() -> LakeUncertainties {
        LakeUncertainties::new(0.42, 2.0, 0.02, 0.001, 0.98)
            .expect("documentation defaults are valid")
    }

    fn policy() -> DpsPolicy {
        DpsPolicy::new(0.25, 0.25, 0.5, 0.5, 0.5).expect("documentation defaults are valid")
    }

    #[test]
    fn release_rule_is_clamped_to_documented_bounds() {
        let policy = policy();
        assert_eq!(policy.anthropogenic_release(0.25), MIN_RELEASE);
        assert_eq!(policy.anthropogenic_release(2.0), MAX_RELEASE);
    }

    #[test]
    fn model_is_reproducible_for_a_row_seed() {
        let model = LakeModel::new(0.41, 4, 12).expect("small model controls are valid");
        let first = model
            .evaluate(uncertainties(), policy(), 42)
            .expect("valid lake model evaluates");
        let repeated = model
            .evaluate(uncertainties(), policy(), 42)
            .expect("valid lake model evaluates");
        assert_eq!(first, repeated);
        assert!(first.max_phosphorus().is_finite());
        assert!(first.utility.is_finite());
        assert!((0.0..=1.0).contains(&first.inertia));
        assert!((0.0..=1.0).contains(&first.reliability));
    }

    #[test]
    fn deterministic_inflow_matches_an_independent_equation_trace() {
        let model = LakeModel::new(0.41, 1, 5).expect("small model controls are valid");
        let deterministic = LakeUncertainties::new(0.42, 2.0, 0.02, 0.0, 0.98)
            .expect("zero-variance inflow is valid");
        let outcomes = model
            .evaluate(deterministic, policy(), 42)
            .expect("deterministic lake model evaluates");

        // Calculated independently from the five annual recurrence equations.
        assert!((outcomes.max_phosphorus() - 0.122_176_788_894_415_48).abs() < 1.0e-14);
        assert!((outcomes.utility - 0.101_024_598_706_517_28).abs() < 1.0e-14);
        assert!((outcomes.inertia - 0.6).abs() < f64::EPSILON);
        assert!((outcomes.reliability - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_policy_radius_is_rejected_explicitly() {
        assert!(matches!(
            DpsPolicy::new(0.25, 0.25, 0.0, 0.5, 0.5),
            Err(LakeModelError::NonPositiveParameter {
                parameter: LakeParameter::FirstRadius,
                ..
            })
        ));
    }
}
