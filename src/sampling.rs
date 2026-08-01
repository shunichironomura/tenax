//! Reproducible sampling primitives over validated model domains.

use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform, Uniform};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;
use thiserror::Error;

use crate::DataError;
use crate::data::Feature;
use crate::evaluation::{EvalRequest, EvaluationId};
use crate::input::{InputChunk, InputChunkError};
use crate::schema::{FeatureDomain, ModelSchema};
use crate::seed::derive_seed;

const SAMPLE_REQUEST_DOMAIN: u64 = 0x7465_6e61_785f_7361;
const SAMPLE_FEATURE_DOMAIN: u64 = 0x7465_6e61_785f_6665;
const LHS_REQUEST_DOMAIN: u64 = 0x7465_6e61_785f_6c68;
const LHS_FEATURE_DOMAIN: u64 = 0x7465_6e61_785f_6c66;
const MODEL_SEED_DOMAIN: u64 = 0x7465_6e61_785f_6d6f;
const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << f64::MANTISSA_DIGITS;

/// Errors raised while drawing a model-input request.
#[derive(Debug, Error, PartialEq)]
pub enum SamplingError {
    /// A request cannot contain zero input rows.
    #[error("sampling requires at least one row")]
    NoRows,

    /// The row count cannot be represented by deterministic sampler arithmetic.
    #[error("row count {row_count} exceeds the supported u64 sampling space")]
    RowCountOverflow {
        /// Rejected number of rows.
        row_count: usize,
    },

    /// A continuous Latin-hypercube design needs exactly representable stratum
    /// indices in `f64`.
    #[error(
        "Latin hypercube sampling of continuous inputs supports at most {maximum} rows, got {row_count}"
    )]
    TooManyContinuousStrata {
        /// Rejected number of strata.
        row_count: usize,
        /// Largest exactly represented integer stratum count.
        maximum: u64,
    },

    /// A feature position cannot be represented by the deterministic seed
    /// derivation.
    #[error("feature position {position} exceeds the supported u64 stream space")]
    FeaturePositionOverflow {
        /// Zero-based schema feature position.
        position: usize,
    },

    /// A validated domain could not initialize its uniform distribution.
    #[error("input '{name}' cannot be represented by this sampler")]
    UnsampleableDomain {
        /// Name of the affected model input.
        name: String,
    },

    /// Generated values unexpectedly violated a native feature invariant.
    #[error("sampler generated an invalid feature: {0}")]
    InvalidFeature(#[from] DataError),

    /// Generated columns unexpectedly violated the source model schema.
    #[error("sampler generated an invalid input chunk: {0}")]
    InvalidChunk(#[from] InputChunkError),
}

/// Draws one reproducible uniform request over every input domain.
///
/// Continuous values are sampled over closed finite bounds, integers over
/// closed integer bounds, and categories with equal probability. Each feature
/// uses a separately derived `ChaCha12` stream, while the model seed uses another
/// domain-separated stream. The returned evaluation ID combines `run_seed` and
/// `request_sequence` directly. Reusing a sequence intentionally reproduces
/// the same idempotent request; distinct requests in one run must use distinct
/// sequence values.
///
/// Repeating a call with the same schema and arguments produces identical
/// values, ID, and model seed for a fixed Tenax dependency set. Callers should
/// retain their dependency lockfile when exact sampled values must survive a
/// library upgrade.
///
/// # Errors
///
/// Returns [`SamplingError::NoRows`] for an empty request, or another
/// [`SamplingError`] if a validated domain cannot be sampled or generated data
/// violates an invariant.
pub fn sample_uniform(
    schema: &ModelSchema,
    row_count: usize,
    run_seed: u64,
    request_sequence: u64,
) -> Result<EvalRequest, SamplingError> {
    if row_count == 0 {
        return Err(SamplingError::NoRows);
    }

    let request_sampling_seed = derive_seed(run_seed, request_sequence, SAMPLE_REQUEST_DOMAIN);
    let features =
        schema
            .inputs()
            .iter()
            .enumerate()
            .map(|(position, input)| {
                let position = u64::try_from(position)
                    .map_err(|_| SamplingError::FeaturePositionOverflow { position })?;
                let mut rng = ChaCha12Rng::seed_from_u64(derive_seed(
                    request_sampling_seed,
                    position,
                    SAMPLE_FEATURE_DOMAIN,
                ));
                match input.domain() {
                    FeatureDomain::Continuous(domain) => {
                        let distribution = Uniform::new_inclusive(domain.lower(), domain.upper())
                            .map_err(|_| SamplingError::UnsampleableDomain {
                            name: input.name().to_owned(),
                        })?;
                        let values = (0..row_count)
                            .map(|_| distribution.sample(&mut rng))
                            .collect();
                        Feature::continuous(input.name(), values).map_err(Into::into)
                    }
                    FeatureDomain::Integer(domain) => {
                        let distribution = Uniform::new_inclusive(domain.lower(), domain.upper())
                            .map_err(|_| SamplingError::UnsampleableDomain {
                            name: input.name().to_owned(),
                        })?;
                        let values = (0..row_count)
                            .map(|_| distribution.sample(&mut rng))
                            .collect();
                        Feature::integer(input.name(), values).map_err(Into::into)
                    }
                    FeatureDomain::Categorical(domain) => {
                        let distribution = Uniform::try_from(0..domain.categories().len())
                            .map_err(|_| SamplingError::UnsampleableDomain {
                                name: input.name().to_owned(),
                            })?;
                        let values = (0..row_count)
                            .map(|_| domain.categories()[distribution.sample(&mut rng)].clone())
                            .collect();
                        Feature::categorical(input.name(), values).map_err(Into::into)
                    }
                }
            })
            .collect::<Result<Vec<_>, SamplingError>>()?;
    sampled_request(schema, features, run_seed, request_sequence)
}

/// Draws one reproducible Latin-hypercube request over every input domain.
///
/// Each continuous input has exactly one randomized point in each of
/// `row_count` equal-probability strata, followed by an independently seeded
/// permutation. Integer and categorical inputs use midpoint inverse-CDF
/// strata and an independent permutation. The discrete construction is
/// balanced even when there are fewer domain values than requested rows; it
/// deliberately repeats levels rather than fabricating continuous meaning for
/// a nominal or integer value.
///
/// Feature streams, the request ID, and the model seed are deterministic and
/// domain-separated. Repeating a call with the same schema and arguments
/// produces the same request for a fixed Tenax dependency set. As with
/// [`sample_uniform`], callers must not reuse one `(run_seed,
/// request_sequence)` pair for a logically different request.
///
/// # Errors
///
/// Returns [`SamplingError::NoRows`] for an empty design,
/// [`SamplingError::TooManyContinuousStrata`] when a continuous design's
/// stratum indices cannot be represented exactly by `f64`, or another
/// [`SamplingError`] if generated values violate a schema invariant.
pub fn sample_latin_hypercube(
    schema: &ModelSchema,
    row_count: usize,
    run_seed: u64,
    request_sequence: u64,
) -> Result<EvalRequest, SamplingError> {
    if row_count == 0 {
        return Err(SamplingError::NoRows);
    }
    let row_count_u64 =
        u64::try_from(row_count).map_err(|_| SamplingError::RowCountOverflow { row_count })?;
    let has_continuous = schema
        .inputs()
        .iter()
        .any(|input| matches!(input.domain(), FeatureDomain::Continuous(_)));
    if has_continuous && row_count_u64 > MAX_EXACT_F64_INTEGER {
        return Err(SamplingError::TooManyContinuousStrata {
            row_count,
            maximum: MAX_EXACT_F64_INTEGER,
        });
    }

    let request_sampling_seed = derive_seed(run_seed, request_sequence, LHS_REQUEST_DOMAIN);
    let features = schema
        .inputs()
        .iter()
        .enumerate()
        .map(|(position, input)| {
            let position = u64::try_from(position)
                .map_err(|_| SamplingError::FeaturePositionOverflow { position })?;
            let mut rng = ChaCha12Rng::seed_from_u64(derive_seed(
                request_sampling_seed,
                position,
                LHS_FEATURE_DOMAIN,
            ));
            match input.domain() {
                FeatureDomain::Continuous(domain) => {
                    let strata_count = exact_u64_as_f64(row_count_u64);
                    let width = domain.upper() - domain.lower();
                    let mut values = (0..row_count_u64)
                        .map(|stratum| {
                            let jitter: f64 = StandardUniform.sample(&mut rng);
                            let quantile = (exact_u64_as_f64(stratum) + jitter) / strata_count;
                            quantile
                                .mul_add(width, domain.lower())
                                .clamp(domain.lower(), domain.upper())
                        })
                        .collect::<Vec<_>>();
                    values.shuffle(&mut rng);
                    Feature::continuous(input.name(), values).map_err(Into::into)
                }
                FeatureDomain::Integer(domain) => {
                    let cardinality =
                        u128::try_from(i128::from(domain.upper()) - i128::from(domain.lower()) + 1)
                            .map_err(|_| SamplingError::UnsampleableDomain {
                                name: input.name().to_owned(),
                            })?;
                    let levels = discrete_lhs_levels(cardinality, row_count_u64, &mut rng);
                    let values = levels
                        .into_iter()
                        .map(|level| {
                            i64::try_from(i128::from(domain.lower()) + i128::try_from(level)?)
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| SamplingError::UnsampleableDomain {
                            name: input.name().to_owned(),
                        })?;
                    Feature::integer(input.name(), values).map_err(Into::into)
                }
                FeatureDomain::Categorical(domain) => {
                    let cardinality = u128::try_from(domain.categories().len()).map_err(|_| {
                        SamplingError::UnsampleableDomain {
                            name: input.name().to_owned(),
                        }
                    })?;
                    let levels = discrete_lhs_levels(cardinality, row_count_u64, &mut rng);
                    let values = levels
                        .into_iter()
                        .map(|level| {
                            usize::try_from(level)
                                .ok()
                                .and_then(|position| domain.categories().get(position))
                                .cloned()
                                .ok_or_else(|| SamplingError::UnsampleableDomain {
                                    name: input.name().to_owned(),
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Feature::categorical(input.name(), values).map_err(Into::into)
                }
            }
        })
        .collect::<Result<Vec<_>, SamplingError>>()?;

    sampled_request(schema, features, run_seed, request_sequence)
}

fn sampled_request(
    schema: &ModelSchema,
    features: Vec<Feature>,
    run_seed: u64,
    request_sequence: u64,
) -> Result<EvalRequest, SamplingError> {
    let inputs = InputChunk::new(schema, features)?;
    let id = EvaluationId::from_run_seed(run_seed, request_sequence);
    let model_seed = derive_seed(run_seed, request_sequence, MODEL_SEED_DOMAIN);
    Ok(EvalRequest::new(id, model_seed, inputs))
}

fn discrete_lhs_levels(cardinality: u128, row_count: u64, rng: &mut ChaCha12Rng) -> Vec<u128> {
    debug_assert!(cardinality > 0);
    debug_assert!(row_count > 0);
    let row_count = u128::from(row_count);
    let midpoint = cardinality / 2;
    let mut levels = (0..row_count)
        .map(|stratum| (stratum * cardinality + midpoint) / row_count)
        .collect::<Vec<_>>();
    levels.shuffle(rng);
    levels
}

#[expect(
    clippy::cast_precision_loss,
    reason = "callers prove values are at most 2^53 and therefore exactly representable by f64"
)]
const fn exact_u64_as_f64(value: u64) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureView, InputSchema, OutputSchema};

    fn mixed_schema() -> ModelSchema {
        ModelSchema::new(
            vec![
                InputSchema::continuous("x", -1.0, 1.0).unwrap(),
                InputSchema::integer("count", -2, 2).unwrap(),
                InputSchema::categorical("mode", vec!["safe".to_owned(), "risky".to_owned()])
                    .unwrap(),
            ],
            vec![OutputSchema::boolean("failure").unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn uniform_sampling_is_reproducible_and_in_domain() {
        let schema = mixed_schema();
        let first = sample_uniform(&schema, 512, 42, 3).unwrap();
        let repeated = sample_uniform(&schema, 512, 42, 3).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.id().get(), (u128::from(42_u64) << 64) | 3);
        assert_eq!(first.seed(), 0x5027_a070_f5ed_0cd9);

        let views = first.inputs().feature_views().collect::<Vec<_>>();
        let FeatureView::Continuous(continuous) = views[0] else {
            panic!("wrong sampled feature kind");
        };
        assert!(continuous.iter().all(|value| (-1.0..=1.0).contains(value)));
        let FeatureView::Integer(integers) = views[1] else {
            panic!("wrong sampled feature kind");
        };
        assert!(integers.iter().all(|value| (-2..=2).contains(value)));
        let FeatureView::Categorical(categories) = views[2] else {
            panic!("wrong sampled feature kind");
        };
        assert!(
            categories
                .codes()
                .iter()
                .all(|code| categories.category(*code).is_some())
        );
    }

    #[test]
    fn request_sequences_separate_ids_seeds_and_samples() {
        let schema = mixed_schema();
        let first = sample_uniform(&schema, 8, 42, 0).unwrap();
        let second = sample_uniform(&schema, 8, 42, 1).unwrap();
        assert_ne!(first.id(), second.id());
        assert_ne!(first.seed(), second.seed());
        assert_ne!(first.inputs(), second.inputs());
    }

    #[test]
    fn latin_hypercube_sampling_is_reproducible_and_stratified() {
        let schema = mixed_schema();
        let first = sample_latin_hypercube(&schema, 10, 42, 3).unwrap();
        let repeated = sample_latin_hypercube(&schema, 10, 42, 3).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.id(), EvaluationId::from_run_seed(42, 3));
        assert_eq!(
            first.seed(),
            sample_uniform(&schema, 10, 42, 3).unwrap().seed()
        );

        let views = first.inputs().feature_views().collect::<Vec<_>>();
        let FeatureView::Continuous(continuous) = views[0] else {
            panic!("wrong sampled feature kind");
        };
        let mut continuous = continuous.to_vec();
        continuous.sort_by(f64::total_cmp);
        for (stratum, value) in continuous.into_iter().enumerate() {
            let stratum = exact_u64_as_f64(u64::try_from(stratum).unwrap());
            let row_count = exact_u64_as_f64(10);
            let lower = -1.0 + 2.0 * stratum / row_count;
            let upper = -1.0 + 2.0 * (stratum + 1.0) / row_count;
            assert!((lower..=upper).contains(&value));
        }

        let FeatureView::Integer(integers) = views[1] else {
            panic!("wrong sampled feature kind");
        };
        let mut integers = integers.to_vec();
        integers.sort_unstable();
        assert_eq!(integers, [-2, -2, -1, -1, 0, 0, 1, 1, 2, 2]);

        let FeatureView::Categorical(categories) = views[2] else {
            panic!("wrong sampled feature kind");
        };
        let safe_count = categories
            .codes()
            .iter()
            .filter(|code| categories.category(**code) == Some("safe"))
            .count();
        assert_eq!(safe_count, 5);
    }

    #[test]
    fn latin_hypercube_supports_the_complete_i64_domain() {
        let schema = ModelSchema::new(
            vec![InputSchema::integer("value", i64::MIN, i64::MAX).unwrap()],
            vec![OutputSchema::boolean("failure").unwrap()],
        )
        .unwrap();
        let request = sample_latin_hypercube(&schema, 5, 7, 0).unwrap();
        let values = request.inputs().features()[0].integer_values().unwrap();
        assert_eq!(values.len(), 5);
        assert!(values.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn singleton_domains_are_supported() {
        let schema = ModelSchema::new(
            vec![
                InputSchema::continuous("x", 0.5, 0.5).unwrap(),
                InputSchema::integer("count", 7, 7).unwrap(),
                InputSchema::categorical("mode", vec!["only".to_owned()]).unwrap(),
            ],
            vec![OutputSchema::boolean("failure").unwrap()],
        )
        .unwrap();
        let request = sample_uniform(&schema, 4, 1, 0).unwrap();
        let lhs_request = sample_latin_hypercube(&schema, 4, 1, 1).unwrap();
        assert_eq!(
            request.inputs().features()[0].continuous_values(),
            Some(&[0.5; 4][..])
        );
        assert_eq!(
            lhs_request.inputs().features()[0].continuous_values(),
            Some(&[0.5; 4][..])
        );
        assert_eq!(
            request.inputs().features()[1].integer_values(),
            Some(&[7; 4][..])
        );
        assert_eq!(
            request.inputs().features()[2].categorical_codes(),
            Some(&[0; 4][..])
        );
    }

    #[test]
    fn rejects_empty_requests() {
        assert_eq!(
            sample_uniform(&mixed_schema(), 0, 42, 0),
            Err(SamplingError::NoRows)
        );
        assert_eq!(
            sample_latin_hypercube(&mixed_schema(), 0, 42, 0),
            Err(SamplingError::NoRows)
        );
    }
}
