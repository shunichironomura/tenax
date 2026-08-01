//! Reproducible sampling primitives over validated model domains.

use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
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
const MODEL_SEED_DOMAIN: u64 = 0x7465_6e61_785f_6d6f;

/// Errors raised while drawing a uniform model-input request.
#[derive(Debug, Error, PartialEq)]
pub enum SamplingError {
    /// A request cannot contain zero input rows.
    #[error("uniform sampling requires at least one row")]
    NoRows,

    /// A feature position cannot be represented by the deterministic seed
    /// derivation.
    #[error("feature position {position} exceeds the supported u64 stream space")]
    FeaturePositionOverflow {
        /// Zero-based schema feature position.
        position: usize,
    },

    /// A validated domain could not initialize its uniform distribution.
    #[error("input '{name}' cannot be represented by the uniform sampler")]
    UnsampleableDomain {
        /// Name of the affected model input.
        name: String,
    },

    /// Generated values unexpectedly violated a native feature invariant.
    #[error("uniform sampler generated an invalid feature: {0}")]
    InvalidFeature(#[from] DataError),

    /// Generated columns unexpectedly violated the source model schema.
    #[error("uniform sampler generated an invalid input chunk: {0}")]
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
    let inputs = InputChunk::new(schema, features)?;
    let id = EvaluationId::from_run_seed(run_seed, request_sequence);
    let model_seed = derive_seed(run_seed, request_sequence, MODEL_SEED_DOMAIN);
    Ok(EvalRequest::new(id, model_seed, inputs))
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
        assert_eq!(
            request.inputs().features()[0].continuous_values(),
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
    }
}
