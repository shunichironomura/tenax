//! Reproducible sampling primitives over session input domains.

use std::sync::Arc;

use graphcal_model_protocol::{
    Domain, FiniteF64, InputColumn, KeyPosition, KeyType, PortName, RequestBatch,
    RequestBatchError, SessionInput, SessionSchema, WireType,
};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform, Uniform};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;
use thiserror::Error;

use crate::evaluation::{EvalRequest, EvaluationId};
use crate::seed::derive_seed;

const SAMPLE_REQUEST_DOMAIN: u64 = 0x7465_6e61_785f_7361;
const SAMPLE_FEATURE_DOMAIN: u64 = 0x7465_6e61_785f_6665;
const LHS_REQUEST_DOMAIN: u64 = 0x7465_6e61_785f_6c68;
const LHS_FEATURE_DOMAIN: u64 = 0x7465_6e61_785f_6c66;
const MODEL_SEED_DOMAIN: u64 = 0x7465_6e61_785f_6d6f;
const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << f64::MANTISSA_DIGITS;

/// The two levels of a Boolean input, sampled like a two-level categorical
/// input.
const BOOL_LEVELS: [bool; 2] = [false, true];

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

    /// An input position cannot be represented by the deterministic seed
    /// derivation.
    #[error("input position {position} exceeds the supported u64 stream space")]
    FeaturePositionOverflow {
        /// Zero-based session input position.
        position: usize,
    },

    /// A quantity or integer input does not declare both domain bounds, so it
    /// has no finite sampling interval.
    #[error("input '{name}' must declare both a lower and an upper domain bound to be sampled")]
    UnboundedInput {
        /// Name of the affected input.
        name: PortName,
    },

    /// A domain could not initialize its sampling distribution, for example
    /// because the interval width overflows `f64`.
    #[error("input '{name}' cannot be represented by this sampler")]
    UnsampleableDomain {
        /// Name of the affected input.
        name: PortName,
    },

    /// The input's type has no sampling interval or set of levels: a complex,
    /// indexed, or algebraic value.
    #[error("input '{name}' is a {kind} value, which the samplers cannot draw")]
    UnsupportedInput {
        /// Name of the affected input.
        name: PortName,
        /// The input's wire kind.
        kind: &'static str,
    },

    /// Generated columns unexpectedly violated the session schema.
    #[error("sampler generated an invalid request batch: {0}")]
    InvalidBatch(#[from] RequestBatchError),
}

/// Draws one reproducible uniform request over every session input.
///
/// Quantities and datetimes (seconds since J2000) are sampled over their
/// closed declared bounds, integers over
/// their closed integer bounds, key variants and Booleans (`false`, `true`)
/// with equal probability. Each input uses a separately derived `ChaCha12`
/// stream, while the model seed uses another domain-separated stream. The
/// returned evaluation ID combines `run_seed` and `request_sequence` directly.
/// Reusing a sequence intentionally reproduces the same idempotent request;
/// distinct requests in one run must use distinct sequence values.
///
/// Repeating a call with the same schema and arguments produces identical
/// values, ID, and model seed for a fixed Tenax dependency set. Callers should
/// retain their dependency lockfile when exact sampled values must survive a
/// library upgrade.
///
/// # Errors
///
/// Returns [`SamplingError::NoRows`] for an empty request,
/// [`SamplingError::UnboundedInput`] for a quantity or integer input without
/// both domain bounds, or another [`SamplingError`] if a domain cannot be
/// sampled.
pub fn sample_uniform(
    schema: &Arc<SessionSchema>,
    row_count: usize,
    run_seed: u64,
    request_sequence: u64,
) -> Result<EvalRequest, SamplingError> {
    if row_count == 0 {
        return Err(SamplingError::NoRows);
    }

    let request_sampling_seed = derive_seed(run_seed, request_sequence, SAMPLE_REQUEST_DOMAIN);
    let columns = schema
        .inputs()
        .iter()
        .enumerate()
        .map(|(position, input)| {
            let mut rng = input_rng(request_sampling_seed, position, SAMPLE_FEATURE_DOMAIN)?;
            uniform_column(input, row_count, &mut rng)
        })
        .collect::<Result<Vec<_>, SamplingError>>()?;
    sampled_request(schema, columns, run_seed, request_sequence)
}

fn uniform_column(
    input: &SessionInput,
    row_count: usize,
    rng: &mut ChaCha12Rng,
) -> Result<InputColumn, SamplingError> {
    let unsampleable = || SamplingError::UnsampleableDomain {
        name: input.name().clone(),
    };
    match input.wire() {
        WireType::Quantity(_) | WireType::Datetime(_) => {
            let (lower, upper) = continuous_bounds(input)?;
            let distribution = Uniform::new_inclusive(lower, upper).map_err(|_| unsampleable())?;
            let values = (0..row_count)
                .map(|_| FiniteF64::new(distribution.sample(rng)).map_err(|_| unsampleable()))
                .collect::<Result<_, _>>()?;
            Ok(continuous_column(input, values))
        }
        WireType::Int => {
            let (lower, upper) = integer_bounds(input)?;
            let distribution = Uniform::new_inclusive(lower, upper).map_err(|_| unsampleable())?;
            Ok(InputColumn::Int(
                (0..row_count).map(|_| distribution.sample(rng)).collect(),
            ))
        }
        WireType::Key(key) => {
            let positions = key_positions(key);
            let distribution = Uniform::try_from(0..positions.len()).map_err(|_| unsampleable())?;
            Ok(InputColumn::Key(
                (0..row_count)
                    .map(|_| positions[distribution.sample(rng)])
                    .collect(),
            ))
        }
        WireType::Bool => {
            let distribution =
                Uniform::try_from(0..BOOL_LEVELS.len()).map_err(|_| unsampleable())?;
            Ok(InputColumn::Bool(
                (0..row_count)
                    .map(|_| BOOL_LEVELS[distribution.sample(rng)])
                    .collect(),
            ))
        }
        WireType::Complex(_) | WireType::Indexed(_) | WireType::Algebraic(_) => {
            Err(unsupported(input))
        }
    }
}

/// Draws one reproducible Latin-hypercube request over every session input.
///
/// Each quantity or datetime input has exactly one randomized point in each
/// of `row_count`
/// equal-probability strata of its closed declared bounds, followed by an
/// independently seeded permutation. Integer, key, and Boolean inputs use
/// midpoint inverse-CDF strata over their levels (integers in order, key
/// variants in declaration order, then `false`, `true`) and an independent
/// permutation. The discrete construction is balanced even when there are
/// fewer levels than requested rows; it deliberately repeats levels rather
/// than fabricating continuous meaning for a nominal or integer value.
///
/// Input streams, the request ID, and the model seed are deterministic and
/// domain-separated. Repeating a call with the same schema and arguments
/// produces the same request for a fixed Tenax dependency set. As with
/// [`sample_uniform`], callers must not reuse one `(run_seed,
/// request_sequence)` pair for a logically different request.
///
/// # Errors
///
/// Returns [`SamplingError::NoRows`] for an empty design,
/// [`SamplingError::TooManyContinuousStrata`] when a quantity design's stratum
/// indices cannot be represented exactly by `f64`,
/// [`SamplingError::UnboundedInput`] for a quantity or integer input without
/// both domain bounds, or another [`SamplingError`] if a domain cannot be
/// sampled.
pub fn sample_latin_hypercube(
    schema: &Arc<SessionSchema>,
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
        .any(|input| matches!(input.wire(), WireType::Quantity(_) | WireType::Datetime(_)));
    if has_continuous && row_count_u64 > MAX_EXACT_F64_INTEGER {
        return Err(SamplingError::TooManyContinuousStrata {
            row_count,
            maximum: MAX_EXACT_F64_INTEGER,
        });
    }

    let request_sampling_seed = derive_seed(run_seed, request_sequence, LHS_REQUEST_DOMAIN);
    let columns = schema
        .inputs()
        .iter()
        .enumerate()
        .map(|(position, input)| {
            let mut rng = input_rng(request_sampling_seed, position, LHS_FEATURE_DOMAIN)?;
            latin_hypercube_column(input, row_count_u64, &mut rng)
        })
        .collect::<Result<Vec<_>, SamplingError>>()?;

    sampled_request(schema, columns, run_seed, request_sequence)
}

fn latin_hypercube_column(
    input: &SessionInput,
    row_count: u64,
    rng: &mut ChaCha12Rng,
) -> Result<InputColumn, SamplingError> {
    let unsampleable = || SamplingError::UnsampleableDomain {
        name: input.name().clone(),
    };
    match input.wire() {
        WireType::Quantity(_) | WireType::Datetime(_) => {
            let (lower, upper) = continuous_bounds(input)?;
            let width = upper - lower;
            if !width.is_finite() {
                return Err(unsampleable());
            }
            let strata_count = exact_u64_as_f64(row_count);
            let mut values = (0..row_count)
                .map(|stratum| {
                    let jitter: f64 = StandardUniform.sample(rng);
                    let quantile = (exact_u64_as_f64(stratum) + jitter) / strata_count;
                    quantile.mul_add(width, lower).clamp(lower, upper)
                })
                .collect::<Vec<_>>();
            values.shuffle(rng);
            let values = values
                .into_iter()
                .map(|value| FiniteF64::new(value).map_err(|_| unsampleable()))
                .collect::<Result<_, _>>()?;
            Ok(continuous_column(input, values))
        }
        WireType::Int => {
            let (lower, upper) = integer_bounds(input)?;
            let cardinality = u128::try_from(i128::from(upper) - i128::from(lower) + 1)
                .map_err(|_| unsampleable())?;
            let values = discrete_lhs_levels(cardinality, row_count, rng)
                .into_iter()
                .map(|level| i64::try_from(i128::from(lower) + i128::try_from(level)?))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| unsampleable())?;
            Ok(InputColumn::Int(values))
        }
        WireType::Key(key) => {
            let positions = key_positions(key);
            let values = discrete_levels(&positions, row_count, rng).ok_or_else(unsampleable)?;
            Ok(InputColumn::Key(values))
        }
        WireType::Bool => {
            let values = discrete_levels(&BOOL_LEVELS, row_count, rng).ok_or_else(unsampleable)?;
            Ok(InputColumn::Bool(values))
        }
        WireType::Complex(_) | WireType::Indexed(_) | WireType::Algebraic(_) => {
            Err(unsupported(input))
        }
    }
}

/// Draws a balanced Latin-hypercube column over the nominal `levels`.
fn discrete_levels<T: Copy>(levels: &[T], row_count: u64, rng: &mut ChaCha12Rng) -> Option<Vec<T>> {
    let cardinality = u128::try_from(levels.len()).ok()?;
    discrete_lhs_levels(cardinality, row_count, rng)
        .into_iter()
        .map(|level| {
            usize::try_from(level)
                .ok()
                .and_then(|position| levels.get(position))
                .copied()
        })
        .collect()
}

fn input_rng(
    request_sampling_seed: u64,
    position: usize,
    domain: u64,
) -> Result<ChaCha12Rng, SamplingError> {
    let position =
        u64::try_from(position).map_err(|_| SamplingError::FeaturePositionOverflow { position })?;
    Ok(ChaCha12Rng::seed_from_u64(derive_seed(
        request_sampling_seed,
        position,
        domain,
    )))
}

/// The column of sampled continuous `values` for a quantity or datetime
/// input.
const fn continuous_column(input: &SessionInput, values: Vec<FiniteF64>) -> InputColumn {
    match input.wire() {
        WireType::Datetime(_) => InputColumn::Datetime(values),
        _ => InputColumn::Quantity(values),
    }
}

fn unsupported(input: &SessionInput) -> SamplingError {
    SamplingError::UnsupportedInput {
        name: input.name().clone(),
        kind: match input.wire() {
            WireType::Complex(_) => "complex",
            WireType::Indexed(_) => "indexed",
            _ => "algebraic",
        },
    }
}

/// The closed bounds of a quantity or datetime input's domain.
fn continuous_bounds(input: &SessionInput) -> Result<(f64, f64), SamplingError> {
    match input.domain() {
        Some(Domain::Quantity(bounds) | Domain::Datetime(bounds)) => {
            match (bounds.min(), bounds.max()) {
                (Some(lower), Some(upper)) => Ok((lower.get(), upper.get())),
                _ => Err(unbounded(input)),
            }
        }
        _ => Err(unbounded(input)),
    }
}

fn integer_bounds(input: &SessionInput) -> Result<(i64, i64), SamplingError> {
    match input.domain() {
        Some(Domain::Int(bounds)) => match (bounds.min(), bounds.max()) {
            (Some(lower), Some(upper)) => Ok((*lower, *upper)),
            _ => Err(unbounded(input)),
        },
        _ => Err(unbounded(input)),
    }
}

fn unbounded(input: &SessionInput) -> SamplingError {
    SamplingError::UnboundedInput {
        name: input.name().clone(),
    }
}

fn key_positions(key: &KeyType) -> Vec<KeyPosition> {
    key.positions().collect()
}

fn sampled_request(
    schema: &Arc<SessionSchema>,
    columns: Vec<InputColumn>,
    run_seed: u64,
    request_sequence: u64,
) -> Result<EvalRequest, SamplingError> {
    let inputs = RequestBatch::new(Arc::clone(schema), columns)?;
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
    use graphcal_model_protocol::{Bounds, Dimension, KeyRef, QuantityType, ValueRef};

    use super::*;
    use crate::ModelSchemaBuilder;

    fn mixed_schema() -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .continuous("x", -1.0, 1.0)
            .unwrap()
            .integer("count", -2, 2)
            .unwrap()
            .categorical("mode", "Mode", ["risky", "safe"])
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap()
    }

    fn quantities(request: &EvalRequest, position: usize) -> Vec<f64> {
        let InputColumn::Quantity(values) = &request.inputs().columns()[position] else {
            panic!("wrong sampled column kind");
        };
        values.iter().map(|value| value.get()).collect()
    }

    fn integers(request: &EvalRequest, position: usize) -> Vec<i64> {
        let InputColumn::Int(values) = &request.inputs().columns()[position] else {
            panic!("wrong sampled column kind");
        };
        values.clone()
    }

    fn variants(request: &EvalRequest, name: &str) -> Vec<String> {
        let slot = request.inputs().schema().input_slot(name).unwrap();
        (0..request.inputs().rows())
            .map(|row| match request.inputs().value(slot, row) {
                Some(ValueRef::Key(KeyRef::Variant(variant))) => variant.as_str().to_owned(),
                _ => panic!("wrong sampled column kind"),
            })
            .collect()
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Draws are pinned to the values produced before Tenax adopted the
    /// Graphcal protocol vocabulary, for equivalent continuous, integer, and
    /// (lexically ordered) categorical domains.
    #[test]
    fn draws_are_unchanged_for_equivalent_domains() {
        let schema = mixed_schema();
        let uniform = sample_uniform(&schema, 4, 42, 3).unwrap();
        assert_eq!(uniform.seed(), 0x5027_a070_f5ed_0cd9);
        assert_eq!(
            bits(&quantities(&uniform, 0)),
            [
                4_605_558_378_744_413_236,
                13_829_446_476_597_519_408,
                13_828_648_858_583_450_540,
                13_829_524_151_802_318_712
            ]
        );
        assert_eq!(integers(&uniform, 1), [-1, 0, -1, -2]);
        assert_eq!(
            variants(&uniform, "mode"),
            ["safe", "risky", "safe", "safe"]
        );

        let lhs = sample_latin_hypercube(&schema, 4, 42, 3).unwrap();
        assert_eq!(lhs.seed(), 0x5027_a070_f5ed_0cd9);
        assert_eq!(
            bits(&quantities(&lhs, 0)),
            [
                13_815_063_467_261_683_504,
                4_604_103_699_310_678_274,
                13_828_052_241_695_135_710,
                4_602_036_452_934_967_292
            ]
        );
        assert_eq!(integers(&lhs, 1), [1, -1, 2, -2]);
        assert_eq!(variants(&lhs, "mode"), ["safe", "risky", "risky", "safe"]);
    }

    #[test]
    fn uniform_sampling_is_reproducible_and_in_domain() {
        let schema = mixed_schema();
        let first = sample_uniform(&schema, 512, 42, 3).unwrap();
        let repeated = sample_uniform(&schema, 512, 42, 3).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.id().get(), (u128::from(42_u64) << 64) | 3);
        assert_eq!(first.seed(), 0x5027_a070_f5ed_0cd9);
        assert!(
            quantities(&first, 0)
                .iter()
                .all(|value| (-1.0..=1.0).contains(value))
        );
        assert!(
            integers(&first, 1)
                .iter()
                .all(|value| (-2..=2).contains(value))
        );
        assert_eq!(variants(&first, "mode").len(), 512);
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

        let mut continuous = quantities(&first, 0);
        continuous.sort_by(f64::total_cmp);
        for (stratum, value) in continuous.into_iter().enumerate() {
            let stratum = exact_u64_as_f64(u64::try_from(stratum).unwrap());
            let row_count = exact_u64_as_f64(10);
            let lower = -1.0 + 2.0 * stratum / row_count;
            let upper = -1.0 + 2.0 * (stratum + 1.0) / row_count;
            assert!((lower..=upper).contains(&value));
        }

        let mut integers = integers(&first, 1);
        integers.sort_unstable();
        assert_eq!(integers, [-2, -2, -1, -1, 0, 0, 1, 1, 2, 2]);

        let safe_count = variants(&first, "mode")
            .iter()
            .filter(|variant| *variant == "safe")
            .count();
        assert_eq!(safe_count, 5);
    }

    #[test]
    fn booleans_are_sampled_as_two_balanced_levels() {
        let schema = ModelSchemaBuilder::new()
            .boolean("enabled")
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        let request = sample_latin_hypercube(&schema, 10, 3, 0).unwrap();
        let InputColumn::Bool(values) = &request.inputs().columns()[0] else {
            panic!("wrong sampled column kind");
        };
        assert_eq!(values.iter().filter(|value| **value).count(), 5);

        let uniform = sample_uniform(&schema, 64, 3, 0).unwrap();
        let InputColumn::Bool(values) = &uniform.inputs().columns()[0] else {
            panic!("wrong sampled column kind");
        };
        assert!(values.contains(&true) && values.contains(&false));
    }

    #[test]
    fn latin_hypercube_supports_the_complete_i64_domain() {
        let schema = ModelSchemaBuilder::new()
            .integer("value", i64::MIN, i64::MAX)
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        let request = sample_latin_hypercube(&schema, 5, 7, 0).unwrap();
        let values = integers(&request, 0);
        assert_eq!(values.len(), 5);
        assert!(values.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn singleton_domains_are_supported() {
        let schema = ModelSchemaBuilder::new()
            .continuous("x", 0.5, 0.5)
            .unwrap()
            .integer("count", 7, 7)
            .unwrap()
            .categorical("mode", "Mode", ["only"])
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        let request = sample_uniform(&schema, 4, 1, 0).unwrap();
        let lhs_request = sample_latin_hypercube(&schema, 4, 1, 1).unwrap();
        assert_eq!(quantities(&request, 0), [0.5; 4]);
        assert_eq!(quantities(&lhs_request, 0), [0.5; 4]);
        assert_eq!(integers(&request, 1), [7; 4]);
        assert_eq!(variants(&request, "mode"), ["only"; 4]);
    }

    #[test]
    fn half_open_and_overflowing_domains_are_rejected() {
        let lower_only = Bounds::new(Some(FiniteF64::new(0.0).unwrap()), None).unwrap();
        let schema = ModelSchemaBuilder::new()
            .quantity(
                "x",
                QuantityType::new(Dimension::dimensionless(), None).unwrap(),
                lower_only,
            )
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        let unbounded = Err(SamplingError::UnboundedInput {
            name: PortName::new("x").unwrap(),
        });
        assert_eq!(sample_uniform(&schema, 4, 1, 0), unbounded);
        assert_eq!(sample_latin_hypercube(&schema, 4, 1, 0), unbounded);

        let wide = ModelSchemaBuilder::new()
            .continuous("x", -f64::MAX, f64::MAX)
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        let unsampleable = Err(SamplingError::UnsampleableDomain {
            name: PortName::new("x").unwrap(),
        });
        assert_eq!(sample_uniform(&wide, 4, 1, 0), unsampleable);
        assert_eq!(sample_latin_hypercube(&wide, 4, 1, 0), unsampleable);
    }

    fn datetime_schema() -> Arc<SessionSchema> {
        let finite = |value| FiniteF64::new(value).unwrap();
        ModelSchemaBuilder::new()
            .datetime(
                "launch",
                graphcal_model_protocol::DatetimeType::new(graphcal_model_protocol::TimeScale::Tai),
                Bounds::new(Some(finite(1.0e8)), Some(finite(1.0e8 + 86_400.0))).unwrap(),
            )
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap()
    }

    fn datetimes(request: &EvalRequest) -> Vec<f64> {
        let InputColumn::Datetime(values) = &request.inputs().columns()[0] else {
            panic!("wrong sampled column kind");
        };
        values.iter().map(|value| value.get()).collect()
    }

    #[test]
    fn datetimes_are_sampled_as_continuous_seconds_since_j2000() {
        let schema = datetime_schema();
        let in_domain = |values: &[f64]| {
            values
                .iter()
                .all(|value| (1.0e8..=1.0e8 + 86_400.0).contains(value))
        };
        let uniform = sample_uniform(&schema, 64, 7, 0).unwrap();
        assert!(in_domain(&datetimes(&uniform)));
        let lhs = sample_latin_hypercube(&schema, 8, 7, 0).unwrap();
        let mut strata = datetimes(&lhs)
            .iter()
            .map(|value| ((value - 1.0e8) / 86_400.0 * 8.0).floor())
            .collect::<Vec<_>>();
        strata.sort_by(f64::total_cmp);
        assert_eq!(strata, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn complex_indexed_and_algebraic_inputs_are_rejected() {
        let complex = graphcal_model_protocol::WireType::Complex(
            QuantityType::new(Dimension::dimensionless(), None).unwrap(),
        );
        let interface = graphcal_model_protocol::ModelInterface::new(
            vec![
                graphcal_model_protocol::ParameterDescription::new(
                    PortName::new("z").unwrap(),
                    "Complex",
                    graphcal_model_protocol::PortWire::Supported(complex),
                    None,
                    false,
                )
                .unwrap(),
            ],
            vec![graphcal_model_protocol::OutputDescription::new(
                PortName::new("failure").unwrap(),
                "Bool",
                graphcal_model_protocol::PortWire::Supported(
                    graphcal_model_protocol::WireType::Bool,
                ),
            )],
        )
        .unwrap();
        let schema = Arc::new(
            interface
                .configure(
                    &graphcal_model_protocol::SessionSpec::new(
                        vec![PortName::new("z").unwrap()],
                        vec![PortName::new("failure").unwrap()],
                    ),
                    graphcal_model_protocol::CanonicalUnitsOnly,
                )
                .unwrap(),
        );
        for result in [
            sample_uniform(&schema, 4, 1, 0),
            sample_latin_hypercube(&schema, 4, 1, 0),
        ] {
            assert!(matches!(
                result,
                Err(SamplingError::UnsupportedInput {
                    kind: "complex",
                    ..
                })
            ));
        }
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
