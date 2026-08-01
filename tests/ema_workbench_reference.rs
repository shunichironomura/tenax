//! Pins tenax's PRIM trajectories to EMA Workbench 3.0.0.
//!
//! The fixture is regenerated with `scripts/generate_ema_reference.py`.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "the whole crate is test code, but `allow-unwrap-in-tests` only covers `#[test]` functions"
)]

use std::collections::BTreeSet;

use serde::Deserialize;
use tenax::{
    BoxLimits, BoxStatistics, Dataset, Feature, Objective, Prim, PrimConfig, PrimPhase,
    QuasiPValue, Restriction,
};

const FIXTURE: &str = include_str!("fixtures/ema_workbench_3_0_0.json");
const TOLERANCE: f64 = 2e-12;

#[derive(Deserialize)]
struct ReferenceSuite {
    reference: ReferenceMetadata,
    cases: Vec<ReferenceCase>,
}

#[derive(Deserialize)]
struct ReferenceMetadata {
    implementation: String,
    package: String,
    version: String,
}

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    config: ReferenceConfig,
    features: Vec<ReferenceFeature>,
    target: Vec<u8>,
    trajectory: Vec<ReferenceStep>,
}

#[derive(Deserialize)]
struct ReferenceConfig {
    objective: String,
    peel_alpha: f64,
    paste_alpha: f64,
    mass_min: f64,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum ReferenceFeature {
    #[serde(rename = "continuous")]
    Continuous { name: String, values: Vec<f64> },
    #[serde(rename = "integer")]
    Integer { name: String, values: Vec<i64> },
    #[serde(rename = "categorical")]
    Categorical { name: String, values: Vec<String> },
}

#[derive(Deserialize)]
struct ReferenceStep {
    stats: ReferenceStats,
    limits: Vec<ReferenceLimit>,
    indices: Vec<usize>,
    quasi_p_values: Vec<ReferencePValue>,
}

#[derive(Deserialize)]
struct ReferenceStats {
    coverage: f64,
    density: f64,
    mean: f64,
    mass: f64,
    restricted_dimensions: usize,
    points: usize,
    cases_of_interest: usize,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum ReferenceLimit {
    #[serde(rename = "continuous")]
    Continuous {
        name: String,
        lower: f64,
        upper: f64,
    },
    #[serde(rename = "integer")]
    Integer {
        name: String,
        lower: i64,
        upper: i64,
    },
    #[serde(rename = "categorical")]
    Categorical {
        name: String,
        categories: BTreeSet<String>,
    },
}

#[derive(Deserialize)]
struct ReferencePValue {
    name: String,
    lower: Option<f64>,
    upper: Option<f64>,
}

#[test]
fn trajectories_match_ema_workbench_3_0_0() {
    let suite: ReferenceSuite = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(suite.reference.implementation, "EMA Workbench");
    assert_eq!(suite.reference.package, "ema-workbench");
    assert_eq!(suite.reference.version, "3.0.0");

    for reference in suite.cases {
        compare_case(reference);
    }
}

fn build_dataset(features: Vec<ReferenceFeature>, target: Vec<u8>) -> Dataset {
    let features = features
        .into_iter()
        .map(|feature| match feature {
            ReferenceFeature::Continuous { name, values } => Feature::continuous(name, values),
            ReferenceFeature::Integer { name, values } => Feature::integer(name, values),
            ReferenceFeature::Categorical { name, values } => Feature::categorical(name, values),
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let target = target.into_iter().map(|value| value == 1).collect();
    Dataset::new(features, target).unwrap()
}

fn build_config(config: &ReferenceConfig) -> PrimConfig {
    let objective = match config.objective.as_str() {
        "lenient1" => Objective::Lenient1,
        "lenient2" => Objective::Lenient2,
        "original" => Objective::Original,
        other => panic!("unknown fixture objective {other}"),
    };
    PrimConfig::new(
        config.peel_alpha,
        config.paste_alpha,
        config.mass_min,
        objective,
    )
    .unwrap()
}

fn compare_case(reference: ReferenceCase) {
    let dataset = build_dataset(reference.features, reference.target);
    let config = build_config(&reference.config);
    let actual = Prim::new(&dataset, config).find_box().unwrap();
    let expected_pastes = reference
        .trajectory
        .windows(2)
        .filter(|steps| steps[1].stats.points > steps[0].stats.points)
        .count();
    if reference.name == "continuous_with_pasting" {
        assert_eq!(expected_pastes, 3, "reference case must exercise pasting");
    }

    assert_eq!(
        actual.trajectory().len(),
        reference.trajectory.len(),
        "{} trajectory length",
        reference.name
    );
    let mut previous_points = None;
    for (position, (actual, expected)) in actual
        .trajectory()
        .iter()
        .zip(reference.trajectory)
        .enumerate()
    {
        let context = format!("{} trajectory step {position}", reference.name);
        let expected_phase = match previous_points {
            None => PrimPhase::Initial,
            Some(points) if expected.stats.points < points => PrimPhase::Peel,
            Some(_) => PrimPhase::Paste,
        };
        assert_eq!(actual.phase(), expected_phase, "{context}: phase");
        previous_points = Some(expected.stats.points);
        compare_statistics(actual.statistics(), &expected.stats, &context);
        assert_eq!(actual.indices(), expected.indices, "{context}: row indices");
        compare_limits(actual.limits(), expected.limits, &context);
        compare_quasi_p_values(actual.quasi_p_values(), expected.quasi_p_values, &context);
    }
}

fn compare_statistics(actual: BoxStatistics, expected: &ReferenceStats, context: &str) {
    assert_close(actual.coverage(), expected.coverage, context);
    assert_close(actual.density(), expected.density, context);
    assert_close(actual.mean(), expected.mean, context);
    assert_close(actual.mass(), expected.mass, context);
    assert_eq!(
        actual.restricted_dimensions(),
        expected.restricted_dimensions,
        "{context}: restricted dimensions"
    );
    assert_eq!(actual.points(), expected.points, "{context}: points");
    assert_eq!(
        actual.cases_of_interest(),
        expected.cases_of_interest,
        "{context}: cases of interest"
    );
}

fn compare_limits(actual: &BoxLimits, expected: Vec<ReferenceLimit>, context: &str) {
    assert_eq!(
        actual.limits().len(),
        expected.len(),
        "{context}: limit count"
    );
    for (actual_limit, expected_limit) in actual.limits().iter().zip(expected) {
        match (actual_limit.restriction(), expected_limit) {
            (
                Restriction::Continuous(actual),
                ReferenceLimit::Continuous { name, lower, upper },
            ) => {
                assert_eq!(actual_limit.name(), name, "{context}: feature name");
                assert_close(actual.lower(), lower, context);
                assert_close(actual.upper(), upper, context);
            }
            (Restriction::Integer(actual), ReferenceLimit::Integer { name, lower, upper }) => {
                assert_eq!(actual_limit.name(), name, "{context}: feature name");
                assert_eq!(actual.lower(), lower, "{context}: lower limit");
                assert_eq!(actual.upper(), upper, "{context}: upper limit");
            }
            (
                Restriction::Categorical(actual),
                ReferenceLimit::Categorical { name, categories },
            ) => {
                assert_eq!(actual_limit.name(), name, "{context}: feature name");
                assert_eq!(actual.values(), &categories, "{context}: categories");
            }
            _ => panic!("{context}: limit type differs from EMA Workbench"),
        }
    }
}

fn compare_quasi_p_values(actual: &[QuasiPValue], expected: Vec<ReferencePValue>, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: quasi-p value count"
    );
    for (actual_p, expected_p) in actual.iter().zip(expected) {
        assert_eq!(actual_p.name(), expected_p.name, "{context}: p-value name");
        assert_optional_close(actual_p.lower(), expected_p.lower, context);
        assert_optional_close(actual_p.upper(), expected_p.upper, context);
    }
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{context}: expected {expected:.17}, got {actual:.17}"
    );
}

fn assert_optional_close(actual: Option<f64>, expected: Option<f64>, context: &str) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_close(actual, expected, context),
        (None, None) => {}
        values => panic!("{context}: optional values differ: {values:?}"),
    }
}
