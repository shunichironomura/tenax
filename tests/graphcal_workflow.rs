//! Exercises the complete seeded Latin-hypercube sample → Graphcal evaluate →
//! PRIM workflow against a real `graphcal model serve` process.
//!
//! The test needs a `graphcal` executable. Run it explicitly with
//! `GRAPHCAL_BIN=/path/to/graphcal cargo test --features graphcal -- --ignored`.

use std::collections::BTreeSet;
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use tenax::graphcal_model_client::{Bool, GraphcalCommand, Key, Model, Quantity};
use tenax::{
    CandidateSelection, EvaluationTable, Evaluator, EvaluatorPool, GraphcalEvaluator,
    MinimumDensity, Objective, Prim, PrimConfig, Restriction, RowOutcome, SelectionObjective,
    sample_latin_hypercube,
};

const MODEL: &str = "pub index Mode = { Stable, Fragile };\n\
    param load: Length(min: 0.0 m, max: 10.0 m);\n\
    param mode: Key<Mode>;\n\
    pub node failure: Bool = @load > 7.0 m && @mode == Mode#Fragile;\n";

#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test helper fails its test on setup errors"
)]
fn describe(name: &str) -> Model {
    let graphcal_bin = std::env::var_os("GRAPHCAL_BIN")
        .map(PathBuf::from)
        .expect("set GRAPHCAL_BIN to the path of a graphcal executable");
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    fs::create_dir_all(&directory).unwrap();
    let file = directory.join("model.gcl");
    fs::write(&file, MODEL).unwrap();
    Model::describe(
        GraphcalCommand::new(&file)
            .program(graphcal_bin)
            .root(&directory),
    )
    .unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "a test helper fails its test on setup errors"
)]
fn spawn(model: &Model) -> GraphcalEvaluator {
    let load = model.input::<Quantity>("load").unwrap();
    let mode = model.input::<Key>("mode").unwrap();
    let failure = model.output::<Bool>("failure").unwrap();
    let session = model
        .session()
        .vary(&load)
        .vary(&mode)
        .output(&failure)
        .spawn()
        .unwrap();
    GraphcalEvaluator::new(session)
}

#[test]
#[ignore = "requires GRAPHCAL_BIN"]
fn latin_hypercube_through_graphcal_recovers_the_known_failure_region() {
    let evaluator = spawn(&describe("graphcal_workflow"));
    let schema = evaluator.schema().clone();
    let failure_slot = schema.output_slot("failure").unwrap();

    let request = sample_latin_hypercube(&schema, 4_000, 0x5eed, 0).unwrap();
    let retained_request = request.clone();
    let result = evaluator.evaluate(vec![request]).next().unwrap().unwrap();
    assert!(
        result
            .rows()
            .iter()
            .all(|row| matches!(row, RowOutcome::Success(_)))
    );
    evaluator.shutdown().unwrap();

    let mut table = EvaluationTable::new(schema);
    table.push(retained_request, result).unwrap();
    #[cfg(feature = "arrow")]
    assert_eq!(stream_round_trip(&table).unwrap(), table);
    let dataset = table.dataset(failure_slot).unwrap();
    let config = PrimConfig::new(0.05, 0.05, 0.05, Objective::Lenient1).unwrap();
    let discovered = Prim::new(&dataset, config).find_box().unwrap();
    let final_step = discovered.final_step();

    assert!(final_step.statistics().density() >= 0.99);
    assert!(final_step.statistics().coverage() >= 0.9);

    let Some(Restriction::Continuous(load)) = final_step.limits().get("load") else {
        panic!("PRIM did not return continuous load limits");
    };
    assert!((6.9..=7.2).contains(&load.lower()));

    let Some(Restriction::Categorical(mode)) = final_step.limits().get("mode") else {
        panic!("PRIM did not return categorical mode limits");
    };
    assert_eq!(mode.values(), &BTreeSet::from(["Fragile".to_owned()]));
}

#[test]
#[ignore = "requires GRAPHCAL_BIN"]
fn pooled_sessions_evaluate_a_split_design_like_one_session() {
    let model = describe("graphcal_pool");
    let single = spawn(&model);
    let schema = single.schema().clone();
    let design = sample_latin_hypercube(&schema, 1_000, 0x5eed, 0).unwrap();
    let result = single
        .evaluate(vec![design.clone()])
        .next()
        .unwrap()
        .unwrap();
    single.shutdown().unwrap();
    let mut whole = EvaluationTable::new(schema.clone());
    whole.push(design.clone(), result).unwrap();

    let pool = EvaluatorPool::new((0..3).map(|_| spawn(&model)).collect()).unwrap();
    let chunks = design.split(NonZeroUsize::new(64).unwrap()).unwrap();
    let mut pooled = EvaluationTable::new(schema.clone());
    for result in pool.evaluate(chunks.clone()) {
        let result = result.unwrap();
        let chunk = chunks
            .iter()
            .find(|chunk| chunk.id() == result.id())
            .unwrap();
        pooled.push(chunk.clone(), result).unwrap();
    }
    pooled.sort_by_id();
    for evaluator in pool.into_evaluators() {
        evaluator.shutdown().unwrap();
    }
    assert_eq!(
        pooled.outcomes().collect::<Vec<_>>(),
        whole.outcomes().collect::<Vec<_>>()
    );

    let failure = schema.output_slot("failure").unwrap();
    let dataset = pooled.dataset(failure).unwrap();
    let found = Prim::new(&dataset, PrimConfig::default())
        .find_box()
        .unwrap();
    let position = found
        .select_candidate(
            CandidateSelection::new(
                2,
                SelectionObjective::HighDensity(MinimumDensity::new(0.9).unwrap()),
            )
            .restricting_at_least_one(),
        )
        .unwrap();
    let effects = found.restriction_effects(&dataset, position).unwrap();
    assert_eq!(
        effects
            .iter()
            .map(|effect| effect.limit().name())
            .collect::<Vec<_>>(),
        ["load", "mode"]
    );
    assert!(effects.iter().all(|effect| effect.density_gain() > 0.2));
}

/// Stores a Graphcal session's evaluations and restores them, session
/// included, without the model.
#[cfg(feature = "arrow")]
fn stream_round_trip(
    table: &EvaluationTable,
) -> Result<EvaluationTable, Box<dyn std::error::Error>> {
    let mut writer = tenax::EvaluationWriter::try_new(Vec::new(), table.schema())?;
    writer.write_table(table)?;
    let bytes = writer.finish()?;
    Ok(tenax::EvaluationReader::try_new(bytes.as_slice())?.read_table()?)
}
