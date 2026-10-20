//! Exercises the complete seeded Latin-hypercube sample → Graphcal evaluate →
//! PRIM workflow against a real `graphcal model serve` process.
//!
//! The test needs a `graphcal` executable. Run it explicitly with
//! `GRAPHCAL_BIN=/path/to/graphcal cargo test --features graphcal -- --ignored`.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use tenax::graphcal_model_client::{Bool, GraphcalCommand, Key, Model, Quantity};
use tenax::{
    EvaluationTable, Evaluator, GraphcalEvaluator, Objective, Prim, PrimConfig, Restriction,
    RowOutcome, sample_latin_hypercube,
};

const MODEL: &str = "pub index Mode = { Stable, Fragile };\n\
    param load: Length(min: 0.0 m, max: 10.0 m);\n\
    param mode: Key<Mode>;\n\
    pub node failure: Bool = @load > 7.0 m && @mode == Mode#Fragile;\n";

#[test]
#[ignore = "requires GRAPHCAL_BIN"]
fn latin_hypercube_through_graphcal_recovers_the_known_failure_region() {
    let graphcal_bin = std::env::var_os("GRAPHCAL_BIN")
        .map(PathBuf::from)
        .expect("set GRAPHCAL_BIN to the path of a graphcal executable");
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("graphcal_workflow");
    fs::create_dir_all(&directory).unwrap();
    let file = directory.join("model.gcl");
    fs::write(&file, MODEL).unwrap();

    let model = Model::describe(
        GraphcalCommand::new(&file)
            .program(graphcal_bin)
            .root(&directory),
    )
    .unwrap();
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
    let evaluator = GraphcalEvaluator::new(session);
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
