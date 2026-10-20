//! Exercises the seeded sample → evaluate → stream to disk → read back → PRIM
//! workflow, and reuses one stored evaluation for two analyses.

#![cfg(feature = "arrow")]

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;

use tenax::graphcal_model_protocol::{FiniteF64, QuantityType, RowFailure, Value};
use tenax::{
    CaseRule, ChunkingPolicy, Comparison, EvaluatedRequest, EvaluationReader, EvaluationWriter,
    Evaluator, InputRow, ModelSchemaBuilder, Objective, ParallelInProcessEvaluator, Prim,
    PrimConfig, Restriction, RowContext, sample_latin_hypercube,
};

#[test]
fn stored_evaluations_feed_prim_after_reading_back() {
    let dimensionless = QuantityType::new(
        tenax::graphcal_model_protocol::Dimension::dimensionless(),
        None,
    )
    .unwrap();
    let schema = ModelSchemaBuilder::new()
        .continuous("load", 0.0, 1.0)
        .unwrap()
        .categorical("regime", "Regime", ["stable", "fragile"])
        .unwrap()
        .quantity_output("stress", dimensionless)
        .unwrap()
        .boolean_output("failure")
        .unwrap()
        .build()
        .unwrap();
    let load = schema.input_slot("load").unwrap();
    let evaluator = ParallelInProcessEvaluator::new(
        Arc::clone(&schema),
        move |row: InputRow<'_>, _context: RowContext| -> Result<Vec<Value>, RowFailure> {
            let load = row.quantity(load)?;
            Ok(vec![
                Value::Quantity(FiniteF64::new(load * 2.0).unwrap()),
                Value::Bool(load >= 0.7),
            ])
        },
        ChunkingPolicy::new(64).unwrap(),
    );

    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("evaluation_stream");
    fs::create_dir_all(&directory).unwrap();
    let path = directory.join("evaluations.arrows");
    let mut writer =
        EvaluationWriter::try_new(BufWriter::new(File::create(&path).unwrap()), &schema).unwrap();
    for sequence in 0..3 {
        let request = sample_latin_hypercube(&schema, 500, 0x5eed, sequence).unwrap();
        let retained = request.clone();
        let result = evaluator.evaluate(vec![request]).next().unwrap().unwrap();
        writer
            .write(&EvaluatedRequest::new(&schema, retained, result).unwrap())
            .unwrap();
    }
    writer.finish().unwrap();

    let table = EvaluationReader::try_new(File::open(&path).unwrap())
        .unwrap()
        .read_table()
        .unwrap();
    assert_eq!(table.row_count(), 1_500);
    assert_eq!(table.schema().as_ref(), schema.as_ref());

    let failure = table.schema().output_slot("failure").unwrap();
    let dataset = table.dataset(failure).unwrap();
    let config = PrimConfig::new(0.05, 0.05, 0.05, Objective::Lenient1).unwrap();
    let discovered = Prim::new(&dataset, config).find_box().unwrap();
    assert!(discovered.final_step().statistics().density() >= 0.99);

    // The same stored evaluation answers a different question without
    // re-evaluating the model.
    let stress = table.schema().output_slot("stress").unwrap();
    let high_stress = CaseRule::Quantity {
        comparison: Comparison::AtLeast,
        threshold: FiniteF64::new(1.8).unwrap(),
    };
    let dataset = table.dataset_where(stress, &high_stress).unwrap();
    let discovered = Prim::new(&dataset, config).find_box().unwrap();
    let Some(Restriction::Continuous(load)) = discovered.final_step().limits().get("load") else {
        panic!("PRIM did not return continuous load limits");
    };
    assert!((0.88..=0.92).contains(&load.lower()));
}
