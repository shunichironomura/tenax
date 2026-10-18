//! Exercises the complete seeded sample → parallel evaluate → PRIM workflow.

use std::collections::BTreeSet;
use std::sync::Arc;

use tenax::graphcal_model_protocol::{InputSlot, KeyRef, RowFailure, SessionSchema, Value};
use tenax::{
    ChunkingPolicy, EvalRequest, Evaluator, InputRow, ModelSchemaBuilder, ModelSchemaError,
    Objective, ParallelInProcessEvaluator, Prim, PrimConfig, Restriction, RowContext,
    evaluation_to_dataset, sample_uniform,
};

fn toy_schema() -> Result<Arc<SessionSchema>, ModelSchemaError> {
    ModelSchemaBuilder::new()
        .continuous("load", 0.0, 1.0)?
        .categorical("regime", "Regime", ["fragile", "stable"])?
        .boolean_output("failure")?
        .build()
}

fn evaluate_toy_model(
    row: InputRow<'_>,
    load: InputSlot,
    regime: InputSlot,
) -> Result<Vec<Value>, RowFailure> {
    let load = row.quantity(load)?;
    let fragile =
        matches!(row.key(regime)?, KeyRef::Variant(variant) if variant.as_str() == "fragile");
    Ok(vec![Value::Bool(load >= 0.7 && fragile)])
}

#[test]
fn sampling_through_in_process_evaluation_recovers_the_known_failure_region() {
    let schema = toy_schema().unwrap();
    let load = schema.input_slot("load").unwrap();
    let regime = schema.input_slot("regime").unwrap();
    let failure = schema.output_slot("failure").unwrap();
    let evaluator = ParallelInProcessEvaluator::new(
        Arc::clone(&schema),
        move |row: InputRow<'_>, _context: RowContext| evaluate_toy_model(row, load, regime),
        ChunkingPolicy::new(64).unwrap(),
    );
    let request = sample_uniform(&schema, 5_000, 0x5eed, 0).unwrap();
    let retained_request: EvalRequest = request.clone();
    let result = evaluator.evaluate(vec![request]).next().unwrap().unwrap();
    let dataset = evaluation_to_dataset(&schema, retained_request, result, failure).unwrap();

    let config = PrimConfig::new(0.05, 0.05, 0.05, Objective::Lenient1).unwrap();
    let discovered = Prim::new(&dataset, config).find_box().unwrap();
    let final_step = discovered.final_step();

    assert!(final_step.statistics().density() >= 0.99);
    assert!(final_step.statistics().coverage() >= 0.9);

    let Some(Restriction::Continuous(load)) = final_step.limits().get("load") else {
        panic!("PRIM did not return continuous load limits");
    };
    assert!((0.69..=0.72).contains(&load.lower()));

    let Some(Restriction::Categorical(regime)) = final_step.limits().get("regime") else {
        panic!("PRIM did not return categorical regime limits");
    };
    assert_eq!(regime.values(), &BTreeSet::from(["fragile".to_owned()]));
}
