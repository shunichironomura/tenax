//! Exercises the complete seeded sample → evaluate → PRIM Phase A workflow.

use std::collections::BTreeSet;

use tenax::{
    EvalRequest, Evaluator, InProcessEvaluator, InputPosition, InputRow, InputSchema, InputValue,
    ModelError, ModelSchema, Objective, OutputSchema, OutputValue, Prim, PrimConfig, Restriction,
    RowContext, SchemaError, evaluation_to_dataset, sample_uniform,
};

fn toy_schema() -> Result<ModelSchema, SchemaError> {
    ModelSchema::new(
        vec![
            InputSchema::continuous("load", 0.0, 1.0)?,
            InputSchema::categorical("regime", vec!["stable".to_owned(), "fragile".to_owned()])?,
        ],
        vec![OutputSchema::boolean("failure")?],
    )
}

fn evaluate_toy_model(
    row: InputRow<'_>,
    load_position: InputPosition,
    regime_position: InputPosition,
) -> Result<Vec<OutputValue>, ModelError> {
    let InputValue::Continuous(load) = row
        .value(load_position)
        .map_err(|error| ModelError::new(error.to_string()))?
    else {
        return Err(ModelError::new("load must be continuous"));
    };
    let InputValue::Categorical(regime) = row
        .value(regime_position)
        .map_err(|error| ModelError::new(error.to_string()))?
    else {
        return Err(ModelError::new("regime must be categorical"));
    };

    Ok(vec![OutputValue::Boolean(
        load >= 0.7 && regime == "fragile",
    )])
}

#[test]
fn sampling_through_in_process_evaluation_recovers_the_known_failure_region() {
    let schema = toy_schema().unwrap();
    let load = schema.input_position("load").unwrap();
    let regime = schema.input_position("regime").unwrap();
    let failure = schema.output_position("failure").unwrap();
    let evaluator = InProcessEvaluator::new(
        schema.clone(),
        move |row: InputRow<'_>, _context: RowContext| evaluate_toy_model(row, load, regime),
    );
    let request = sample_uniform(&schema, 5_000, 0x5eed, 0).unwrap();
    let retained_request: EvalRequest = request.clone();
    let result = evaluator.evaluate(vec![request]).next().unwrap();
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
