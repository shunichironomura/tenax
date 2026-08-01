//! Exercises Phase D Latin-hypercube sampling through real stdio IPC children.

use std::collections::BTreeSet;
use std::convert::Infallible;
use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::io;
use std::process::Command;

use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use tenax::arrow::{
    ChunkResultRef, RecordBatch, evaluation_result_schema, validate_evaluation_request_schema,
};
use tenax::stdio::stdio_discovery_schema;
use tenax::{
    EvalRequest, EvaluationId, Evaluator, Feature, InProcessEvaluator, InputChunk, InputRow,
    InputSchema, InputValue, ModelError, ModelSchema, Objective, OutputSchema, OutputValue, Prim,
    PrimConfig, Restriction, RowContext, RowFailure, RowOutcome, StdioEvaluator,
    evaluation_to_dataset, sample_latin_hypercube, serve_stdio,
};

const CHILD_MODE_ENV: &str = "TENAX_STDIO_TEST_CHILD";
const CHILD_MODE_TOKEN: &str = "tenax-stdio-workflow-child-v1";
const REVERSE_MODE_ENV: &str = "TENAX_STDIO_TEST_REVERSE_TWO";

#[derive(Clone, Copy)]
enum ChildMode {
    Ordered,
    ReverseFirstTwo,
}

fn main() -> Result<(), Box<dyn Error>> {
    if env::var_os(CHILD_MODE_ENV).as_deref() == Some(OsStr::new(CHILD_MODE_TOKEN)) {
        let mode = match env::var_os(REVERSE_MODE_ENV) {
            Some(_) => ChildMode::ReverseFirstTwo,
            None => ChildMode::Ordered,
        };
        return run_child(mode);
    }

    persistent_stdio_arrow_ipc_preserves_failures_and_supports_lhs_prim_workflow()?;
    stdio_results_may_arrive_out_of_request_order()?;
    Ok(())
}

fn model_schema() -> Result<ModelSchema, Box<dyn Error>> {
    Ok(ModelSchema::new(
        vec![
            InputSchema::continuous("load", 0.0, 1.0)?.with_unit("MW")?,
            InputSchema::integer("count", -2, 2)?,
            InputSchema::categorical("mode", vec!["safe".to_owned(), "risky".to_owned()])?,
        ],
        vec![OutputSchema::boolean("failure")?],
    )?)
}

fn run_child(mode: ChildMode) -> Result<(), Box<dyn Error>> {
    let schema = model_schema()?;
    let load = schema.input_position("load")?;
    let mode_position = schema.input_position("mode")?;
    let evaluator =
        InProcessEvaluator::new(schema, move |row: InputRow<'_>, context: RowContext| {
            if context.request_seed() == u64::MAX && context.row_index() == 1 {
                return Err(ModelError::new("intentional conformance-row failure"));
            }
            let InputValue::Continuous(load) = row
                .value(load)
                .map_err(|error| ModelError::new(error.to_string()))?
            else {
                return Err(ModelError::new("load must be continuous"));
            };
            let InputValue::Categorical(mode) = row
                .value(mode_position)
                .map_err(|error| ModelError::new(error.to_string()))?
            else {
                return Err(ModelError::new("mode must be categorical"));
            };
            Ok(vec![OutputValue::Boolean(load >= 0.7 && mode == "risky")])
        });

    match mode {
        ChildMode::Ordered => {
            serve_stdio(&evaluator)?;
            Ok(())
        }
        ChildMode::ReverseFirstTwo => serve_first_two_reversed(&evaluator),
    }
}

fn serve_first_two_reversed<E>(evaluator: &E) -> Result<(), Box<dyn Error>>
where
    E: Evaluator<Error = Infallible>,
{
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let discovery_schema = stdio_discovery_schema(evaluator.schema())?;
    {
        let mut discovery = StreamWriter::try_new(&mut stdout, &discovery_schema)?;
        discovery.finish()?;
    }
    let result_schema = evaluation_result_schema(evaluator.schema());
    let mut result_writer = StreamWriter::try_new(&mut stdout, &result_schema)?;
    result_writer.flush()?;

    let stdin = io::stdin();
    let mut request_reader = StreamReader::try_new(stdin.lock(), None)?;
    validate_evaluation_request_schema(evaluator.schema(), request_reader.schema().as_ref())?;
    let batches = request_reader
        .by_ref()
        .take(2)
        .collect::<Result<Vec<_>, _>>()?;
    if batches.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "reverse-order test server requires two request batches",
        )
        .into());
    }
    let requests = batches
        .iter()
        .map(|batch| EvalRequest::try_from((evaluator.schema(), batch)))
        .collect::<Result<Vec<_>, _>>()?;
    let results = evaluator
        .evaluate(requests)
        .collect::<Result<Vec<_>, _>>()?;
    for result in results.into_iter().rev() {
        let batch = RecordBatch::try_from(ChunkResultRef::new(evaluator.schema(), &result))?;
        result_writer.write(&batch)?;
        result_writer.flush()?;
    }

    match request_reader.next() {
        None => {}
        Some(Ok(_)) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "reverse-order test server received an unexpected third request",
            )
            .into());
        }
        Some(Err(error)) => return Err(error.into()),
    }
    drop(request_reader);
    result_writer.finish()?;
    Ok(())
}

fn spawn_evaluator(reverse_first_two: bool) -> Result<StdioEvaluator, Box<dyn Error>> {
    let mut command = Command::new(env::current_exe()?);
    command.env(CHILD_MODE_ENV, CHILD_MODE_TOKEN);
    command.env_remove(REVERSE_MODE_ENV);
    if reverse_first_two {
        command.env(REVERSE_MODE_ENV, "1");
    }
    Ok(StdioEvaluator::spawn(command)?)
}

fn one_row_request(
    schema: &ModelSchema,
    id: u128,
    load: f64,
    mode: &str,
) -> Result<EvalRequest, Box<dyn Error>> {
    let inputs = InputChunk::new(
        schema,
        vec![
            Feature::continuous("load", vec![load])?,
            Feature::integer("count", vec![0])?,
            Feature::categorical("mode", vec![mode.to_owned()])?,
        ],
    )?;
    Ok(EvalRequest::new(EvaluationId::new(id), 42, inputs))
}

fn persistent_stdio_arrow_ipc_preserves_failures_and_supports_lhs_prim_workflow()
-> Result<(), Box<dyn Error>> {
    let evaluator = spawn_evaluator(false)?;
    let schema = evaluator.schema().clone();
    assert_eq!(schema.inputs()[0].unit(), Some("MW"));

    let explicit_inputs = InputChunk::new(
        &schema,
        vec![
            Feature::continuous("load", vec![0.2, 0.8, 0.9])?,
            Feature::integer("count", vec![0, 1, 2])?,
            Feature::categorical(
                "mode",
                vec!["safe".to_owned(), "risky".to_owned(), "safe".to_owned()],
            )?,
        ],
    )?;
    let explicit = EvalRequest::new(EvaluationId::new(10), u64::MAX, explicit_inputs);
    let explicit_result = evaluator
        .evaluate(vec![explicit])
        .next()
        .ok_or_else(|| io::Error::other("one request produced no result"))??;
    assert!(matches!(explicit_result.rows()[0], RowOutcome::Success(_)));
    assert!(matches!(
        &explicit_result.rows()[1],
        RowOutcome::Failure(RowFailure::Model(error))
            if error.message() == "intentional conformance-row failure"
    ));
    assert!(matches!(explicit_result.rows()[2], RowOutcome::Success(_)));

    // A second call uses the same persistent child, proving that schema
    // discovery and model initialization are not repeated per request batch.
    let request = sample_latin_hypercube(&schema, 5_000, 0x5eed, 1)?;
    let retained = request.clone();
    let result = evaluator
        .evaluate(vec![request])
        .next()
        .ok_or_else(|| io::Error::other("one request produced no result"))??;
    assert!(
        result
            .rows()
            .iter()
            .all(|outcome| matches!(outcome, RowOutcome::Success(_)))
    );

    let failure = schema.output_position("failure")?;
    let dataset = evaluation_to_dataset(&schema, retained, result, failure)?;
    let config = PrimConfig::new(0.05, 0.05, 0.05, Objective::Lenient1)?;
    let discovered = Prim::new(&dataset, config)
        .find_box()
        .ok_or_else(|| io::Error::other("the sampled failure region was not discovered"))?;
    let final_step = discovered.final_step();
    assert!(final_step.statistics().density() >= 0.99);
    assert!(final_step.statistics().coverage() >= 0.9);

    let Some(Restriction::Continuous(load)) = final_step.limits().get("load") else {
        return Err(io::Error::other("PRIM did not return continuous load limits").into());
    };
    assert!((0.68..=0.72).contains(&load.lower()));
    let Some(Restriction::Categorical(mode)) = final_step.limits().get("mode") else {
        return Err(io::Error::other("PRIM did not return categorical mode limits").into());
    };
    assert_eq!(mode.values(), &BTreeSet::from(["risky".to_owned()]));

    let status = evaluator.shutdown()?;
    assert!(status.success());
    Ok(())
}

fn stdio_results_may_arrive_out_of_request_order() -> Result<(), Box<dyn Error>> {
    let evaluator = spawn_evaluator(true)?;
    let first = one_row_request(evaluator.schema(), 21, 0.2, "safe")?;
    let second = one_row_request(evaluator.schema(), 22, 0.8, "risky")?;

    let results = evaluator
        .evaluate(vec![first, second])
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        results
            .iter()
            .map(tenax::ChunkResult::id)
            .collect::<Vec<_>>(),
        [EvaluationId::new(22), EvaluationId::new(21)]
    );

    let status = evaluator.shutdown()?;
    assert!(status.success());
    Ok(())
}
