//! End-to-end in-process workflow for EMA Workbench's DPS lake problem.
//!
//! Run with `cargo run --release --example lake_model`, then render the
//! exported PRIM trajectory with `./examples/lake_model/plot.py`.

mod model;

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use model::{DpsPolicy, LakeModel, LakeUncertainties};
use tenax::graphcal_model_protocol::{
    Domain, InputSlot, RowFailure, SessionInput, SessionSchema, Value,
};
use tenax::{
    BoxStep, ChunkingPolicy, Dataset, Evaluator, FeatureView, InputRow, ModelSchemaBuilder,
    ParallelInProcessEvaluator, Prim, PrimBox, PrimPhase, Restriction, RowContext,
    evaluation_to_dataset, sample_uniform,
};
use thiserror::Error;

const EXPERIMENT_COUNT: usize = 5_000;
const ROWS_PER_WORK_CHUNK: usize = 16;
const RUN_SEED: u64 = 0x5eed_1a6e;
const REQUEST_SEQUENCE: u64 = 0;
const MAX_PHOSPHORUS_THRESHOLD: f64 = 0.8;
const STOCHASTIC_REALIZATIONS: u32 = 150;
const MODEL_YEARS: u32 = 100;
const UTILITY_FROM_POLLUTION: f64 = 0.41;
const DESIRABLE_OUTPUT: &str = "desirable_lake_state";

#[derive(Clone, Copy, Debug)]
struct LakeInputSlots {
    decay_rate: InputSlot,
    recycling_exponent: InputSlot,
    inflow_mean: InputSlot,
    inflow_standard_deviation: InputSlot,
    discount_rate: InputSlot,
    first_center: InputSlot,
    second_center: InputSlot,
    first_radius: InputSlot,
    second_radius: InputSlot,
    first_weight: InputSlot,
}

impl LakeInputSlots {
    fn resolve(schema: &SessionSchema) -> Result<Self, ExampleError> {
        let slot = |name: &'static str| {
            schema
                .input_slot(name)
                .ok_or(ExampleError::MissingInput { name })
        };
        Ok(Self {
            decay_rate: slot("b")?,
            recycling_exponent: slot("q")?,
            inflow_mean: slot("mean")?,
            inflow_standard_deviation: slot("stdev")?,
            discount_rate: slot("delta")?,
            first_center: slot("c1")?,
            second_center: slot("c2")?,
            first_radius: slot("r1")?,
            second_radius: slot("r2")?,
            first_weight: slot("w1")?,
        })
    }

    fn read(self, row: InputRow<'_>) -> Result<(LakeUncertainties, DpsPolicy), RowFailure> {
        let uncertainties = LakeUncertainties::new(
            row.quantity(self.decay_rate)?,
            row.quantity(self.recycling_exponent)?,
            row.quantity(self.inflow_mean)?,
            row.quantity(self.inflow_standard_deviation)?,
            row.quantity(self.discount_rate)?,
        )
        .map_err(evaluation_error)?;
        let policy = DpsPolicy::new(
            row.quantity(self.first_center)?,
            row.quantity(self.second_center)?,
            row.quantity(self.first_radius)?,
            row.quantity(self.second_radius)?,
            row.quantity(self.first_weight)?,
        )
        .map_err(evaluation_error)?;
        Ok((uncertainties, policy))
    }
}

fn evaluation_error(error: impl std::fmt::Display) -> RowFailure {
    RowFailure::EvaluationError {
        message: error.to_string(),
    }
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error("usage: cargo run --release --example lake_model -- [OUTPUT_DIRECTORY]")]
    TooManyArguments,

    #[error("the in-process evaluator did not return its requested chunk")]
    MissingEvaluationResult,

    #[error("the lake schema has no input named '{name}'")]
    MissingInput { name: &'static str },

    #[error("the lake schema has no output named '{name}'")]
    MissingOutput { name: &'static str },

    #[error("PRIM did not return a first box for the non-empty lake dataset")]
    MissingPrimBox,

    #[error("lake example feature '{name}' unexpectedly has kind {actual:?}")]
    NonContinuousFeature {
        name: String,
        actual: tenax::FeatureKind,
    },

    #[error("lake example input '{name}' unexpectedly lacks a closed quantity domain")]
    NonContinuousDomain { name: String },

    #[error("PRIM limits do not contain lake input '{name}'")]
    MissingFeatureLimit { name: String },

    #[error("PRIM returned non-continuous limits for lake input '{name}'")]
    NonContinuousLimit { name: String },

    #[error("feature '{name}' does not contain exported row {row}")]
    MissingFeatureRow { name: String, row: usize },
}

fn main() -> Result<(), Box<dyn Error>> {
    let output_directory = output_directory(env::args_os())?;
    let schema = lake_schema()?;
    let slots = LakeInputSlots::resolve(&schema)?;
    let desirable = schema
        .output_slot(DESIRABLE_OUTPUT)
        .ok_or(ExampleError::MissingOutput {
            name: DESIRABLE_OUTPUT,
        })?;
    let lake_model = LakeModel::new(UTILITY_FROM_POLLUTION, STOCHASTIC_REALIZATIONS, MODEL_YEARS)?;
    let evaluator = ParallelInProcessEvaluator::new(
        schema.clone(),
        move |row: InputRow<'_>, context: RowContext| {
            let (uncertainties, policy) = slots.read(row)?;
            let outcomes = lake_model
                .evaluate(uncertainties, policy, context.seed())
                .map_err(evaluation_error)?;
            Ok(vec![Value::Bool(
                outcomes.max_phosphorus() < MAX_PHOSPHORUS_THRESHOLD,
            )])
        },
        ChunkingPolicy::new(ROWS_PER_WORK_CHUNK)?,
    );

    let request = sample_uniform(&schema, EXPERIMENT_COUNT, RUN_SEED, REQUEST_SEQUENCE)?;
    let retained_request = request.clone();
    let result = evaluator
        .evaluate(vec![request])
        .next()
        .ok_or(ExampleError::MissingEvaluationResult)??;
    let dataset = evaluation_to_dataset(&schema, retained_request, result, desirable)?;
    let mut prim = Prim::new(&dataset, tenax::PrimConfig::default());
    let first_box = prim.find_box().ok_or(ExampleError::MissingPrimBox)?;

    fs::create_dir_all(&output_directory)?;
    write_experiments(&output_directory.join("experiments.csv"), &dataset)?;
    write_trajectory(&output_directory.join("trajectory.csv"), &first_box)?;
    write_limits(&output_directory.join("limits.csv"), &schema, &first_box)?;
    write_summary(
        &output_directory.join("summary.txt"),
        &schema,
        &dataset,
        &first_box,
    )?;

    let stdout = io::stdout();
    writeln!(
        stdout.lock(),
        "Lake PRIM analysis written to {}",
        output_directory.display()
    )?;
    Ok(())
}

fn lake_schema() -> Result<Arc<SessionSchema>, tenax::ModelSchemaError> {
    ModelSchemaBuilder::new()
        // Deep uncertainties from the EMA Workbench tutorial.
        .continuous("b", 0.1, 0.45)?
        .continuous("q", 2.0, 4.5)?
        .continuous("mean", 0.01, 0.05)?
        .continuous("stdev", 0.001, 0.005)?
        .continuous("delta", 0.93, 0.99)?
        // Direct Policy Search levers from the same tutorial.
        .continuous("c1", -2.0, 2.0)?
        .continuous("c2", -2.0, 2.0)?
        .continuous("r1", 0.0, 2.0)?
        .continuous("r2", 0.0, 2.0)?
        .continuous("w1", 0.0, 1.0)?
        // PRIM needs a binary output, so classification happens at the model
        // boundary rather than in a later Python dataframe operation.
        .boolean_output(DESIRABLE_OUTPUT)?
        .build()
}

/// The closed bounds of a lake input, which are all dimensionless quantities.
fn continuous_domain(input: &SessionInput) -> Result<(f64, f64), ExampleError> {
    match input.domain() {
        Some(Domain::Quantity(bounds)) => match (bounds.min(), bounds.max()) {
            (Some(lower), Some(upper)) => Ok((lower.get(), upper.get())),
            _ => Err(non_continuous_domain(input)),
        },
        _ => Err(non_continuous_domain(input)),
    }
}

fn non_continuous_domain(input: &SessionInput) -> ExampleError {
    ExampleError::NonContinuousDomain {
        name: input.name().to_string(),
    }
}

fn output_directory(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<PathBuf, ExampleError> {
    let _executable = arguments.next();
    let output = arguments
        .next()
        .map_or_else(|| PathBuf::from("target/lake_model"), PathBuf::from);
    if arguments.next().is_some() {
        Err(ExampleError::TooManyArguments)
    } else {
        Ok(output)
    }
}

fn write_experiments(path: &Path, dataset: &Dataset) -> Result<(), Box<dyn Error>> {
    let columns = dataset
        .features()
        .iter()
        .map(|feature| match feature.view() {
            FeatureView::Continuous(values) => Ok((feature.name(), values)),
            values => Err(ExampleError::NonContinuousFeature {
                name: feature.name().to_owned(),
                actual: values.kind(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"row_index")?;
    for (name, _) in &columns {
        writer.write_all(b",")?;
        write_csv_field(&mut writer, name)?;
    }
    writer.write_all(b",case_of_interest\n")?;

    for row in 0..dataset.row_count() {
        write!(writer, "{row}")?;
        for (name, values) in &columns {
            let value = values
                .get(row)
                .ok_or_else(|| ExampleError::MissingFeatureRow {
                    name: (*name).to_owned(),
                    row,
                })?;
            write!(writer, ",{value}")?;
        }
        writeln!(writer, ",{}", dataset.cases_of_interest()[row])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_trajectory(path: &Path, prim_box: &PrimBox) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(
        b"step,phase,coverage,density,mass,restricted_dimensions,points,cases_of_interest\n",
    )?;
    for (step_index, step) in prim_box.trajectory().iter().enumerate() {
        let statistics = step.statistics();
        writeln!(
            writer,
            "{step_index},{},{},{},{},{},{},{}",
            phase_name(step.phase()),
            statistics.coverage(),
            statistics.density(),
            statistics.mass(),
            statistics.restricted_dimensions(),
            statistics.points(),
            statistics.cases_of_interest(),
        )?;
    }
    writer.flush()
}

fn write_limits(
    path: &Path,
    schema: &SessionSchema,
    prim_box: &PrimBox,
) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(
        b"step,feature,domain_lower,domain_upper,box_lower,box_upper,quasi_p_lower,quasi_p_upper\n",
    )?;
    for (step_index, step) in prim_box.trajectory().iter().enumerate() {
        for input in schema.inputs() {
            let name = input.name().as_str();
            let (domain_lower, domain_upper) = continuous_domain(input)?;
            let Some(Restriction::Continuous(range)) = step.limits().get(name) else {
                return Err(match step.limits().get(name) {
                    Some(_) => ExampleError::NonContinuousLimit {
                        name: name.to_owned(),
                    },
                    None => ExampleError::MissingFeatureLimit {
                        name: name.to_owned(),
                    },
                }
                .into());
            };
            let quasi_p = step
                .quasi_p_values()
                .iter()
                .find(|value| value.name() == name);

            write!(writer, "{step_index},")?;
            write_csv_field(&mut writer, name)?;
            write!(
                writer,
                ",{domain_lower},{domain_upper},{},{}",
                range.lower(),
                range.upper()
            )?;
            write_optional_f64(&mut writer, quasi_p.and_then(tenax::QuasiPValue::lower))?;
            write_optional_f64(&mut writer, quasi_p.and_then(tenax::QuasiPValue::upper))?;
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_summary(
    path: &Path,
    schema: &SessionSchema,
    dataset: &Dataset,
    prim_box: &PrimBox,
) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    let final_index = prim_box.trajectory().len() - 1;
    let final_step = prim_box.final_step();
    let statistics = final_step.statistics();
    writeln!(writer, "EMA Workbench DPS lake model / Tenax Phase B")?;
    writeln!(writer, "experiments: {EXPERIMENT_COUNT}")?;
    writeln!(writer, "rows per Rayon work chunk: {ROWS_PER_WORK_CHUNK}")?;
    writeln!(writer, "run seed: {RUN_SEED}")?;
    writeln!(
        writer,
        "stochastic realizations per experiment: {STOCHASTIC_REALIZATIONS}"
    )?;
    writeln!(writer, "model years: {MODEL_YEARS}")?;
    writeln!(writer, "case criterion: max_P < {MAX_PHOSPHORUS_THRESHOLD}")?;
    writeln!(writer, "cases of interest: {}", dataset.case_count())?;
    writeln!(writer, "trajectory steps: {}", prim_box.trajectory().len())?;
    writeln!(writer, "final step: {final_index}")?;
    writeln!(writer, "final coverage: {}", statistics.coverage())?;
    writeln!(writer, "final density: {}", statistics.density())?;
    writeln!(writer, "final mass: {}", statistics.mass())?;
    writeln!(
        writer,
        "final restricted dimensions: {}",
        statistics.restricted_dimensions()
    )?;
    writeln!(writer, "final limits:")?;
    write_restricted_limits(&mut writer, schema, &prim_box.trajectory()[0], final_step)?;
    writer.flush()?;
    Ok(())
}

fn write_restricted_limits(
    writer: &mut impl Write,
    schema: &SessionSchema,
    initial_step: &BoxStep,
    selected_step: &BoxStep,
) -> Result<(), Box<dyn Error>> {
    for input in schema.inputs() {
        let name = input.name().as_str();
        continuous_domain(input)?;
        let Some(Restriction::Continuous(initial_range)) = initial_step.limits().get(name) else {
            return Err(ExampleError::MissingFeatureLimit {
                name: name.to_owned(),
            }
            .into());
        };
        let Some(Restriction::Continuous(selected_range)) = selected_step.limits().get(name) else {
            return Err(ExampleError::MissingFeatureLimit {
                name: name.to_owned(),
            }
            .into());
        };
        if selected_range.lower().to_bits() != initial_range.lower().to_bits()
            || selected_range.upper().to_bits() != initial_range.upper().to_bits()
        {
            writeln!(
                writer,
                "  {name}: [{}, {}]",
                selected_range.lower(),
                selected_range.upper()
            )?;
        }
    }
    Ok(())
}

fn write_optional_f64(writer: &mut impl Write, value: Option<f64>) -> io::Result<()> {
    writer.write_all(b",")?;
    value.map_or(Ok(()), |value| write!(writer, "{value}"))
}

fn write_csv_field(writer: &mut impl Write, value: &str) -> io::Result<()> {
    writer.write_all(b"\"")?;
    writer.write_all(value.replace('"', "\"\"").as_bytes())?;
    writer.write_all(b"\"")
}

const fn phase_name(phase: PrimPhase) -> &'static str {
    match phase {
        PrimPhase::Initial => "initial",
        PrimPhase::Peel => "peel",
        PrimPhase::Paste => "paste",
    }
}
