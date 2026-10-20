//! Tenax provides scenario-discovery algorithms for decision-making under deep
//! uncertainty.
//!
//! The current release implements conventional Patient Rule Induction Method
//! (PRIM) analysis for static binary input/output datasets, plus reproducible
//! uniform and Latin-hypercube samplers, a transport-independent evaluator
//! interface, and sequential and Rayon-parallel in-process workflows. PRIM
//! supports continuous, integer, and categorical inputs and returns the
//! complete peeling and pasting trajectory with coverage, density, mass, and
//! quasi-p diagnostics.
//!
//! Model schemas, requests, values, and per-row failures use the vocabulary of
//! the Graphcal batch model-evaluation protocol, re-exported as
//! [`graphcal_model_protocol`]. Graphcal is the primary model implementation:
//! the optional `graphcal` feature adds `GraphcalEvaluator`, which drives a
//! Graphcal model through the re-exported `graphcal_model_client`. Simple Rust
//! models are declared with [`ModelSchemaBuilder`] and evaluated in process.
//!
//! # Example
//!
//! ```
//! use tenax::{Dataset, Feature, Objective, Prim, PrimConfig};
//!
//! let dataset = Dataset::new(
//!     vec![
//!         Feature::continuous("load", vec![0.1, 0.4, 0.8, 0.9])?,
//!         Feature::categorical(
//!             "regime",
//!             ["stable", "stable", "fragile", "fragile"]
//!                 .into_iter()
//!                 .map(str::to_owned)
//!                 .collect(),
//!         )?,
//!     ],
//!     vec![false, false, true, true],
//! )?;
//! let config = PrimConfig::new(0.1, 0.1, 0.25, Objective::Lenient1)?;
//! let first_box = Prim::new(&dataset, config).find_box().unwrap();
//!
//! assert_eq!(first_box.trajectory()[0].statistics().mass(), 1.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # In-process model evaluation
//!
//! ```
//! use tenax::graphcal_model_protocol::{RowFailure, Value};
//! use tenax::{
//!     ChunkingPolicy, EvaluationTable, Evaluator, InputRow, ModelSchemaBuilder,
//!     ParallelInProcessEvaluator, RowContext, sample_uniform,
//! };
//!
//! let schema = ModelSchemaBuilder::new()
//!     .continuous("load", 0.0, 1.0)?
//!     .boolean_output("failure")?
//!     .build()?;
//! let load = schema.input_slot("load").expect("load is an input");
//! let failure = schema.output_slot("failure").expect("failure is an output");
//! let evaluator = ParallelInProcessEvaluator::new(
//!     schema.clone(),
//!     move |row: InputRow<'_>, _context: RowContext| -> Result<Vec<Value>, RowFailure> {
//!         Ok(vec![Value::Bool(row.quantity(load)? >= 0.7)])
//!     },
//!     ChunkingPolicy::new(64)?,
//! );
//!
//! let request = sample_uniform(&schema, 1_000, 42, 0)?;
//! let retained = request.clone();
//! let result = evaluator
//!     .evaluate(vec![request])
//!     .next()
//!     .expect("one request produces one result")?;
//! let mut table = EvaluationTable::new(schema.clone());
//! table.push(retained, result)?;
//! let dataset = table.dataset(failure)?;
//!
//! assert_eq!(dataset.row_count(), 1_000);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod data;
mod error;
mod evaluation;
#[cfg(feature = "arrow")]
mod evaluation_arrow;
mod evaluation_dataset;
mod evaluation_table;
#[cfg(feature = "graphcal")]
mod graphcal;
mod input;
mod model_schema;
mod parallel;
mod prim;
mod sampling;
mod seed;

#[cfg(feature = "arrow")]
pub use arrow_array;
#[cfg(feature = "arrow")]
pub use arrow_ipc;
#[cfg(feature = "arrow")]
pub use arrow_schema;
#[cfg(feature = "graphcal")]
pub use graphcal_model_client;
pub use graphcal_model_protocol;

pub use data::{CategoricalView, Dataset, Feature, FeatureKind, FeatureView};
pub use error::{DataError, PrimError};
pub use evaluation::{
    ChunkResult, ChunkResultError, EvalRequest, EvaluationId, Evaluator, InProcessEvaluator,
    ModelPanic, OutputRow, OutputRowError, RowContext, RowFailure, RowOutcome, SessionMismatch,
};
#[cfg(feature = "arrow")]
pub use evaluation_arrow::{
    EVALUATION_FORMAT_VERSION, EvaluationDecodeError, TENAX_METADATA_KEY, decode_evaluation_schema,
    evaluation_schema,
};
pub use evaluation_dataset::EvaluationDatasetError;
pub use evaluation_table::{EvaluatedRequest, EvaluationTable, EvaluationTableError};
#[cfg(feature = "graphcal")]
pub use graphcal::GraphcalEvaluator;
pub use input::{InputAccessError, InputRow, ValueKind};
pub use model_schema::{ModelSchemaBuilder, ModelSchemaError};
pub use parallel::{ChunkingPolicy, ChunkingPolicyError, ParallelInProcessEvaluator};
pub use prim::{
    BoxLimits, BoxStatistics, BoxStep, CategorySet, ContinuousRange, FeatureLimit, IntegerRange,
    Objective, Prim, PrimBox, PrimConfig, PrimPhase, QuasiPValue, Restriction,
};
pub use sampling::{SamplingError, sample_latin_hypercube, sample_uniform};
