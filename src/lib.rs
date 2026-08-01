//! Tenax provides scenario-discovery algorithms for decision-making under deep
//! uncertainty.
//!
//! The current release implements conventional Patient Rule Induction Method
//! (PRIM) analysis for static binary input/output datasets, plus a validated
//! model schema, reproducible uniform sampler, transport-independent evaluator
//! interface, and single-threaded in-process evaluation workflow. PRIM supports
//! continuous, integer, and categorical inputs and returns the complete peeling
//! and pasting trajectory with coverage, density, mass, and quasi-p diagnostics.
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
//! use tenax::{
//!     Evaluator, InProcessEvaluator, InputRow, InputSchema, InputValue, ModelError,
//!     ModelSchema, OutputSchema, OutputValue, RowContext, evaluation_to_dataset,
//!     sample_uniform,
//! };
//!
//! let schema = ModelSchema::new(
//!     vec![InputSchema::continuous("load", 0.0, 1.0)?],
//!     vec![OutputSchema::boolean("failure")?],
//! )?;
//! let load_position = schema.input_position("load")?;
//! let failure = schema.output_position("failure")?;
//! let evaluator = InProcessEvaluator::new(
//!     schema.clone(),
//!     move |row: InputRow<'_>, _context: RowContext| {
//!         let InputValue::Continuous(load) = row
//!             .value(load_position)
//!             .map_err(|error| ModelError::new(error.to_string()))?
//!         else {
//!             return Err(ModelError::new("load must be continuous"));
//!         };
//!         Ok(vec![OutputValue::Boolean(load >= 0.7)])
//!     },
//! );
//!
//! let request = sample_uniform(&schema, 1_000, 42, 0)?;
//! let retained = request.clone();
//! let result = evaluator
//!     .evaluate(vec![request])
//!     .next()
//!     .expect("one request produces one result");
//! let dataset = evaluation_to_dataset(&schema, retained, result, failure)?;
//!
//! assert_eq!(dataset.row_count(), 1_000);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod data;
mod error;
mod evaluation;
mod input;
mod prim;
mod sampling;
mod schema;
mod seed;

pub use data::{CategoricalView, Dataset, Feature, FeatureKind, FeatureView};
pub use error::{DataError, PrimError};
pub use evaluation::{
    ChunkResult, ChunkResultError, EvalRequest, EvaluationDatasetError, EvaluationId, Evaluator,
    InProcessEvaluator, ModelError, OutputRow, OutputRowError, OutputValue, RowContext, RowFailure,
    RowOutcome, evaluation_to_dataset,
};
pub use input::{InputAccessError, InputChunk, InputChunkError, InputRow, InputValue};
pub use prim::{
    BoxLimits, BoxStatistics, BoxStep, CategorySet, ContinuousRange, FeatureLimit, IntegerRange,
    Objective, Prim, PrimBox, PrimConfig, PrimPhase, QuasiPValue, Restriction,
};
pub use sampling::{SamplingError, sample_uniform};
pub use schema::{
    CategoricalDomain, ContinuousDomain, FeatureDomain, InputPosition, InputSchema, IntegerDomain,
    ModelFieldRole, ModelSchema, OutputKind, OutputPosition, OutputSchema, SchemaError,
    SchemaLookupError,
};
