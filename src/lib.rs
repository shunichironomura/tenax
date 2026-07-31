#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Tenax provides scenario-discovery algorithms for decision-making under deep
//! uncertainty.
//!
//! The current release implements conventional Patient Rule Induction Method
//! (PRIM) analysis for static binary input/output datasets. It supports
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
//! # Ok::<(), tenax::PrimError>(())
//! ```

mod data;
mod error;
mod prim;

pub use data::{Dataset, Feature, FeatureKind};
pub use error::PrimError;
pub use prim::{
    BoxLimits, BoxStatistics, BoxStep, CategorySet, ContinuousRange, FeatureLimit, IntegerRange,
    Objective, Prim, PrimBox, PrimConfig, PrimPhase, QuasiPValue, Restriction,
};
