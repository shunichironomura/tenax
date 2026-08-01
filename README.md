# Tenax

> Hold fast under uncertainty.

**Status:** Roadmap step 1 and the Phase A in-process slice of step 2 are implemented as an experimental Rust library. The API is not yet stable and no release has been published.

Tenax is a Rust-first toolkit for scenario discovery and robust decision-making under deep uncertainty (DMDU). Scenario discovery identifies combinations of uncertain inputs under which a candidate policy succeeds or fails. Tenax aims to support both analysis of existing experiment data and adaptive evaluation of callable simulation models.

The initial algorithmic focus is the Patient Rule Induction Method (PRIM). Additional scenario-discovery and DMDU methods, such as Classification and Regression Trees (CART), may follow once that foundation has been validated.

## Current functionality

Tenax currently provides a complete single-process path from a callable model to scenario discovery:

- Validated model schemas with continuous and integer bounds, categorical domains, binary outputs, and optional input units.
- Deterministic seeded uniform sampling with stable evaluation IDs and explicit model seeds.
- A synchronous, transport-independent, batch-in/chunk-stream-out `Evaluator` trait.
- Per-row success or failure data and a single-threaded in-process closure evaluator.
- Zero-copy borrowed column views over the native `Vec`-backed container, including deterministic `Int32` categorical dictionary codes.
- Explicit conversion of a successful evaluated chunk into a static PRIM dataset; failed rows are rejected rather than silently dropped.

Tenax also implements conventional Patient Rule Induction Method (PRIM) analysis for static binary input/output datasets:

- Continuous, integer, and categorical input features.
- EMA Workbench's `lenient1` default objective, the `lenient2` objective, and the original PRIM objective.
- Quantile-based peeling, data-aware categorical peeling, and pasting.
- Complete candidate trajectories with box limits, member row indices, coverage, density, mass, restricted-dimension counts, and one-sided quasi-p values.
- Repeated box discovery with the members of each final box removed from subsequent searches.

`true` output values identify the cases of interest. A minimal analysis looks like this:

```rust
use tenax::{Dataset, Feature, Objective, Prim, PrimConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Dataset::new(
        vec![
            Feature::continuous("load", vec![0.1, 0.4, 0.8, 0.9])?,
            Feature::categorical(
                "regime",
                ["stable", "stable", "fragile", "fragile"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            )?,
        ],
        vec![false, false, true, true],
    )?;
    let config = PrimConfig::new(0.1, 0.1, 0.25, Objective::Lenient1)?;
    let first_box = Prim::new(&data, config).find_box().unwrap();

    for candidate in first_box.trajectory() {
        println!("{:?}", candidate.statistics());
    }
    Ok(())
}
```

### In-process model workflow

A callable model uses the same schema and columnar request types that future process and network adapters will use:

```rust
use tenax::{
    Evaluator, InProcessEvaluator, InputRow, InputSchema, InputValue, ModelError,
    ModelSchema, OutputSchema, OutputValue, RowContext, evaluation_to_dataset,
    sample_uniform,
};

let schema = ModelSchema::new(
    vec![InputSchema::continuous("load", 0.0, 1.0)?],
    vec![OutputSchema::boolean("failure")?],
)?;
let load_position = schema.input_position("load")?;
let failure_position = schema.output_position("failure")?;
let evaluator = InProcessEvaluator::new(
    schema.clone(),
    move |row: InputRow<'_>, _context: RowContext| {
        let InputValue::Continuous(load) = row
            .value(load_position)
            .map_err(|error| ModelError::new(error.to_string()))?
        else {
            return Err(ModelError::new("load must be continuous"));
        };
        Ok(vec![OutputValue::Boolean(load >= 0.7)])
    },
);

let request = sample_uniform(&schema, 1_000, 42, 0)?;
let retained_request = request.clone();
let result = evaluator
    .evaluate(vec![request])
    .next()
    .expect("one request produces one result chunk");
let dataset = evaluation_to_dataset(
    &schema,
    retained_request,
    result,
    failure_position,
)?;
assert_eq!(dataset.row_count(), 1_000);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The evaluator yields one `ChunkResult` per request and may yield chunks out of request order. Each result retains row order within its request. The Phase A in-process evaluator is sequential; parallel and remote drivers are deferred.

Run the complete test suite with `cargo test --all-targets --all-features`.

### Independent reference suite

The integration fixture in [`tests/fixtures/ema_workbench_3_0_0.json`](tests/fixtures/ema_workbench_3_0_0.json) is generated independently by EMA Workbench 3.0.0. The Rust integration test compares every trajectory entry—including limits, selected rows, diagnostics, and quasi-p values—across all three objectives, mixed feature types, and a trajectory with explicit pasting.

Regenerate the reference fixture with:

```console
./scripts/generate_ema_reference.py
```

The executable script uses `uv` inline metadata to pin EMA Workbench and does not import or invoke Tenax.

## Vision

Tenax is planned as a toolbox composed of:

- A reusable Rust library containing the analysis algorithms and transport-independent evaluator interfaces.
- A Rust CLI that analyzes existing datasets or drives callable models through local or remote evaluators.
- A Python library, backed by the Rust core through PyO3, for use in scripts and notebooks.

A model evaluator maps a batch of input configurations to model outputs. It may run in process or behind a remote protocol; analysis algorithms should not depend on the transport used.

## Design goals

- **Correctness and reproducibility:** Validate conventional implementations against synthetic fixtures, behavioral invariants, published examples, and reference implementations such as EMA Workbench.
- **Efficient use of expensive models:** Use adaptive sampling to concentrate evaluations in informative regions instead of relying only on a fixed, precomputed ensemble.
- **Incremental execution:** Reuse previous model evaluations as new observations become available.
- **Batch-parallel execution:** Select and evaluate multiple model configurations concurrently.
- **Anytime results:** After each evaluation batch, return a valid intermediate result together with progress or uncertainty diagnostics, allowing execution to stop at a budget or deadline.
- **Measured performance:** Use Rust for a low-overhead, memory-efficient core, and substantiate performance claims with benchmarks.

## Roadmap

1. **Complete:** Implement conventional PRIM for static input/output datasets and establish a correctness test suite against EMA Workbench.
2. **In progress (Phase A complete):** The transport-independent evaluator abstraction, validated model schema, seeded uniform sampling, and in-process end-to-end workflow are implemented. Parallel execution, Arrow interchange, and process transports remain for Phases B–D.
3. Implement adaptive scenario discovery with explicit acquisition and stopping rules. Benchmark it against fixed sampling, such as Latin hypercube sampling, on representative problems.
4. Define a remote evaluation protocol that supports schema discovery, batch evaluation, failures, cancellation, and reproducible execution. Provide a CLI client and reference servers for Rust and Python.
5. Publish a Python package that wraps the Rust core through PyO3 and provides a notebook-friendly API.
6. Evaluate additional analysis methods, such as CART, robustness metrics, and sensitivity analysis, based on demonstrated user needs.

## What is not in scope

- Implementing users' domain-specific simulation models. Tenax treats those models as evaluators.
- Building a dedicated high-density visualization engine. Tenax should expose results in formats that work with visualization libraries such as [Datashader](https://datashader.org/) and [XY](https://reflex.dev/docs/xy/).

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
