# DPS lake model: sample → evaluate → PRIM → visualize

This example runs Tenax's complete Phase A workflow on the Direct Policy Search
(DPS) lake problem from the EMA Workbench [open-exploration
tutorial](https://emaworkbench.readthedocs.io/en/latest/indepth_tutorial/open-exploration.html):

1. declare the five deep uncertainties and five policy levers as a
   `ModelSchema`;
2. draw 5,000 deterministic uniform experiments;
3. evaluate 150 stochastic lake realizations over 100 years for each row with
   the sequential `InProcessEvaluator`;
4. classify `max_P < 0.8` as the case of interest;
5. fit the first conventional PRIM box and export its complete trajectory; and
6. use Python and [XY](https://reflex.dev/docs/xy/) to create interactive HTML
   and static PNG plots, plus Matplotlib for an EMA-style pairwise
   scatter-and-box matrix.

The lake equations and constants follow EMA Workbench's
[`dps_lake_model.py`](https://github.com/quaquel/EMAworkbench/blob/master/docs/source/indepth_tutorial/dps_lake_model.py).
The model calculates all four tutorial outcomes internally, although the current
Phase A schema exposes only the binary `max_P < 0.8` classification used by
PRIM.

## Run it

From the repository root:

```console
cargo run --release --example lake_model
./examples/lake_model/plot.py
```

The Rust executable writes to `target/lake_model/` by default. An alternative
output directory can be passed as its sole argument:

```console
cargo run --release --example lake_model -- /tmp/lake-analysis
./examples/lake_model/plot.py --input /tmp/lake-analysis
```

The plotting script requires Python 3.14 or newer and uses `uv` inline metadata
with a checked-in lockfile, so no separate environment setup is needed. It pins
XY 0.0.5 because XY's pre-1.0 releases may contain breaking changes and uses the
locked Matplotlib release for the pairwise plot.

## Outputs

The Rust workflow writes:

| File | Contents |
| --- | --- |
| `experiments.csv` | sampled model inputs and the binary case-of-interest output |
| `trajectory.csv` | phase, coverage, density, mass, dimensions, and counts for every PRIM candidate |
| `limits.csv` | every candidate's inclusive feature limits and quasi-p values |
| `summary.txt` | run controls and final-box statistics |

The Python script validates that the selected limits reproduce the exported
point and case counts, then writes PNG and self-contained interactive HTML
versions of:

- `plots/prim_tradeoff`: coverage versus density across the complete trajectory;
- `plots/experiments_b_q`: the experiments and selected box projected onto the
  two most important lake uncertainties; and
- `plots/selected_box_limits`: an EMA-style normalized view of every restricted
  dimension.

It also writes `plots/prim_pairs_scatter.png`, a Matplotlib pair matrix analogous
to EMA Workbench's `box.show_pairs_scatter(...)`. It includes only the selected
candidate's restricted dimensions, colors all experiments by case-of-interest
status, shows class histograms on the diagonal, and projects the selected box
onto every off-diagonal feature pair.

By default, all plots highlight the final trajectory step. Inspect another
candidate without rerunning the model or PRIM, for example:

```console
./examples/lake_model/plot.py --step 44
```

For the pinned seed and dependency lockfile, the final candidate is step 58:

| Statistic | Value |
| --- | ---: |
| Coverage | 0.3516 |
| Density | 0.9735 |
| Mass | 0.0528 |
| Cases of interest / all experiments | 731 / 5,000 |

It restricts three dimensions:

```text
b    ∈ [0.3859493824, 0.4499131339]
q    ∈ [3.5831123410, 4.4997556579]
mean ∈ [0.0100063096, 0.0408175587]
```

This recovers the tutorial's qualitative result: desirable lake states are
concentrated where phosphorus removal (`b`) and the recycling exponent (`q`)
are high, with an additional weaker natural-inflow restriction.

## Differences from the EMA Workbench notebook

This is intentionally an example of **Tenax's current workflow**, not a
byte-for-byte reproduction of EMA Workbench's notebook output:

- Phase A has seeded independent uniform sampling, not Latin hypercube
  sampling. It jointly samples 5,000 uncertainty-policy rows instead of taking
  a full factorial product of 1,000 scenarios and five policies.
- The current schema does not yet distinguish uncertainties from levers; all
  ten inputs are ordinary typed features for PRIM.
- Output schemas are currently binary, so `max_P < 0.8` is classified at the
  evaluator boundary rather than from a retained scalar-output dataframe.
- Each row uses its deterministic `RowContext` seed and a local `ChaCha12` RNG.
  EMA Workbench's Python source uses NumPy's legacy global RNG, so the
  stochastic samples and exact trajectory differ.
- Evaluation is sequential. Parallel execution and Latin hypercube sampling
  are deferred to later roadmap phases.
- XY does not yet provide a dedicated scatter-matrix composition, so the three
  single-panel plots use XY while the pairwise scatter-and-box matrix uses
  Matplotlib.

The documented lever domains include zero for `r1` and `r2`, while the DPS
release equation divides by each radius. The model therefore rejects an exact
zero radius as explicit per-row failure data. The pinned continuous sample does
not contain either endpoint.
