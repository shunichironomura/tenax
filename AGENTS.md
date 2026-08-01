# Guidelines for Coding Agents

## Project Overview

- Tenax is a Rust-first toolkit for scenario discovery and robust decision-making under deep uncertainty (DMDU).
- The current implementation is an experimental library for conventional Patient Rule Induction Method (PRIM) analysis of static binary input/output datasets. The roadmap also includes transport-independent model evaluation, adaptive scenario discovery, a CLI, and Python bindings.
- Prioritize correctness, reproducibility, and explicit domain semantics over convenience or premature performance work.
  - Reject invalid, ambiguous, missing, or non-finite data at a well-defined boundary rather than allowing it to acquire accidental algorithmic meaning.
  - Make ordering, tie-breaking, randomness, and stopping behavior explicit and deterministic where practical.
  - Validate algorithm behavior against independent implementations, published definitions, and behavioral invariants.
- Tenax has not been published and its API is not stable. Breaking changes are acceptable when they produce a simpler, safer, or cleaner design.
  - Do not add compatibility workarounds for an unpublished API. Make the coherent design change and update the repository consistently.

## Documentation

- `README.md` is the primary user-facing overview, usage guide, status report, and roadmap. Keep it aligned with the implemented public API and behavior.
- Public Rust APIs are documented with rustdoc comments in `src/`. Update module-level and item-level documentation, including examples, when changing an API.
- `tests/fixtures/ema_workbench_3_0_0.json` is an independently generated reference suite. `tests/ema_workbench_reference.rs` compares complete Tenax PRIM trajectories with it.
  - Regenerate the fixture only through `./scripts/generate_ema_reference.py`.
  - The generator must remain independent of Tenax; it must not import, invoke, or derive expected values from this crate.
  - Do not hand-edit or blindly regenerate the fixture to make a failing test pass. Investigate the semantic difference first.
- Detailed research and design discussions live in `.local/`. These are raw notes and may be incomplete, speculative, or obsolete. Prefer the current code, tests, `README.md`, and explicit user direction when they disagree.

## Implementation Guidelines

When adding, modifying, or removing behavior, update the applicable artifacts together:

- Focused unit tests for validation, edge cases, candidate selection, and numerical behavior.
- Integration and independent-reference tests for externally observable PRIM behavior.
- Public rustdoc and examples in `src/`.
- `README.md` when capabilities, usage, status, or roadmap claims change.
- The reference generator and fixture when the intentionally tested EMA Workbench behavior or reference inputs change.

Additional rules:

- Keep reference data independent. A fixture mismatch is evidence to investigate, not output to bless.
- Preserve reproducibility. Do not depend on hash iteration order, ambient randomness, wall-clock timing, or unspecified sort stability for observable results. Pass seeds or random-number generators explicitly when randomness is introduced.
- Define edge-case behavior deliberately, especially for empty sets, singleton samples, duplicate values, tied candidates, zero cases of interest, and floating-point boundaries.
- Prefer a clear, correct implementation backed by tests before optimizing it. Support performance claims with measurements representative of the intended workload.
- Keep algorithm code transport-independent. Filesystem access, serialization, Arrow conversion, Python bindings, networking, retries, cancellation, and concurrency belong in adapters or drivers around the core.

Before considering a code change complete, run the applicable checks from the repository root:

```console
cargo fmt --all --check
cargo clippy --all-targets --all-features
cargo test --all-targets --all-features
```

## GitHub Issues and Pull Requests

If a pull request template is present, comply with it.

For pull request classification, a **breaking change** is a change that requires downstream users to modify Rust source code or persisted/interchange data, or that intentionally changes documented algorithm semantics. Changes only to private implementation details do not count as breaking changes.

When an agent files a GitHub Issue or opens a GitHub Pull Request, put this alert note at the very top of the description. Add the same note at the beginning of any issue or pull request comment written by an agent:

> [!WARNING]
> This content was written by an AI agent and must be verified by a human developer. After human verification, this alert may be removed.

The human developer must remove this note after verifying the description or comment's contents.

## Type Safety: Encode Semantics in Types, Not Conventions

Tenax is scientific software. Distinct domain concepts must be distinct types; do not lean on string conventions, naming patterns, unvalidated primitives, or an “everyone knows” rule when a typed alternative can make invalid use impossible.

### Hard rules

- **No flat-string encodings of structured data.** If a value has parts (model and feature identity, evaluation and chunk identity, role and name, etc.), model it as a struct or enum that names those parts. Do not concatenate fields with a separator (`":"`, `"/"`, `"@"`, etc.) and later split or inspect the string to recover them. In particular, avoid:
  - `format!("{model}:{feature}")` to fabricate a composite identifier.
  - `split_once`, `contains`, `starts_with`, or casing checks to recover structure that should have been retained in a type.
  - `HashMap<String, …>` or `HashSet<String>` keyed by ad hoc composite strings when the components already have domain types.
  - Opaque user-provided labels, such as a feature name or categorical value, may remain strings when they carry no hidden structure.
- **No name-based dispatch.** Do not infer feature kinds, input/output roles, evaluation status, or algorithm phases from prefixes, suffixes, casing, or reserved names. Carry the category as a typed field or enum variant.
- **No string-matched control flow on internal identifiers.** Closed sets such as objectives, feature kinds, PRIM phases, failure reasons, or transport kinds belong in enums. Parsing from a string should happen once at a serialization, CLI, or other boundary; the core should pattern-match on the typed value.
- **Stringify only at boundaries.** Rendering for diagnostics, logs, fixtures, file/wire formats, or third-party APIs is appropriate, but conversion happens at the boundary rather than throughout the functional core.
- **No sentinel values for failure or absence.** Do not use NaN, infinity, magic numbers, empty strings, or out-of-range indices to encode errors. Reject invalid values with a descriptive `Result`; use `Option` only when absence is a valid domain state.

### Functional core, imperative shell

Treat Tenax as a functional core surrounded by imperative shells:

- The core owns validated domain values and deterministic transformations: datasets and schemas → candidate generation and scoring → trajectories or findings.
- Future adaptive algorithms should be sans-I/O state machines that emit typed evaluation requests and consume typed results. They must not invoke an evaluator directly.
- Drivers and adapters own model invocation, disk access, Arrow/Parquet conversion, Python FFI, network transports, process management, concurrency, retries, cancellation, and serialization.

Keep dependencies directed from domain types and numerical primitives toward algorithms, then from adapters and applications toward that core. Do not make the core depend on a particular container, runtime, transport, or language binding merely because the first consumer uses one.

### Module scope and layering

Prefer small modules with a single, atomic responsibility over broad domain grab bags. Two concepts belonging to scenario discovery is not enough reason to put them in the same file; they should also live at the same abstraction layer and have a real implementation dependency. Keep validated data/schema types, numerical algorithm machinery, evaluator orchestration, transport adapters, and user-facing bindings separate when their dependencies differ.

Before adding a type to an existing module, ask:

1. Is this type at the same abstraction layer as the existing contents (domain data, PRIM algorithm, evaluator interface, driver, transport, CLI/binding, etc.)?
2. Does it depend on the same upstream concepts, or would it pull in unrelated dependencies?
3. Would moving it to a smaller focused module make dependency direction clearer?
4. Is the module name still accurate after adding it?

If a file starts mixing core algorithm structures with I/O, transport, or binding structures, split it. Re-exporting for public API ergonomics is acceptable when it does not hide dependency direction or create cycles.

### Avoid zero-value wrapper types

Do not introduce a newtype merely because a value can be described with a different noun. A wrapper is justified only when it preserves or enforces a real invariant, abstraction boundary, ownership/scope distinction, unit, phase distinction, or API safety property. If a wrapper only forwards `Display`, `AsRef`, or `inner()` to an already precise enum or struct, use the underlying type directly.

Examples of wrappers that need extra justification:

- `ObjectiveName(Objective)` when `Objective` is already a closed semantic enum.
- `ValidatedConfig(PrimConfig)` when `PrimConfig` already enforces its invariants at construction.
- A boundary-layer wrapper around a core value that is immediately unwrapped by the next function.

When a distinction is useful, encode the missing information explicitly—for example, a bounded probability type, a stable `EvaluationId`, or a source value paired with its validated representation—rather than adding an opaque wrapper with no additional semantics.

### Numerical and algorithmic correctness

- Validate public inputs at construction or ingestion boundaries. State whether ranges are open or closed, reject non-finite values, and keep invalid states out of the algorithm core.
- Give floating-point comparisons an explicit mathematical meaning. Do not add broad tolerances or loosen the independent-reference tolerance merely to conceal a regression.
- Make candidate ordering and tie-breaking total and deterministic. If a result depends on input order, document and test that fact.
- Test properties and invariants in addition to examples: membership must agree with box limits, trajectory statistics must agree with members, peeling/pasting must respect mass constraints, and repeated discovery must remove only the intended rows.
- Keep units and denominators explicit in names and types. Do not interchange counts, fractions, percentages, row indices, feature indices, or evaluation IDs because their primitive representations happen to match.

### When you reach for a string

Stop and ask:

1. Does this string carry structure, a closed set of variants, or a parseable shape?
2. Will multiple sites need to construct or destructure it the same way?
3. If the convention changed, how many sites would need to change?

If the answers indicate hidden semantics, introduce a type: a validated newtype for an opaque identifier, an enum for finite variants, or a struct for composites. Place it where the data belongs in the layering, not merely where it is first consumed.

### When the rule conflicts with adjacent code

If existing code uses a convention that violates these rules, prefer fixing it over copying it. If the fix is too large for the current change, fence the convention into the smallest possible boundary, leave a `// TODO(#NNN):` that points to a tracking issue, and do not widen the convention. Conventions spread; types contain them.
