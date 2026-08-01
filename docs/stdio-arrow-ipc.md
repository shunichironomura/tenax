# Tenax stdio Arrow IPC protocol

Status: **version 1**, implemented by the `stdio` crate feature. The Arrow
schema contract carried by this protocol is version **2**.

This document is the language-neutral contract for persistent local model
servers, including the planned Graphcal adapter. It uses standard Arrow IPC
streams; there is no JSON envelope, shell framing, or Tenax-native binary
encoding.

## Goals

- Compile or initialize a model once, then evaluate many request batches.
- Use the same Arrow request/result payloads later carried by HTTP or Flight.
- Support Rust, PyArrow, Graphcal, and other Arrow implementations.
- Keep evaluation IDs and seeds stable across retries.
- Preserve one failure per row without turning model-domain failures into
  transport failures.
- Permit result batches to arrive out of request order.

## Process and stream lifecycle

The child is started with piped stdin and stdout. Stderr remains available for
logs and human diagnostics. **The child must never write text or logs to
stdout.**

Stdout is the concatenation of two complete Arrow IPC streaming-format
streams. Stdin is one Arrow IPC streaming-format stream:

```text
parent / Tenax                                  child / model server

                         spawn
                           │
stdout ◀── discovery IPC stream: schema + EOS ──┤
       ◀── result IPC stream: schema ────────────┤
                           │                     │
stdin  ├── request IPC stream: schema ─────────▶│
       ├── request RecordBatch A ──────────────▶│
       ├── request RecordBatch B ──────────────▶│
       │                                        │ evaluate, possibly parallel
stdout ◀── result RecordBatch B ────────────────┤
       ◀── result RecordBatch A ────────────────┤
       │                                        │
stdin  └── EOS (graceful shutdown) ────────────▶│
stdout ◀── EOS ─────────────────────────────────┘
```

Startup order is normative:

1. Load enough of the model to discover and validate its public interface.
2. Write and finish a **schema-only discovery stream** on stdout.
3. Create the result stream on the same stdout and flush its schema.
4. Only then open/read the request stream on stdin.

This order prevents both processes from waiting for the other pipe's schema.
The discovery stream must contain no record batches. Arrow's EOS marker
self-delimits it; the result stream begins immediately after that marker.
Implementations must not place a buffered reader around both streams if that
reader can consume bytes past the first EOS without making them available to
the second stream reader.

A process stays alive across repeated Tenax `evaluate` calls. Request-stream
EOS asks it to finish the result stream and exit. Unexpected EOF, malformed
IPC, an unknown ID, or a process exit while IDs remain pending is an
evaluator-level error. Dropping a Tenax result iterator before all requested
results arrive terminates the child because unread bytes would make reuse
ambiguous.

## Common metadata

All schemas carry:

| Metadata key | Value | Meaning |
| --- | --- | --- |
| `tenax.schema.version` | `2` | Arrow field and batch contract |
| `tenax.batch.kind` | See below | Semantic schema/batch kind |

The discovery schema additionally carries:

| Metadata key | Value |
| --- | --- |
| `tenax.stdio.version` | `1` |

Unknown schema and field metadata keys are retained or ignored, not rejected.
Graphcal should namespace richer annotations and provenance as `graphcal.*`.

## Discovery schema

`tenax.batch.kind = model_schema`.

Fields are all model inputs in declaration order followed by all selected
public outputs in declaration order. Every discovery field is non-nullable.
Every name is an opaque user/model name; Tenax does not infer semantics from
its spelling.

| Model concept | Arrow type | Required field metadata |
| --- | --- | --- |
| Continuous input | `Float64` | `tenax.field.role=input`, finite decimal `tenax.input.lower` and `tenax.input.upper` |
| Integer input | `Int64` | `tenax.field.role=input`, decimal `tenax.input.lower` and `tenax.input.upper` |
| Categorical input | `Dictionary<Int32, Utf8>` | `tenax.field.role=input`, `tenax.input.categories` as a JSON string array |
| Boolean output | `Boolean` | `tenax.field.role=output` |

An input may carry `tenax.input.unit`. Numeric values and bounds are expressed
in that unit. A Graphcal adapter should parse that unit at interface creation
and convert bound/request values to Graphcal's internal representation; it
must not infer units from field names. Dimensioned Graphcal parameters need a
unit annotation. Deterministic models may ignore request seeds.

Categories in `tenax.input.categories` form the complete permitted domain.
Tenax writes them in lexical order, but readers must bind dictionary values by
value rather than assuming a particular incoming dictionary key assignment.

Tenax currently requires:

- at least one input and one Boolean output;
- finite, closed continuous bounds;
- closed `Int64` integer bounds;
- a non-empty finite category set; and
- names unique across the model's input/output interface.

A Graphcal model interface with an unbounded parameter, a non-Boolean selected
output, a structured/indexed value, or another unsupported shape must be
rejected explicitly by the adapter. It must not silently invent bounds,
flatten structure into delimiter-encoded names, or coerce a quantity to an
untyped scalar. Future schema versions can add those types coherently.

## Request stream and batches

The request stream schema is fixed for the process lifetime and has
`tenax.batch.kind = evaluation_request`.

Fields are:

1. Every model input, using the discovery declaration and order.
2. Evaluation ID context.
3. Request seed context.

| Trailing field name | Arrow type | Nullable | Metadata |
| --- | --- | ---: | --- |
| `tenax.evaluation_id` | `FixedSizeBinary(16)` | no | `tenax.field.role=context`, `tenax.field.kind=evaluation_id` |
| `tenax.evaluation_seed` | `UInt64` | no | `tenax.field.role=context`, `tenax.field.kind=evaluation_seed` |

One non-empty record batch is one `EvalRequest`. The 16 ID bytes are the
unsigned `u128` value in big-endian/network byte order. The ID and seed are
repeated on every row and must be constant within the batch. Putting context
in arrays—not per-batch schema metadata—is intentional: Arrow IPC streams have
one schema, so this permits arbitrarily many IDs in one standard stream and
maps unchanged to a future streaming HTTP body.

The model server must preserve row order within each request. For a
row-at-a-time stochastic model, derive row randomness deterministically from
the request seed and zero-based row position. The exact Tenax derivation is a
native evaluator convenience, not a requirement for foreign models; a model
server must document and keep its own stream policy stable.

## Result stream and batches

The result stream schema is fixed for the process lifetime and has
`tenax.batch.kind = evaluation_result`.

Its required prefix is:

1. Every selected model output in discovery order, but **nullable** so failed
   rows can be represented without a sentinel value.
2. Evaluation ID context.
3. Outcome status.
4. Failure message.

| Trailing field name | Arrow type | Nullable | Metadata |
| --- | --- | ---: | --- |
| `tenax.evaluation_id` | `FixedSizeBinary(16)` | no | `tenax.field.role=context`, `tenax.field.kind=evaluation_id` |
| `tenax.outcome_status` | `UInt8` | no | `tenax.field.role=outcome`, `tenax.field.kind=outcome_status` |
| `tenax.failure_message` | `Utf8` | yes | `tenax.field.role=outcome`, `tenax.field.kind=failure_message` |

Status codes are versioned protocol data, not provider-specific strings:

| Code | Meaning | Outputs | Message |
| ---: | --- | --- | --- |
| `0` | success | all non-null | null |
| `1` | model returned/reported an error | all null | non-null |
| `2` | model invocation panicked with a string payload | all null | non-null |
| `3` | model invocation panicked without a string payload | all null | null |
| `4` | peer rejected inputs beyond the shared boundary checks | all null | non-null |
| `5` | peer produced schema-invalid outputs | all null | non-null |

For Graphcal, parameter-binding failures, runtime/domain failures, failed
assertions that invalidate the requested output, and other ordinary per-case
calculation failures normally map to code `1`, with a stable human diagnostic.
If Graphcal exposes a typed diagnostic code, assertion details, or provenance,
it may append columns as described below rather than encoding control
semantics into the message text.

One result record batch corresponds to exactly one request record batch. Its ID
is repeated and constant, its row count is identical, and its rows remain in
request-row order. Different result batches may be emitted in any completion
order. Unknown, duplicate, inconsistent, or missing IDs are protocol errors.

### Extensible result columns

A server may append peer-specific result columns after the required prefix.
They are declared once in the fixed result-stream schema, and every result
batch supplies them. Each must carry `tenax.field.role=extension`; Tenax ignores
its name, type, nullability, and values. This is the compatible place for Graphcal diagnostic
codes, assertion outcomes, model/plugin provenance, or timing data. Such
columns cannot alter the meaning of required status/output fields.

## Failure scope

- A valid result batch with nonzero row statuses is **model behavior/data**.
  Other rows and requests remain valid.
- Invalid Arrow, a schema mismatch, duplicate/unknown IDs, the wrong row count,
  broken pipes, or child exit is an **evaluator/process error**. Tenax yields an
  error from the evaluation iterator and marks that process unusable.
- Stable IDs let a driver retry a whole affected request against a fresh child
  without changing model identity or confusing it with a new evaluation.

No NaN, infinity, magic output value, empty string, or missing batch represents
failure. Continuous inputs are finite and output nullability is controlled by
the status field.

## Graphcal implementation checklist

A compatible Graphcal command can use this protocol without depending on
Tenax's Rust-native data types:

1. Load and compile the `.gcl` project once; retain an immutable/reusable plan.
2. Project configured public scalar parameters and Boolean outputs into the
   discovery schema. Require explicit finite domains and units where Tenax
   needs them.
3. Emit the discovery and result headers in the normative startup order.
4. Decode request Arrow arrays into typed Graphcal runtime bindings. Bind
   quantities using `tenax.input.unit`, not bare `f64` convention.
5. Evaluate rows independently, optionally in parallel. A deterministic
   Graphcal model can ignore the seed.
6. Map each requested output set to success or one row failure; never leak NaN
   or infinity as a value.
7. Echo ID and row order exactly, while allowing request batches to complete
   out of order.
8. Put logs on stderr and reserve stdout exclusively for the two IPC streams.
9. On stdin EOS, finish the result stream and exit successfully.

The same request/result schemas are suitable for an HTTP POST body and response
stream. A future network adapter therefore changes transport framing and
asynchrony, not Graphcal's model binding or Arrow payload semantics.
