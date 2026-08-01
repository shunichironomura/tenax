//! Measures validated conversion of 160 MB of inputs plus 24 MB of context.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tenax::arrow::{EvalRequestRef, RecordBatch};
use tenax::{
    EvalRequest, EvaluationId, Feature, InputChunk, InputSchema, ModelSchema, OutputSchema,
};

const ROW_COUNT: usize = 1_000_000;
const FEATURE_COUNT: usize = 20;
const WIRE_BYTES: u64 = 184_000_000;

#[expect(
    clippy::expect_used,
    reason = "a benchmark fixture should fail immediately when its static setup is invalid"
)]
fn fixture() -> (ModelSchema, EvalRequest, RecordBatch) {
    let names = (0..FEATURE_COUNT)
        .map(|position| format!("x_{position}"))
        .collect::<Vec<_>>();
    let inputs = names
        .iter()
        .map(|name| InputSchema::continuous(name, 0.0, 20.0))
        .collect::<Result<Vec<_>, _>>()
        .expect("benchmark input schemas are valid");
    let schema = ModelSchema::new(
        inputs,
        vec![OutputSchema::boolean("failure").expect("benchmark output schema is valid")],
    )
    .expect("benchmark model schema is valid");
    let features = names
        .iter()
        .enumerate()
        .map(|(position, name)| {
            let value = f64::from(
                u32::try_from(position).expect("the benchmark feature count fits in u32"),
            );
            Feature::continuous(name, vec![value; ROW_COUNT])
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("benchmark feature columns are valid");
    let request = EvalRequest::new(
        EvaluationId::new(1),
        42,
        InputChunk::new(&schema, features).expect("benchmark input chunk is valid"),
    );
    let batch = RecordBatch::try_from(EvalRequestRef::new(&schema, &request))
        .expect("native benchmark fixture converts to Arrow");
    EvalRequest::try_from((&schema, &batch))
        .expect("Arrow benchmark fixture converts back to native");
    (schema, request, batch)
}

fn arrow_conversion(criterion: &mut Criterion) {
    let (schema, request, batch) = fixture();
    let mut group = criterion.benchmark_group("arrow_conversion/1m_rows_x_20_f64");
    group.throughput(Throughput::Bytes(WIRE_BYTES));

    group.bench_function("native_to_arrow", |bencher| {
        bencher.iter(|| {
            black_box(RecordBatch::try_from(EvalRequestRef::new(
                black_box(&schema),
                black_box(&request),
            )))
        });
    });
    group.bench_function("arrow_to_native", |bencher| {
        bencher.iter(|| {
            black_box(EvalRequest::try_from((
                black_box(&schema),
                black_box(&batch),
            )))
        });
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8));
    targets = arrow_conversion
}
criterion_main!(benches);
