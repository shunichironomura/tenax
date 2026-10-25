//! Splitting one evaluation request into consecutive row chunks.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use graphcal_model_protocol::{InputColumn, RequestBatch};
use thiserror::Error;

use crate::evaluation::{EvalRequest, EvaluationId};
use crate::seed::derive_seed;

const SPLIT_SEED_DOMAIN: u64 = 0x7465_6e61_785f_7370;

/// Errors raised while splitting an evaluation request.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum RequestSplitError {
    /// The chunk identifiers would leave the request's sequence space.
    #[error(
        "request {id} cannot be split into {chunks} chunks: the request sequence would overflow"
    )]
    SequenceOverflow {
        /// Identifier of the request being split.
        id: EvaluationId,
        /// Number of chunks the split needs.
        chunks: usize,
    },
}

impl EvalRequest {
    /// Splits this request into consecutive requests of at most
    /// `rows_per_request` rows, in row order.
    ///
    /// Splitting lets a long design be evaluated, stored, and recovered one
    /// part at a time, for example by several evaluators of an
    /// [`crate::EvaluatorPool`].
    ///
    /// Chunk `k` (counting from zero) takes the request's identifier with its
    /// request sequence advanced by `k`: a request identified by
    /// `from_run_seed(run_seed, sequence)` (see
    /// [`EvaluationId::from_run_seed`]) splits into
    /// `from_run_seed(run_seed, sequence + k)`. A split therefore occupies the
    /// request sequences `sequence..sequence + chunks`, which the caller must
    /// not use for other requests of the run. Sorting the chunks by
    /// identifier restores the original row order.
    ///
    /// Chunk `k`'s model seed is derived from the request seed and `k`, so the
    /// rows of different chunks receive distinct [`crate::RowContext`] seeds.
    /// A stochastic in-process model therefore draws different streams than
    /// it would for the unsplit request; a deterministic model, such as a
    /// Graphcal model, returns the same outputs. A request without rows has
    /// no chunks.
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use tenax::{EvaluationId, ModelSchemaBuilder, sample_latin_hypercube};
    ///
    /// let schema = ModelSchemaBuilder::new()
    ///     .continuous("load", 0.0, 1.0)?
    ///     .boolean_output("failure")?
    ///     .build()?;
    /// let design = sample_latin_hypercube(&schema, 10, 42, 0)?;
    /// let chunks = design.split(NonZeroUsize::new(4).expect("non-zero"))?;
    ///
    /// assert_eq!(
    ///     chunks.iter().map(|chunk| chunk.inputs().rows()).collect::<Vec<_>>(),
    ///     [4, 4, 2]
    /// );
    /// assert_eq!(chunks[2].id(), EvaluationId::from_run_seed(42, 2));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RequestSplitError::SequenceOverflow`] when the last chunk's
    /// request sequence exceeds `u64::MAX`.
    pub fn split(&self, rows_per_request: NonZeroUsize) -> Result<Vec<Self>, RequestSplitError> {
        let rows = self.inputs().rows();
        let chunks = rows.div_ceil(rows_per_request.get());
        let overflow = RequestSplitError::SequenceOverflow {
            id: self.id(),
            chunks,
        };
        let sequence = self.id().get() & u128::from(u64::MAX);
        let last = u128::try_from(chunks.saturating_sub(1)).map_err(|_| overflow)?;
        if sequence + last > u128::from(u64::MAX) {
            return Err(overflow);
        }
        Ok((0..chunks)
            .map(|chunk| {
                let start = chunk * rows_per_request.get();
                let range = start..(start + rows_per_request.get()).min(rows);
                self.chunk(chunk, range)
            })
            .collect())
    }

    /// Returns rows `range` as chunk `chunk`, whose identifier fits by the
    /// check in [`EvalRequest::split`].
    #[expect(
        clippy::expect_used,
        reason = "consecutive slices of a validated batch form a valid batch of the same schema"
    )]
    fn chunk(&self, chunk: usize, range: Range<usize>) -> Self {
        let columns = self
            .inputs()
            .columns()
            .iter()
            .map(|column| slice_column(column, range.clone()))
            .collect();
        let inputs = RequestBatch::new(Arc::clone(self.inputs().schema()), columns)
            .expect("a row range of a valid request batch is a valid request batch");
        let offset = chunk_number(chunk);
        Self::new(
            EvaluationId::new(self.id().get() + u128::from(offset)),
            derive_seed(self.seed(), offset, SPLIT_SEED_DOMAIN),
            inputs,
        )
    }
}

#[expect(
    clippy::expect_used,
    reason = "Rust's supported targets never have a `usize` wider than `u64`"
)]
fn chunk_number(chunk: usize) -> u64 {
    u64::try_from(chunk).expect("supported Rust targets represent every usize as u64")
}

fn slice_column(column: &InputColumn, range: Range<usize>) -> InputColumn {
    match column {
        InputColumn::Quantity(values) => InputColumn::Quantity(values[range].to_vec()),
        InputColumn::Int(values) => InputColumn::Int(values[range].to_vec()),
        InputColumn::Bool(values) => InputColumn::Bool(values[range].to_vec()),
        InputColumn::Key(values) => InputColumn::Key(values[range].to_vec()),
        InputColumn::Complex(values) => InputColumn::Complex(values[range].to_vec()),
        InputColumn::Datetime(values) => InputColumn::Datetime(values[range].to_vec()),
        InputColumn::Structured(values) => InputColumn::Structured(values[range].to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{SessionSchema, Value};

    use super::*;
    use crate::{ModelSchemaBuilder, sample_latin_hypercube};

    fn schema() -> Arc<SessionSchema> {
        ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .integer("n", 1, 9)
            .unwrap()
            .categorical("c", "Category", ["a", "b"])
            .unwrap()
            .boolean("enabled")
            .unwrap()
            .boolean_output("p")
            .unwrap()
            .build()
            .unwrap()
    }

    fn rows(size: usize) -> NonZeroUsize {
        NonZeroUsize::new(size).unwrap()
    }

    fn column_values(requests: &[EvalRequest], schema: &SessionSchema, input: &str) -> Vec<Value> {
        let slot = schema.input_slot(input).unwrap();
        requests
            .iter()
            .flat_map(|request| {
                (0..request.inputs().rows())
                    .map(move |row| request.inputs().value(slot, row).unwrap().to_owned_value())
            })
            .collect()
    }

    #[test]
    fn chunks_keep_every_row_in_order_with_consecutive_sequences() {
        let schema = schema();
        let design = sample_latin_hypercube(&schema, 7, 3, 5).unwrap();
        let chunks = design.split(rows(3)).unwrap();

        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.inputs().rows())
                .collect::<Vec<_>>(),
            [3, 3, 1]
        );
        for input in ["x", "n", "c", "enabled"] {
            assert_eq!(
                column_values(&chunks, &schema, input),
                column_values(std::slice::from_ref(&design), &schema, input),
                "{input}"
            );
        }
        assert_eq!(
            chunks.iter().map(EvalRequest::id).collect::<Vec<_>>(),
            [5, 6, 7].map(|sequence| EvaluationId::from_run_seed(3, sequence))
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.validate_against(&schema).is_ok())
        );
    }

    #[test]
    fn chunks_have_distinct_seeds() {
        let design = sample_latin_hypercube(&schema(), 4, 3, 0).unwrap();
        let chunks = design.split(rows(1)).unwrap();
        let mut seeds = chunks.iter().map(EvalRequest::seed).collect::<Vec<_>>();
        seeds.sort_unstable();
        seeds.dedup();

        assert_eq!(seeds.len(), 4);
    }

    #[test]
    fn a_request_that_fits_is_one_chunk() {
        let design = sample_latin_hypercube(&schema(), 4, 3, 0).unwrap();
        let chunks = design.split(rows(10)).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id(), design.id());
        assert_eq!(chunks[0].inputs(), design.inputs());
    }

    #[test]
    fn rejects_a_split_past_the_last_sequence() {
        let design = sample_latin_hypercube(&schema(), 3, 3, u64::MAX - 1).unwrap();

        assert!(design.split(rows(2)).is_ok());
        assert_eq!(
            design.split(rows(1)),
            Err(RequestSplitError::SequenceOverflow {
                id: design.id(),
                chunks: 3
            })
        );
    }
}
