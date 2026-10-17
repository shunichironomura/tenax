//! An [`Evaluator`] that drives a Graphcal model through
//! [`graphcal_model_client`].

use std::sync::{Arc, Mutex, PoisonError};

use graphcal_model_client::{ClientError, Session};
use graphcal_model_protocol::{
    OutputSlot, ResultBatch, RowOutcome as ProtocolRowOutcome, SessionSchema, ValueRef,
};

use crate::evaluation::{
    ChunkResult, EvalRequest, Evaluator, OutputRow, RowFailure, RowOutcome, rejected_rows,
};

/// Evaluates requests with one running `graphcal model serve` session.
///
/// The session's schema is the evaluator's schema, so requests sampled from
/// [`Evaluator::schema`] can be sent unchanged. Each request becomes one
/// protocol request batch; rows the model reports as failed (a failed
/// assertion, an input outside its declared domain, or another evaluation
/// error) become [`RowFailure::Model`] outcomes. A request built for another
/// session schema is rejected row by row with [`RowFailure::InvalidInput`]
/// without contacting the server.
///
/// Graphcal models are deterministic, so the request seed is not sent to the
/// server and has no effect on the result.
///
/// Requests are evaluated sequentially in request order, because one session
/// serves one request at a time. An I/O, protocol, or server failure is
/// reported as the iterator's [`ClientError`]; the session is unusable
/// afterwards, so every later request reports
/// [`ClientError::SessionClosed`].
///
/// ```no_run
/// use graphcal_model_client::{Bool, GraphcalCommand, Model, Quantity};
/// use tenax::{Evaluator, GraphcalEvaluator, sample_latin_hypercube};
///
/// let model = Model::describe(GraphcalCommand::new("reliability.gcl"))?;
/// let load = model.input::<Quantity>("load")?;
/// let failure = model.output::<Bool>("failure")?;
/// let session = model.session().vary(&load).output(&failure).spawn()?;
///
/// let evaluator = GraphcalEvaluator::new(session);
/// let request = sample_latin_hypercube(evaluator.schema(), 1_000, 42, 0)?;
/// let result = evaluator
///     .evaluate(vec![request])
///     .next()
///     .expect("one request produces one result")?;
/// assert_eq!(result.rows().len(), 1_000);
/// evaluator.shutdown()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct GraphcalEvaluator {
    schema: Arc<SessionSchema>,
    session: Mutex<Session>,
}

impl GraphcalEvaluator {
    /// Wraps a running session.
    #[must_use]
    pub fn new(session: Session) -> Self {
        Self {
            schema: Arc::clone(session.schema()),
            session: Mutex::new(session),
        }
    }

    /// Ends the session's request stream and waits for the server to exit.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the server does not stop cleanly.
    pub fn shutdown(self) -> Result<(), ClientError> {
        // A poisoned lock only means a panic interrupted an earlier exchange;
        // the session itself reports whether it is still usable.
        self.session
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .shutdown()
    }

    fn evaluate_request(&self, request: EvalRequest) -> Result<ChunkResult, ClientError> {
        if let Err(mismatch) = request.validate_against(&self.schema) {
            return Ok(ChunkResult::from_rows(
                request.id(),
                rejected_rows(&request, mismatch),
            ));
        }
        let id = request.id();
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let protocol_request = session.request(request.into_inputs())?;
        let evaluation = session.evaluate(&protocol_request)?;
        drop(session);
        Ok(ChunkResult::from_rows(
            id,
            result_rows(&self.schema, evaluation.batch()),
        ))
    }
}

/// Converts every protocol row outcome of `batch` to a Tenax row outcome.
fn result_rows(schema: &SessionSchema, batch: &ResultBatch) -> Vec<RowOutcome> {
    let slots = schema
        .outputs()
        .iter()
        .filter_map(|output| schema.output_slot(output.name().as_str()))
        .collect::<Vec<OutputSlot>>();
    batch
        .outcomes()
        .map(|outcome| match outcome {
            ProtocolRowOutcome::Success(values) => {
                let values = slots
                    .iter()
                    .filter_map(|slot| values.get(*slot).map(ValueRef::to_owned_value))
                    .collect();
                match OutputRow::new(schema, values) {
                    Ok(outputs) => RowOutcome::Success(outputs),
                    Err(error) => RowOutcome::Failure(RowFailure::InvalidOutput(error)),
                }
            }
            ProtocolRowOutcome::Failure(failure) => RowOutcome::Failure(RowFailure::Model(failure)),
        })
        .collect()
}

impl Evaluator for GraphcalEvaluator {
    type Error = ClientError;

    fn schema(&self) -> &Arc<SessionSchema> {
        &self.schema
    }

    fn evaluate(
        &self,
        requests: Vec<EvalRequest>,
    ) -> Box<dyn Iterator<Item = Result<ChunkResult, Self::Error>> + '_> {
        Box::new(
            requests
                .into_iter()
                .map(|request| self.evaluate_request(request)),
        )
    }
}

#[cfg(test)]
mod tests {
    use graphcal_model_protocol::{
        Dimension, FiniteF64, QuantityType, ResultBatchBuilder, RowFailure as ProtocolRowFailure,
        Value,
    };

    use super::*;
    use crate::ModelSchemaBuilder;

    #[test]
    fn protocol_outcomes_become_tenax_outcomes() {
        let schema = ModelSchemaBuilder::new()
            .continuous("x", 0.0, 1.0)
            .unwrap()
            .quantity_output(
                "margin",
                QuantityType::new(Dimension::dimensionless(), None).unwrap(),
            )
            .unwrap()
            .boolean_output("failure")
            .unwrap()
            .build()
            .unwrap();
        let mut builder = ResultBatchBuilder::new(Arc::clone(&schema));
        builder
            .push(ProtocolRowOutcome::Success(vec![
                Value::Quantity(FiniteF64::new(2.5).unwrap()),
                Value::Bool(true),
            ]))
            .unwrap();
        let failure = ProtocolRowFailure::AssertionFailed {
            assertion: "x < 1".to_owned(),
            message: "too large".to_owned(),
        };
        builder
            .push(ProtocolRowOutcome::Failure(failure.clone()))
            .unwrap();
        let batch = builder.finish().unwrap();

        let rows = result_rows(&schema, &batch);
        assert_eq!(
            rows,
            [
                RowOutcome::Success(
                    OutputRow::new(
                        &schema,
                        vec![
                            Value::Quantity(FiniteF64::new(2.5).unwrap()),
                            Value::Bool(true)
                        ]
                    )
                    .unwrap()
                ),
                RowOutcome::Failure(RowFailure::Model(failure))
            ]
        );
    }
}
