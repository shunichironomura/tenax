//! Configuring Graphcal sessions whose varied parameters Tenax samples.

use graphcal_model_client::{
    Bool, ClientError, Datetime, Int, Key, Model, Quantity, SessionBuilder,
};

use crate::sampled_parameter::{SampledKind, SampledParameter};

/// Varies each of `parameters` in `session`, a session of `model`, through
/// the typed handle of its [`SampledKind`].
///
/// Pass the parameters [`crate::sampled_parameters`] returns for
/// `model.interface()` to vary every parameter of the model, then choose the
/// outputs. Going through typed handles keeps Graphcal's check that each
/// parameter has the kind it is drawn as.
///
/// ```no_run
/// use graphcal_model_client::{GraphcalCommand, Model};
/// use tenax::{
///     Evaluator, GraphcalEvaluator, sample_latin_hypercube, sampled_parameters,
///     vary_sampled_parameters,
/// };
///
/// let model = Model::describe(GraphcalCommand::new("model.gcl"))?;
/// let parameters = sampled_parameters(model.interface())?;
/// let session = vary_sampled_parameters(&model, model.session(), &parameters)?
///     .output_all_supported()
///     .spawn()?;
/// let evaluator = GraphcalEvaluator::new(session);
/// let request = sample_latin_hypercube(evaluator.schema(), 1_000, 42, 0)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`ClientError`] when `model` has no parameter of a given name or
/// it is not of the given kind, as for parameters taken from another model's
/// interface.
pub fn vary_sampled_parameters<'m>(
    model: &'m Model,
    session: SessionBuilder<'m>,
    parameters: &[SampledParameter],
) -> Result<SessionBuilder<'m>, ClientError> {
    parameters.iter().try_fold(session, |session, parameter| {
        let name = parameter.name().as_str();
        Ok(match parameter.kind() {
            SampledKind::Quantity => session.vary(&model.input::<Quantity>(name)?),
            SampledKind::Datetime => session.vary(&model.input::<Datetime>(name)?),
            SampledKind::Integer => session.vary(&model.input::<Int>(name)?),
            SampledKind::Key => session.vary(&model.input::<Key>(name)?),
            SampledKind::Boolean => session.vary(&model.input::<Bool>(name)?),
        })
    })
}
