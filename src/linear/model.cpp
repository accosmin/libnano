#include <iomanip>
#include <nano/linear/util.h>
#include <nano/linear/model.h>
#include <nano/tensor/stream.h>
#include <nano/linear/function.h>

using namespace nano;

linear_model_t::linear_model_t()
{
    model_t::register_param(iparam1_t{"linear::batch", 1, LE, 32, LE, 4096});
    model_t::register_param(sparam1_t{"linear::l1reg", 0, LE, 0, LE, 1e+10});
    model_t::register_param(sparam1_t{"linear::l2reg", 0, LE, 0, LE, 1e+10});
    model_t::register_param(sparam1_t{"linear::vAreg", 0, LE, 0, LE, 1e+10});
    model_t::register_param(eparam1_t{"linear::scaling", feature_scaling::standard});
}

rmodel_t linear_model_t::clone() const
{
    return std::make_unique<linear_model_t>(*this);
}

scalar_t linear_model_t::fit(
    const loss_t& loss, const dataset_t& dataset, const indices_t& samples, const solver_t& solver)
{
    log_info() << string_t(8, '-') << ::nano::align(" gboost model ", 112U, alignment::left, '-') << string_t(8, '-');
    for (const auto& param : params())
    {
        log_info() << "gboost model: fit using " << std::fixed << std::setprecision(8) << param;
    }
    log_info() << string_t(128, '-');

    for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
    {
        const auto feature = dataset.feature(ifeature);
        critical(
            feature.discrete() || feature.optional(),
            "linear model: cannot fit datasets containing discrete features or with missing feature values!");
    }

    auto function = linear_function_t{loss, dataset, samples};
    function.batch(batch());
    function.l1reg(l1reg());
    function.l2reg(l2reg());
    function.vAreg(vAreg());
    function.scaling(scaling());

    const auto state = solver.minimize(function, vector_t::Zero(function.size()));
    m_bias = function.bias(state.x);
    m_weights = function.weights(state.x);

    // NB: rescale the bias and the weights to match the scaling of the inputs!
    const auto& istats = function.istats();
    istats.upscale(function.scaling(), m_weights, m_bias);

    tensor1d_t errors(samples.size());
    tensor4d_t outputs(cat_dims(samples.size(), dataset.tdims()));

    loopr(samples.size(), batch(), [&] (tensor_size_t begin, tensor_size_t end, size_t)
    {
        const auto range = make_range(begin, end);
        const auto inputs = dataset.inputs(samples.slice(range));
        const auto targets = dataset.targets(samples.slice(range));

        ::nano::linear::predict(inputs, m_weights, m_bias, outputs.slice(range));
        loss.error(targets.tensor(), outputs.slice(range), errors.slice(range));
    });

    const auto tr_value = state.f;
    const auto tr_error = errors.vector().mean();

    log_info() << std::setprecision(8) << std::fixed
        << "linear: tr=" << tr_value << "|" << tr_error
        << ",l1reg=" << function.l1reg()
        << ",l2reg=" << function.l2reg()
        << ",vAreg=" << function.vAreg() << "," << state << ".";

    return tr_error;
}

void linear_model_t::read(std::istream& stream)
{
    model_t::read(stream);

    critical(
        !::nano::read(stream, m_bias) ||
        !::nano::read(stream, m_weights),
        "linear model: failed to read from stream!");

    critical(m_bias.size() != m_weights.cols(),
        "linear model: parameter mismatch!");
}

void linear_model_t::write(std::ostream& stream) const
{
    model_t::write(stream);

    critical(
        !::nano::write(stream, m_bias) ||
        !::nano::write(stream, m_weights),
        "linear model: failed to write to stream!");
}

tensor4d_t linear_model_t::predict(const dataset_t& dataset, const indices_t& samples) const
{
    tensor4d_t outputs(cat_dims(samples.size(), dataset.tdims()));

    loopr(samples.size(), batch(), [&] (tensor_size_t begin, tensor_size_t end, size_t)
    {
        const auto range = make_range(begin, end);
        const auto inputs = dataset.inputs(samples.slice(range));

        ::nano::linear::predict(inputs, m_weights, m_bias, outputs.slice(range));
    });

    return outputs;
}
