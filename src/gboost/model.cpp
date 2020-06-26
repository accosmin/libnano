#include <fstream>
#include <nano/tune.h>
#include <nano/logger.h>
#include <nano/version.h>
#include <nano/gboost/bias.h>
#include <nano/gboost/util.h>
#include <nano/gboost/model.h>
#include <nano/gboost/scale.h>
#include <nano/tensor/stream.h>
#include <nano/dataset/shuffle.h>

using namespace nano;

namespace
{
    void evaluate(const dataset_t& dataset, fold_t fold, const tensor_size_t batch,
        const loss_t& loss, const tensor4d_t& outputs, tensor1d_map_t&& errors)
    {
        assert(errors.size() == dataset.samples(fold));
        assert(outputs.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

        dataset.loop(execution::par, fold, batch, [&] (tensor_range_t range, size_t)
        {
            const auto targets = dataset.targets(fold, range);
            loss.error(targets, outputs.slice(range), errors.slice(range));
        });
    }

    void residual(const dataset_t& dataset, fold_t fold, const tensor_size_t batch,
        const loss_t& loss, const tensor4d_t& outputs, const scalar_t vAreg,
        tensor1d_map_t&& values, tensor4d_map_t&& vgrads, tensor4d_map_t&& vgrads2)
    {
        assert(vgrads.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));
        assert(outputs.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

        dataset.loop(execution::par, fold, batch, [&] (tensor_range_t range, const size_t)
        {
            const auto targets = dataset.targets(fold, range);
            loss.vgrad(targets, outputs.slice(range), vgrads.slice(range));
            if (vAreg > 0.0)
            {
                loss.value(targets, outputs.slice(range), values.slice(range));
                for (tensor_size_t i = range.begin(); i < range.end(); ++ i)
                {
                    vgrads2.vector(i) = values(i) * vgrads.vector(i);
                }
            }
        });

        if (vAreg > 0.0)
        {
            const auto mean_value = values.vector().mean();
            for (tensor_size_t i = 0, size = values.size(); i < size; ++ i)
            {
                vgrads.vector(i) = vgrads.vector(i) + 2.0 * vAreg * (vgrads2.vector(i) - mean_value * vgrads.vector(i));
            }
        }
    }

    template <size_t trank>
    void resize_all(const tensor_dims_t<trank>&)
    {
    }

    template <size_t trank, typename ttensor, typename... ttensors>
    void resize_all(const tensor_dims_t<trank>& dims, ttensor& tensor, ttensors&... tensors)
    {
        tensor.resize(dims);
        resize_all(dims, tensors...);
    }
}

void gboost_model_t::add(const string_t& id)
{
    add(id, wlearner_t::all().get(id));
}

void gboost_model_t::add(string_t id, rwlearner_t&& prototype)
{
    critical(prototype == nullptr, "gboost model: invalid prototype weak learner!");

    m_protos.emplace_back(std::move(id), std::move(prototype));
}

train_result_t gboost_model_t::train(const loss_t& loss, const dataset_t& dataset, const solver_t& solver)
{
    critical(m_protos.empty(), "gboost model: no prototype weak learners to use!");

    log_info() << "training gboost model:";
    log_info()
        << " === batch=" << batch() << ", rounds=" << rounds() << ", patience=" << patience()
        << ", subsample=" << subsample() << ", tune=" << tune_steps() << "x" << tune_trials();
    log_info()
        << " === scale=" << scale() << ", regularization=" << scat(regularization());
    for (const auto& proto : m_protos)
    {
        log_info() << " === proto: id=" << proto.m_id << ", type=" << proto.m_wlearner->type();
    }

    tensor4d_t te_avg_outputs;

    const auto tdim = dataset.tdim();

    m_models.clear();

    // train a model for each fold ...
    train_result_t results;
    for (size_t fold = 0, folds = dataset.folds(); fold < folds; ++ fold)
    {
        auto& result = results.add();

        const auto te_fold = fold_t{fold, protocol::test};
        const auto te_samples = dataset.samples(te_fold);

        tensor1d_t te_errors(te_samples);

        if (fold == 0)
        {
            te_avg_outputs.resize(cat_dims(te_samples, tdim));
            te_avg_outputs.zero();
        }

        // tune the regularization factors (if any)
        model_t model;
        tensor4d_t te_outputs;
        scalar_t vAreg = 0.0;
        scalar_t vd_error = std::numeric_limits<scalar_t>::max();

        switch (regularization())
        {
        case ::nano::regularization::variance:
            std::tie(vAreg, vd_error, model, te_outputs) = nano::grid_tune(
                nano::pow10_space_t{-6.0, +6.0},
                [&] (const scalar_t vAreg)
                {
                    auto& curve = result.add(scat("vAreg=", vAreg));
                    assert(curve.size() == 0);
                    return train(loss, dataset, fold, solver, vAreg, curve);
                },
                tune_trials(), tune_steps());
            break;

        case ::nano::regularization::none:
            {
                auto& curve = result.add(scat("noreg"));
                std::tie(vd_error, model, te_outputs) = train(loss, dataset, fold, solver, 0.0, curve);
            }
            break;

        case ::nano::regularization::lasso:
        case ::nano::regularization::ridge:
        case ::nano::regularization::elastic:
        default:
            critical(true, "unhandled regularization method when training gboost models");
        }

        m_models.emplace_back(std::move(model));

        // update predictions (of the averaged model)
        te_avg_outputs.vector() = (te_avg_outputs.vector() * fold + te_outputs.vector()) / (fold + 1);

        // update measurements
        ::evaluate(dataset, te_fold, batch(), loss, te_outputs, te_errors.tensor());
        result.test(te_errors.vector().mean());
        assert(std::fabs(vd_error - result.vd_error()) < 1e-8);

        ::evaluate(dataset, te_fold, batch(), loss, te_avg_outputs, te_errors.tensor());
        result.avg_test(te_errors.vector().mean());

        log_info() << std::setprecision(8) << std::fixed << ">>> "
            << "tr=" << result.tr_value() << "|" << result.tr_error()
            << ",vd=" << result.vd_error()
            << ",te=" << result.te_error() << ",avg_te=" << result.avg_te_error()
            << ",vA=" << vAreg << ".";
    }

    // OK
    return results;
}

train_status gboost_model_t::done(const tensor_size_t round, const scalar_t vAreg,
    const tensor1d_t& tr_errors, const tensor1d_t& vd_errors, const solver_state_t& state,
    const indices_t& indices, train_curve_t& curve) const
{
    const auto cwidth = static_cast<int>(std::log10(rounds())) + 1;

    const auto tr_value = state.f;
    const auto tr_error = tr_errors.vector().mean();
    const auto vd_error = vd_errors.vector().mean();

    curve.add(tr_value, tr_error, vd_error);
    const auto status = curve.check(patience());

    log_info()
        << std::setprecision(4) << std::fixed
        << std::setw(cwidth) << std::setfill('0') << round << "/"
        << std::setw(cwidth) << std::setfill('0') << rounds()
        << ":tr=" << tr_value << "|" << tr_error << ",vd=" << vd_error << "(" << status << ")"
        << std::setprecision(8) << std::fixed
        << ",vAreg=" << vAreg << "," << state
        << ",feat=[" << indices.array() << "].";

    return status;
}

indices_t gboost_model_t::make_indices(tensor_size_t samples) const
{
    if (subsample() >= 100)
    {
        return ::nano::arange(0, samples);
    }
    else
    {
        return ::nano::sample_without_replacement(samples, subsample());
    }
}

std::tuple<scalar_t, gboost_model_t::model_t, tensor4d_t> gboost_model_t::train(
    const loss_t& loss, const dataset_t& dataset, size_t fold, const solver_t& solver, scalar_t vAreg,
    train_curve_t& curve) const
{
    const auto tdim = dataset.tdim();

    const auto tr_fold = fold_t{fold, protocol::train};
    const auto vd_fold = fold_t{fold, protocol::valid};
    const auto te_fold = fold_t{fold, protocol::test};

    const auto tr_samples = dataset.samples(tr_fold);
    const auto vd_samples = dataset.samples(vd_fold);
    const auto te_samples = dataset.samples(te_fold);

    tensor1d_t tr_errors, vd_errors;
    tensor4d_t tr_outputs, tr_woutputs, tr_vgrads, vd_outputs, vd_woutputs, te_outputs, te_woutputs, te_opt_outputs;

    ::resize_all(cat_dims(tr_samples, tdim), tr_outputs, tr_woutputs, tr_vgrads);
    ::resize_all(cat_dims(vd_samples, tdim), vd_outputs, vd_woutputs);
    ::resize_all(cat_dims(te_samples, tdim), te_outputs, te_woutputs, te_opt_outputs);

    ::resize_all(make_dims(tr_samples), tr_errors);
    ::resize_all(make_dims(vd_samples), vd_errors);

    // estimate bias on the current fold
    auto function = gboost_bias_function_t{loss, dataset, tr_fold};
    function.vAreg(vAreg);
    function.batch(batch());

    const auto state = solver.minimize(function, vector_t::Zero(function.size()));

    // update predictions
    tr_outputs.reshape(tr_samples, -1).matrix().rowwise() = state.x.transpose();
    vd_outputs.reshape(vd_samples, -1).matrix().rowwise() = state.x.transpose();
    te_outputs.reshape(te_samples, -1).matrix().rowwise() = state.x.transpose();

    // update measurements
    ::evaluate(dataset, tr_fold, batch(), loss, tr_outputs, tr_errors.tensor());
    ::evaluate(dataset, vd_fold, batch(), loss, vd_outputs, vd_errors.tensor());

    const auto status = done(0, vAreg, tr_errors, vd_errors, state, indices_t{}, curve);
    critical(status == train_status::diverged, "gboost model: failed to fit bias (check inputs and parameters)!");

    model_t model;
    model.m_bias.resize(state.x.size());
    model.m_bias.vector() = state.x;
    te_opt_outputs = te_outputs;

    cluster_t cluster(tr_samples, 1);
    for (tensor_size_t i = 0; i < tr_samples; ++ i)
    {
        cluster.assign(i, 0);
    }

    // construct the model one boosting round at a time
    for (tensor_size_t round = 0; round < rounds(); ++ round)
    {
        const auto indices = make_indices(tr_samples);

        ::residual(dataset, tr_fold, batch(), loss, tr_outputs, vAreg,
            tr_errors.tensor(), tr_vgrads.tensor(), tr_woutputs.tensor());

        // choose the weak learner that aligns the best with the current residuals
        auto best_id = std::string{};
        auto best_score = std::numeric_limits<scalar_t>::max();
        auto best_wlearner = rwlearner_t{};
        for (const auto& prototype : m_protos)
        {
            auto wlearner = prototype.m_wlearner->clone();
            assert(wlearner);

            const auto score = wlearner->fit(dataset, tr_fold, tr_vgrads, indices);
            if (score < best_score)
            {
                best_id = prototype.m_id;
                best_score = score;
                best_wlearner = std::move(wlearner);
            }
        }

        if (!best_wlearner)
        {
            log_warning() << "cannot fit any new weak learner, stopping.";
            break;
        }
        const auto features = best_wlearner->features();

        // scale the chosen weak learner
        switch (scale())
        {
        case wscale::tboost:
            cluster = best_wlearner->split(dataset, tr_fold, ::nano::arange(0, tr_samples));
            break;

        default:
            break;
        }

        best_wlearner->predict(dataset, tr_fold, make_range(0, tr_samples), tr_woutputs.tensor());

        auto function = gboost_scale_function_t(loss, dataset, tr_fold, cluster, tr_outputs, tr_woutputs);
        function.vAreg(vAreg);
        function.batch(batch());

        const auto state = solver.minimize(function, vector_t::Zero(function.size()));
        if (state.x.minCoeff() < 0.0)
        {
            log_warning() << "invalid scale factor(s): [" << state.x.transpose() << "], stopping.";
            break;
        }

        best_wlearner->scale(state.x);
        best_wlearner->predict(dataset, tr_fold, make_range(0, tr_samples), tr_woutputs.tensor());
        best_wlearner->predict(dataset, vd_fold, make_range(0, vd_samples), vd_woutputs.tensor());
        best_wlearner->predict(dataset, te_fold, make_range(0, te_samples), te_woutputs.tensor());

        tr_outputs.vector() += tr_woutputs.vector();
        vd_outputs.vector() += vd_woutputs.vector();
        te_outputs.vector() += te_woutputs.vector();

        model.m_protos.emplace_back(std::move(best_id), std::move(best_wlearner));

        // update measurements
        ::evaluate(dataset, tr_fold, batch(), loss, tr_outputs, tr_errors.tensor());
        ::evaluate(dataset, vd_fold, batch(), loss, vd_outputs, vd_errors.tensor());

        const auto status = done(round + 1, vAreg, tr_errors, vd_errors, state, features, curve);
        if (status == train_status::better)
        {
            te_opt_outputs = te_outputs;
        }
        else if (status == train_status::overfit || status == train_status::diverged)
        {
            break;
        }
    }

    model.m_protos.erase(model.m_protos.begin() + curve.optindex(), model.m_protos.end());

    return std::make_tuple(curve.optimum().vd_error(), std::move(model), std::move(te_opt_outputs));
}

void gboost_model_t::predict(const dataset_t& dataset, fold_t fold, tensor4d_t& outputs) const
{
    outputs.resize(cat_dims(dataset.samples(fold), dataset.tdim()));
    predict(dataset, fold, outputs.tensor());
}

void gboost_model_t::predict(const dataset_t& dataset, fold_t fold, tensor4d_map_t&& outputs) const
{
    assert(outputs.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    dataset.loop(execution::par, fold, batch(), [&] (tensor_range_t range, size_t)
    {
        tensor4d_t toutputs(cat_dims(range.size(), dataset.tdim()));
        tensor4d_t woutputs(cat_dims(range.size(), dataset.tdim()));

        toutputs.zero();
        for (const auto& model : m_models)
        {
            toutputs.reshape(range.size(), -1).matrix().rowwise() += model.m_bias.vector().transpose();

            for (const auto& proto : model.m_protos)
            {
                const auto& wlearner = proto.m_wlearner;
                wlearner->predict(dataset, fold, range, woutputs.tensor());
                toutputs.vector() += woutputs.vector();
            }
        }

        outputs.slice(range).vector() = toutputs.vector();
    });

    outputs.array() /= std::max(size_t(1), m_models.size());
}

gboost_model_t::proto_t::proto_t(string_t&& id, rwlearner_t&& wlearner) :
    m_id(std::move(id)),
    m_wlearner(std::move(wlearner))
{
}

void gboost_model_t::proto_t::read(std::istream& stream)
{
    critical(
        !::nano::detail::read(stream, m_id),
        "gboost model: failed to read from stream!");

    m_wlearner = wlearner_t::all().get(m_id);
    critical(
        m_wlearner == nullptr,
        scat("gboost model: invalid weak learner id <", m_id, "> read from stream!"));

    m_wlearner->read(stream);
}

void gboost_model_t::proto_t::write(std::ostream& stream) const
{
    critical(
        !::nano::detail::write(stream, m_id),
        "gboost model: failed to write to stream!");

    m_wlearner->write(stream);
}

void gboost_model_t::read(std::istream& stream)
{
    serializable_t::read(stream);

    int32_t ibatch = 0, irounds = 0, ipatience = 0, isubsample = 0, itune_trials = 0, itune_steps = 0;
    int32_t iscale = 0, iregularization = 0;

    critical(
        !::nano::detail::read(stream, ibatch) ||
        !::nano::detail::read(stream, irounds) ||
        !::nano::detail::read(stream, ipatience) ||
        !::nano::detail::read(stream, isubsample) ||
        !::nano::detail::read(stream, itune_trials) ||
        !::nano::detail::read(stream, itune_steps) ||
        !::nano::detail::read(stream, iscale) ||
        !::nano::detail::read(stream, iregularization),
        "gboost model: failed to read from stream!");

    batch(ibatch);
    rounds(irounds);
    patience(ipatience);
    subsample(isubsample);
    tune_steps(itune_steps);
    tune_trials(itune_trials);
    scale(static_cast<::nano::wscale>(iscale));
    regularization(static_cast<::nano::regularization>(iregularization));

    uint32_t size = 0;
    critical(
        !::nano::detail::read(stream, size),
        "gboost model: failed to read from stream!");

    m_protos.resize(size);
    for (auto& proto : m_protos)
    {
        proto.read(stream);
    }

    critical(
        !::nano::detail::read(stream, size),
        "gboost model: failed to read from stream!");

    m_models.resize(size);
    for (auto& model : m_models)
    {
        critical(
            !::nano::read(stream, model.m_bias) ||
            !::nano::detail::read(stream, size),
            "gboost model: failed to read from stream!");

        model.m_protos.resize(size);
        for (auto& proto : model.m_protos)
        {
            proto.read(stream);
        }
    }
}

void gboost_model_t::write(std::ostream& stream) const
{
    serializable_t::write(stream);

    critical(
        !::nano::detail::write(stream, static_cast<int32_t>(batch())) ||
        !::nano::detail::write(stream, static_cast<int32_t>(rounds())) ||
        !::nano::detail::write(stream, static_cast<int32_t>(patience())) ||
        !::nano::detail::write(stream, static_cast<int32_t>(subsample())) ||
        !::nano::detail::write(stream, static_cast<int32_t>(tune_trials())) ||
        !::nano::detail::write(stream, static_cast<int32_t>(tune_steps())) ||
        !::nano::detail::write(stream, static_cast<int32_t>(m_scale)) ||
        !::nano::detail::write(stream, static_cast<int32_t>(m_regularization)),
        "gboost model: failed to write to stream!");

    critical(
        !::nano::detail::write(stream, static_cast<uint32_t>(m_protos.size())),
        "gboost model: failed to write to stream!");

    for (const auto& proto : m_protos)
    {
        proto.write(stream);
    }

    critical(
        !::nano::detail::write(stream, static_cast<uint32_t>(m_models.size())),
        "gboost model: failed to write to stream!");

    for (const auto& model : m_models)
    {
        critical(
            !::nano::write(stream, model.m_bias) ||
            !::nano::detail::write(stream, static_cast<uint32_t>(model.m_protos.size())),
            "gboost model: failed to write to stream!");

        for (const auto& proto : model.m_protos)
        {
            proto.write(stream);
        }
    }
}

feature_infos_t gboost_model_t::features(const loss_t& loss, const dataset_t& dataset, const tensor_size_t trials) const
{
    // construct the list of ALL selected features (across folds)
    std::vector<tensor_size_t> features;
    std::map<tensor_size_t, std::set<size_t>> ffcounts;
    for (size_t imodel = 0; imodel < m_models.size(); ++ imodel)
    {
        const auto& model = m_models[imodel];

        for (const auto& proto : model.m_protos)
        {
            for (const auto feature : proto.m_wlearner->features())
            {
                features.push_back(feature);
                ffcounts[feature].insert(imodel);
            }
        }
    }

    std::sort(features.begin(), features.end());
    features.erase(std::unique(features.begin(), features.end()), features.end());

    // estimate the importance of EACH of the selected features
    std::vector<scalar_t> importances(features.size(), 0.0);
    for (size_t ifold = 0, folds = dataset.folds(); ifold < folds; ++ ifold)
    {
        const auto fold = fold_t{ifold, protocol::valid};
        const auto targets = dataset.targets(fold);

        tensor1d_t errors;
        tensor4d_t outputs;

        const auto evaluate = [&] (const auto& xdataset)
        {
            predict(xdataset, fold, outputs);
            loss.error(targets, outputs, errors);
            return errors.vector().mean();
        };

        // baseline error rate
        const auto baseline_error = evaluate(dataset);
        log_info() << std::fixed << std::setprecision(6)
            << "gboost model: baseline error[" << (ifold + 1) << "|" << folds << "]=" << baseline_error << ".";

        // estimate the impact of shuffling each feature at a time on the error rate
        for (size_t ifeature = 0; ifeature < features.size(); ++ ifeature)
        {
            const auto feature = features[ifeature];
            const auto fdataset = shuffle_dataset_t{dataset, feature};

            scalar_t feature_error = 0.0;
            for (tensor_size_t trial = 0; trial < trials; ++ trial)
            {
                feature_error += evaluate(fdataset) / trials;
            }

            const auto importance = feature_error - baseline_error;
            log_info() << std::fixed << std::setprecision(6)
                << "gboost model: feature[" << feature << "] error[" << (ifold + 1) << "|" << folds
                << "]=" << feature_error << " vs. " << baseline_error << " => importance=" << importance << ".";

            importances[ifeature] += importance;
        }
    }

    // assembly stats per feature
    feature_infos_t infos;
    for (size_t ifeature = 0; ifeature < features.size(); ++ ifeature)
    {
        const auto feature = features[ifeature];
        const auto importance = importances[ifeature];
        const auto folds = static_cast<tensor_size_t>(ffcounts[feature].size());
        infos.emplace_back(feature, folds, importance);
    }
    feature_info_t::sort_by_importance(infos);
    return infos;
}
