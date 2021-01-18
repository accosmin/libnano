#include <iomanip>
#include <nano/mlearn/util.h>
#include <nano/mlearn/kfold.h>
#include <nano/model/grid_search.h>

using namespace nano;

grid_search_model_t::grid_search_model_t(
    const string_t& model_id, param_grid_t grid) :
    grid_search_model_t(model_id, model_t::all().get(model_id), std::move(grid))
{
}

grid_search_model_t::grid_search_model_t(
    string_t model_id, rmodel_t model, param_grid_t grid) :
    m_imodel(std::move(model_id), std::move(model)),
    m_grid(std::move(grid))
{
    critical(
        !m_imodel,
        "grid-search model: invalid prototype model with id (", m_imodel.id(), ")!");

    critical(
        m_grid.empty(),
        "grid-search model: empty hyper-parameter grid!");

    for (const auto& pvalues : m_grid)
    {
        critical(
            pvalues.second.valueless_by_exception() ||
            (
                std::holds_alternative<std::vector<int64_t>>(pvalues.second) ?
                std::get<std::vector<int64_t>>(pvalues.second).empty() :
                std::get<std::vector<scalar_t>>(pvalues.second).empty()
            ),
            "grid-search model: empty hyper-parameter config (", pvalues.first, ")!");

        if (std::holds_alternative<std::vector<int64_t>>(pvalues.second))
        {
            for (const auto value : std::get<std::vector<int64_t>>(pvalues.second))
            {
                m_imodel.get().set(pvalues.first, value);
            }
        }
        else
        {
            for (const auto value : std::get<std::vector<scalar_t>>(pvalues.second))
            {
                m_imodel.get().set(pvalues.first, value);
            }
        }
    }

    model_t::register_param(iparam1_t{"grid-search::folds", 3, LE, 5, LE, 100});
    model_t::register_param(iparam1_t{"grid-search::max_trials", 10, LE, 100, LE, 1000});
}

void grid_search_model_t::read(std::istream& stream)
{
    model_t::read(stream);

    m_imodel.read(stream);
}

void grid_search_model_t::write(std::ostream& stream) const
{
    model_t::write(stream);

    m_imodel.write(stream);
}

rmodel_t grid_search_model_t::clone() const
{
    return std::make_unique<grid_search_model_t>(*this);
}

grid_search_model_t::count_config_t grid_search_model_t::make_counts()const
{
    count_config_t counts(static_cast<tensor_size_t>(m_grid.size()));
    tensor_size_t i = 0;
    for (const auto& pvalues : m_grid)
    {
        if (std::holds_alternative<std::vector<int64_t>>(pvalues.second))
        {
            counts(i ++) = std::get<std::vector<int64_t>>(pvalues.second).size();
        }
        else
        {
            counts(i ++) = std::get<std::vector<scalar_t>>(pvalues.second).size();
        }
    }

    return counts;
}

model_config_t grid_search_model_t::make_config(const count_config_t& indices) const
{
    model_config_t config;

    tensor_size_t iparam = 0;
    for (const auto& pvalues : m_grid)
    {
        const auto ivalue = indices(iparam ++);
        if (std::holds_alternative<std::vector<int64_t>>(pvalues.second))
        {
            const auto& values = std::get<std::vector<int64_t>>(pvalues.second);
            assert(ivalue < values.size());
            config.add(pvalues.first, values[ivalue]);
        }
        else
        {
            const auto& values = std::get<std::vector<scalar_t>>(pvalues.second);
            assert(ivalue < values.size());
            config.add(pvalues.first, values[ivalue]);
        }
    }

    return config;
}

scalar_t grid_search_model_t::fit(
    const loss_t& loss, const dataset_t& dataset, const indices_t& samples, const solver_t& solver)
{
    critical(
        !m_imodel,
        "grid-search model: invalid prototype model with id (", m_imodel.id(), ")!");

    // initialize the hyper-parameter configurations to evaluate
    const auto max_trials = static_cast<tensor_size_t>(this->max_trials());

    auto combinatrix = combinatorial_iterator_t{make_counts()};

    const auto combinations = (combinatrix.size() <= max_trials) ?
        arange(0, combinatrix.size()) :
        sample_without_replacement(combinatrix.size(), max_trials);

    // evaluate the hyper-parameter configurations using cross-validation
    const auto folds = this->folds();
    const auto kfold = kfold_t{samples, folds};

    log_info() << "grid-search model: evaluating " << combinations.size()
        << "/" << combinatrix.size() << " hyper-parameter configurations using " << folds << " folds.";

    auto& model = m_imodel.get();

    m_configs.clear();
    for (tensor_size_t iconfig = 0; combinatrix && iconfig < combinations.size(); )
    {
        while (combinatrix.index() < combinations(iconfig))
        {
            ++ combinatrix;
        }
        ++ iconfig;

        auto config = make_config(*combinatrix);
        model.set(config);

        log_info() << "grid-search model: evaluating configuration {" << config << "}...";

        tensor1d_t errors(folds);
        for (tensor_size_t fold = 0; fold < folds; ++ fold)
        {
            const auto [train_samples, valid_samples] = kfold.split(fold);

            model.fit(loss, dataset, train_samples, solver);
            errors(fold) = model.evaluate(loss, dataset, valid_samples).mean();
        }
        log_info() << std::fixed << std::setprecision(8) << ">>> done, with validation error=" << errors.mean() << ".";

        config.evaluate(errors.mean());
        m_configs.emplace_back(std::move(config));
    }

    // retrain with the best hyper-parameter configuration using both training and validation samples
    const auto compare = [] (const auto& lhs, const auto& rhs) { return lhs.error() < rhs.error(); };
    std::sort(m_configs.begin(), m_configs.end(), compare);

    model.set(*m_configs.begin());
    return model.fit(loss, dataset, samples, solver);
}

tensor4d_t grid_search_model_t::predict(const dataset_t& dataset, const indices_t& samples) const
{
    critical(
        !m_imodel,
        "grid-search model: invalid prototype model with id (", m_imodel.id(), ")!");

    return m_imodel.get().predict(dataset, samples);
}
