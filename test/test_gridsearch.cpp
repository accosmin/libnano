#include "fixture/utils.h"
#include "fixture/memfixed.h"
#include <nano/model/grid_search.h>

using namespace nano;

class fixture_model_t;

namespace nano
{
    template <>
    struct factory_traits_t<fixture_model_t>
    {
        static string_t id() { return "fixture"; }
        static string_t description() { return "description"; }
    };
}

class fixture_model_t final : public model_t
{
public:

    fixture_model_t()
    {
        model_t::register_param(iparam1_t{"iparam1", 1, LE, 2, LE, 10});
        model_t::register_param(iparam1_t{"iparam2", 1, LE, 2, LE, 10});
        model_t::register_param(sparam1_t{"sparam1", 0.0, LE, 0.1, LE, 1.0});
    }

    rmodel_t clone() const override
    {
        return std::make_unique<fixture_model_t>(*this);
    }

    scalar_t fit(const loss_t&, const dataset_t& dataset, const indices_t& samples, const solver_t&) override
    {
        return predict(dataset, samples).mean();
    }

    tensor4d_t predict(const dataset_t& dataset, const indices_t& samples) const override
    {
        tensor4d_t outputs = dataset.targets(samples);
        outputs.array() += delta(iparam1(), iparam2(), sparam1());
        return outputs;
    }

    static scalar_t delta(int64_t iparam1, int64_t iparam2, scalar_t sparam1)
    {
        return static_cast<scalar_t>(10 * iparam1) + static_cast<scalar_t>(iparam2) + sparam1;
    }

    static scalar_t error(const dataset_t& dataset, int64_t iparam1, int64_t iparam2, scalar_t sparam1)
    {
        return delta(iparam1, iparam2, sparam1) * static_cast<scalar_t>(nano::size(dataset.tdims()));
    }

    int64_t iparam1() const { return ivalue("iparam1"); }
    int64_t iparam2() const { return ivalue("iparam2"); }
    scalar_t sparam1() const { return svalue("sparam1"); }
};

static void check_predict(const model_t& model, const dataset_t& dataset, scalar_t delta)
{
    const auto samples = dataset.train_samples();
    const auto targets = dataset.targets(samples);

    tensor4d_t outputs;
    UTEST_CHECK_NOTHROW(outputs = model.predict(dataset, dataset.train_samples()));
    UTEST_CHECK_EQUAL(outputs.dims(), targets.dims());
    UTEST_CHECK_EIGEN_CLOSE(outputs.array(), targets.array() + delta, 1e-12);
}

static auto check_stream(const model_t& model)
{
    string_t str;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(model.write(stream));
        str = stream.str();
    }
    {
        grid_search_model_t xmodel;
        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(xmodel.read(stream));
        return xmodel;
    }
}

UTEST_BEGIN_MODULE(test_gridsearch)

UTEST_CASE(init)
{
    model_t::all().add_by_type<fixture_model_t>();

    const auto grid_NO1 = param_grid_t{};
    const auto grid_NO2 = param_grid_t{{"iparam2", std::vector<int64_t>{}}};
    const auto grid_NO3 = param_grid_t{{"iparamX", std::vector<int64_t>{1}}};
    const auto grid_NO4 = param_grid_t{{"sparam1", std::vector<scalar_t>{0.1, 1.1}}};

    const auto grid_OK1 = param_grid_t{
        {"iparam1", std::vector<int64_t>{1, 3}},
        {"iparam2", std::vector<int64_t>{1, 2, 3, 5}},
        {"sparam1", std::vector<scalar_t>{0.0, 0.1, 0.9, 1.0}}
    };

    const auto model = fixture_model_t{};

    UTEST_CHECK(model_t::all().get("fixture") != nullptr);

    const auto make1 = [=] (const param_grid_t& grid) { return grid_search_model_t{model, grid}; };
    const auto make2 = [=] (const param_grid_t& grid) { return grid_search_model_t{"fixture", grid}; };
    const auto make3 = [=] (const param_grid_t& grid) { return grid_search_model_t{"fixture", model.clone(), grid}; };
    const auto make4 = [=] (const param_grid_t& grid) { return grid_search_model_t{"fixture", nullptr, grid}; };
    const auto make5 = [=] (const param_grid_t& grid) { return grid_search_model_t{"invalid_model_id", grid}; };

    UTEST_CHECK_THROW(make1(grid_NO1), std::runtime_error);
    UTEST_CHECK_THROW(make1(grid_NO2), std::runtime_error);
    UTEST_CHECK_THROW(make1(grid_NO3), std::runtime_error);
    UTEST_CHECK_THROW(make1(grid_NO4), std::runtime_error);
    UTEST_CHECK_NOTHROW(make1(grid_OK1));

    UTEST_CHECK_THROW(make2(grid_NO1), std::runtime_error);
    UTEST_CHECK_THROW(make2(grid_NO2), std::runtime_error);
    UTEST_CHECK_THROW(make2(grid_NO3), std::runtime_error);
    UTEST_CHECK_THROW(make2(grid_NO4), std::runtime_error);
    UTEST_CHECK_NOTHROW(make2(grid_OK1));

    UTEST_CHECK_THROW(make3(grid_NO1), std::runtime_error);
    UTEST_CHECK_THROW(make3(grid_NO2), std::runtime_error);
    UTEST_CHECK_THROW(make3(grid_NO3), std::runtime_error);
    UTEST_CHECK_THROW(make3(grid_NO4), std::runtime_error);
    UTEST_CHECK_NOTHROW(make3(grid_OK1));

    UTEST_CHECK_THROW(make4(grid_NO1), std::runtime_error);
    UTEST_CHECK_THROW(make4(grid_NO2), std::runtime_error);
    UTEST_CHECK_THROW(make4(grid_NO3), std::runtime_error);
    UTEST_CHECK_THROW(make4(grid_NO4), std::runtime_error);
    UTEST_CHECK_THROW(make4(grid_OK1), std::runtime_error);

    UTEST_CHECK_THROW(make5(grid_NO1), std::runtime_error);
    UTEST_CHECK_THROW(make5(grid_NO2), std::runtime_error);
    UTEST_CHECK_THROW(make5(grid_NO3), std::runtime_error);
    UTEST_CHECK_THROW(make5(grid_NO4), std::runtime_error);
    UTEST_CHECK_THROW(make5(grid_OK1), std::runtime_error);
}

UTEST_CASE(empty)
{
    const auto loss = make_loss();
    const auto solver = make_solver();

    auto dataset = fixture_dataset_t{};
    dataset.resize(nano::make_dims(100, 1, 2, 3), nano::make_dims(100, 1, 5, 1));
    UTEST_REQUIRE_NOTHROW(dataset.load());

    auto gridsearch = grid_search_model_t{};
    UTEST_CHECK_THROW(gridsearch.predict(dataset, dataset.train_samples()), std::runtime_error);
}

UTEST_CASE(exhaustive)
{
    const auto grid = param_grid_t{
        {"iparam1", std::vector<int64_t>{1, 3}},
        {"iparam2", std::vector<int64_t>{1, 2, 5}},
        {"sparam1", std::vector<scalar_t>{0.2, 0.1, 0.9}}
    };

    const auto loss = make_loss();
    const auto solver = make_solver();
    const auto model = fixture_model_t{};

    auto dataset = fixture_dataset_t{};
    dataset.resize(nano::make_dims(100, 1, 2, 3), nano::make_dims(100, 1, 5, 1));
    UTEST_REQUIRE_NOTHROW(dataset.load());

    auto gridsearch = grid_search_model_t{model, grid};
    UTEST_CHECK_NOTHROW(gridsearch.folds(3));
    UTEST_CHECK_NOTHROW(gridsearch.max_trials(100));
    UTEST_CHECK_NOTHROW(gridsearch.fit(*loss, dataset, dataset.train_samples(), *solver));

    const auto& configs = gridsearch.configs();
    UTEST_REQUIRE_EQUAL(configs.size(), 18U);
    UTEST_CHECK_CLOSE(configs[0].error(), fixture_model_t::error(dataset, 1, 1, 0.1), 1e-12);
    UTEST_CHECK_CLOSE(configs[1].error(), fixture_model_t::error(dataset, 1, 1, 0.2), 1e-12);
    UTEST_CHECK_CLOSE(configs[2].error(), fixture_model_t::error(dataset, 1, 1, 0.9), 1e-12);
    UTEST_CHECK_CLOSE(configs[3].error(), fixture_model_t::error(dataset, 1, 2, 0.1), 1e-12);
    UTEST_CHECK_CLOSE(configs[4].error(), fixture_model_t::error(dataset, 1, 2, 0.2), 1e-12);
    UTEST_CHECK_CLOSE(configs[5].error(), fixture_model_t::error(dataset, 1, 2, 0.9), 1e-12);
    UTEST_CHECK_CLOSE(configs[6].error(), fixture_model_t::error(dataset, 1, 5, 0.1), 1e-12);
    UTEST_CHECK_CLOSE(configs[7].error(), fixture_model_t::error(dataset, 1, 5, 0.2), 1e-12);
    UTEST_CHECK_CLOSE(configs[8].error(), fixture_model_t::error(dataset, 1, 5, 0.9), 1e-12);
    UTEST_CHECK_CLOSE(configs[9].error(), fixture_model_t::error(dataset, 3, 1, 0.1), 1e-12);
    UTEST_CHECK_CLOSE(configs[10].error(), fixture_model_t::error(dataset, 3, 1, 0.2), 1e-12);
    UTEST_CHECK_CLOSE(configs[11].error(), fixture_model_t::error(dataset, 3, 1, 0.9), 1e-12);
    UTEST_CHECK_CLOSE(configs[12].error(), fixture_model_t::error(dataset, 3, 2, 0.1), 1e-12);
    UTEST_CHECK_CLOSE(configs[13].error(), fixture_model_t::error(dataset, 3, 2, 0.2), 1e-12);
    UTEST_CHECK_CLOSE(configs[14].error(), fixture_model_t::error(dataset, 3, 2, 0.9), 1e-12);
    UTEST_CHECK_CLOSE(configs[15].error(), fixture_model_t::error(dataset, 3, 5, 0.1), 1e-12);
    UTEST_CHECK_CLOSE(configs[16].error(), fixture_model_t::error(dataset, 3, 5, 0.2), 1e-12);
    UTEST_CHECK_CLOSE(configs[17].error(), fixture_model_t::error(dataset, 3, 5, 0.9), 1e-12);

    UTEST_CHECK_EQUAL(gridsearch.model().ivalue("iparam1"), 1);
    UTEST_CHECK_EQUAL(gridsearch.model().ivalue("iparam2"), 1);
    UTEST_CHECK_CLOSE(gridsearch.model().svalue("sparam1"), 0.1, 1e-12);

    check_predict(gridsearch, dataset, fixture_model_t::delta(1, 1, 0.1));
    gridsearch = check_stream(gridsearch);
    check_predict(gridsearch, dataset, fixture_model_t::delta(1, 1, 0.1));
}

UTEST_CASE(max_trials)
{
    const auto grid = param_grid_t{
        {"iparam1", std::vector<int64_t>{1, 3}},
        {"iparam2", std::vector<int64_t>{1, 2, 5}},
        {"sparam1", std::vector<scalar_t>{0.2, 0.1, 0.9}}
    };

    const auto loss = make_loss();
    const auto solver = make_solver();
    const auto model = fixture_model_t{};

    auto dataset = fixture_dataset_t{};
    dataset.resize(nano::make_dims(100, 1, 2, 3), nano::make_dims(100, 1, 5, 1));
    UTEST_REQUIRE_NOTHROW(dataset.load());

    auto gridsearch = grid_search_model_t{model, grid};
    UTEST_CHECK_NOTHROW(gridsearch.folds(3));
    UTEST_CHECK_NOTHROW(gridsearch.max_trials(10));
    UTEST_CHECK_NOTHROW(gridsearch.fit(*loss, dataset, dataset.train_samples(), *solver));

    const auto& configs = gridsearch.configs();
    UTEST_REQUIRE_EQUAL(configs.size(), 10U);

    const auto& optimum = configs[0];
    const auto& values = optimum.values();
    UTEST_REQUIRE_EQUAL(values.size(), 3U);
    UTEST_CHECK_EQUAL(gridsearch.model().ivalue("iparam1"), std::get<int64_t>(values[0].second));
    UTEST_CHECK_EQUAL(gridsearch.model().ivalue("iparam2"), std::get<int64_t>(values[1].second));
    UTEST_CHECK_CLOSE(gridsearch.model().svalue("sparam1"), std::get<scalar_t>(values[2].second), 1e-12);
}

UTEST_END_MODULE()
