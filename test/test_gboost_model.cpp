#include <fstream>
#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"
#include <nano/gboost/model.h>

using namespace nano;

class gboost_dataset_t : public fixture_dataset_t
{
public:

    gboost_dataset_t() = default;

    [[nodiscard]] tensor_size_t groups() const override { return 3; }
    [[nodiscard]] bool is_optional(tensor_size_t, tensor_size_t) const override { return false; }
};

class gboost_linear_dataset_t : public gboost_dataset_t
{
public:

    gboost_linear_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_affine_target<fun1_lin_t>(sample, gt_feature1(), 6, +1.0, -0.5) +
            make_affine_target<fun1_lin_t>(sample, gt_feature2(), 7, +2.0, -1.5) +
            make_affine_target<fun1_lin_t>(sample, gt_feature3(), 8, -1.0, +2.5));
    }

    [[nodiscard]] tensor_size_t gt_feature1(bool discrete = false) const { return get_feature(discrete); }
    [[nodiscard]] tensor_size_t gt_feature2(bool discrete = false) const { return get_feature(gt_feature1(), discrete); }
    [[nodiscard]] tensor_size_t gt_feature3(bool discrete = false) const { return get_feature(gt_feature2(), discrete); }
};

class gboost_mixed_dataset_t : public gboost_dataset_t
{
public:

    gboost_mixed_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_affine_target<fun1_lin_t>(sample, gt_feature1(), 6, +1.0, -0.5) +
            make_stump_target(sample, gt_feature2(), 7, +3.5, +2.0, -1.5, 0) +
            make_stump_target(sample, gt_feature3(), 8, +2.5, -1.0, +2.5, 0));
    }

    [[nodiscard]] tensor_size_t gt_feature1(bool discrete = false) const { return get_feature(discrete); }
    [[nodiscard]] tensor_size_t gt_feature2(bool discrete = false) const { return get_feature(gt_feature1(), discrete); }
    [[nodiscard]] tensor_size_t gt_feature3(bool discrete = false) const { return get_feature(gt_feature2(), discrete); }
};

static auto make_solver(const char* name = "lbfgs", const scalar_t epsilon = epsilon3<scalar_t>())
{
    auto solver = solver_t::all().get(name);
    UTEST_REQUIRE(solver);
    solver->epsilon(epsilon);
    solver->max_iterations(100);
    return solver;
}

static auto check_stream(const gboost_model_t& orig_model)
{
    string_t str;
    {
        std::ostringstream stream;
        UTEST_REQUIRE_NOTHROW(orig_model.write(stream));
        str = stream.str();
    }
    {
        gboost_model_t model;
        std::ifstream stream;
        UTEST_REQUIRE_THROW(model.read(stream), std::runtime_error);
    }
    {
        gboost_model_t model;
        std::istringstream stream(str);
        UTEST_REQUIRE_NOTHROW(model.read(stream));
        return model;
    }
}

static auto check_predict(const dataset_t& dataset, const loss_t& loss, const gboost_model_t& model)
{
    const auto fold = fold_t{0, protocol::test};
    const auto targets = dataset.targets(fold);

    tensor4d_t outputs;
    UTEST_REQUIRE_NOTHROW(model.predict(dataset, fold, outputs));
    UTEST_CHECK_EQUAL(outputs.dims(), cat_dims(dataset.samples(fold), dataset.tdim()));
    UTEST_CHECK_EIGEN_CLOSE(targets.vector(), outputs.vector(), 1e-3);

    const auto imodel = ::check_stream(model);

    UTEST_REQUIRE_NOTHROW(imodel.predict(dataset, fold, outputs));
    UTEST_CHECK_EQUAL(outputs.dims(), cat_dims(dataset.samples(fold), dataset.tdim()));
    UTEST_CHECK_EIGEN_CLOSE(targets.vector(), outputs.vector(), 1e-3);

    tensor1d_t errors(dataset.samples(fold));
    dataset.loop(execution::par, fold, model.batch(), [&] (tensor_range_t range, size_t)
    {
        const auto targets = dataset.targets(fold, range);
        loss.error(targets, outputs.slice(range), errors.slice(range));
    });

    return errors.vector().mean();
}

static void check_result(const dataset_t& dataset, const train_result_t& result, const gboost_model_t& model)
{
    UTEST_CHECK_EQUAL(result.size(), dataset.folds());
    for (const auto& rfold : result)
    {
        UTEST_CHECK_LESS(rfold.tr_error(), 1e-3);
        UTEST_CHECK_LESS(rfold.vd_error(), 1e-3);
        UTEST_CHECK_LESS(rfold.te_error(), 1e-3);
        UTEST_CHECK_LESS(rfold.avg_te_error(), 1e-3);
    }

    UTEST_REQUIRE_EQUAL(result.size(), model.models().size());

    for (size_t i = 0, size = result.size(); i < size; ++ i)
    {
        const auto curve = result[i].optimum().second;
        UTEST_CHECK_EQUAL(curve.optindex(), model.models()[i].m_protos.size());
    }
}

static void check_features(const loss_t& loss, const dataset_t& dataset, const gboost_model_t& model)
{
    auto features = model.features(loss, dataset);
    feature_info_t::sort_by_index(features);
    UTEST_REQUIRE_EQUAL(features.size(), 3U);
    UTEST_CHECK_EQUAL(features[0].feature(), 5);
    UTEST_CHECK_EQUAL(features[1].feature(), 7);
    UTEST_CHECK_EQUAL(features[2].feature(), 9);
    UTEST_CHECK_GREATER_EQUAL(features[0].folds(), 1);
    UTEST_CHECK_GREATER_EQUAL(features[1].folds(), 1);
    UTEST_CHECK_GREATER_EQUAL(features[2].folds(), 1);
    UTEST_CHECK_GREATER_EQUAL(features[0].importance(), 0.5);
    UTEST_CHECK_GREATER_EQUAL(features[1].importance(), 0.5);
    UTEST_CHECK_GREATER_EQUAL(features[2].importance(), 0.5);
}

UTEST_BEGIN_MODULE(test_gboost_model)

// TODO: check that the selected features are the expected ones!
// TODO: check per-fold results!
// TODO: check tuning e.g. of the variance factor!

// TODO: check valid and invalid configurations
// TODO: check model loading and saving
// TODO: check predictions

UTEST_CASE(print)
{
    for (const auto type : {wlearner::real, wlearner::discrete})
    {
        std::stringstream stream;
        stream << type;
        UTEST_CHECK(!stream.str().empty());
    }
    for (const auto type : {wscale::gboost, wscale::tboost})
    {
        std::stringstream stream;
        stream << type;
        UTEST_CHECK(!stream.str().empty());
    }
}

UTEST_CASE(add_protos)
{
    UTEST_CHECK(wlearner_t::all().get("dtree") != nullptr);
    UTEST_CHECK(wlearner_t::all().get("stump") != nullptr);
    UTEST_CHECK(wlearner_t::all().get("table") != nullptr);
    UTEST_CHECK(wlearner_t::all().get("lin1") != nullptr);
    UTEST_CHECK(wlearner_t::all().get("log1") != nullptr);
    UTEST_CHECK(wlearner_t::all().get("cos1") != nullptr);
    UTEST_CHECK(wlearner_t::all().get("sin1") != nullptr);

    auto model = gboost_model_t{};
    UTEST_CHECK_THROW(model.add("invalid_wlearner_id"), std::runtime_error);
    UTEST_CHECK_NOTHROW(model.add("dtree"));
    UTEST_CHECK_NOTHROW(model.add("stump"));
    UTEST_CHECK_NOTHROW(model.add("stump"));
    UTEST_CHECK_NOTHROW(model.add("dtree"));
    UTEST_CHECK_NOTHROW(model.add(wlearner_lin1_t{}));
}

UTEST_CASE(default_predict)
{
    const auto fold = fold_t{0, protocol::test};
    const auto dataset = make_dataset<gboost_linear_dataset_t>(10, 1, 400);
    const auto model = gboost_model_t{};

    tensor4d_t outputs;
    UTEST_REQUIRE_NOTHROW(model.predict(dataset, fold, outputs));
    UTEST_CHECK_EQUAL(outputs.dims(), cat_dims(dataset.samples(fold), dataset.tdim()));
    UTEST_CHECK_CLOSE(outputs.min(), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(outputs.max(), 0.0, 1e-12);

    const auto imodel = ::check_stream(model);

    UTEST_REQUIRE_NOTHROW(imodel.predict(dataset, fold, outputs));
    UTEST_CHECK_EQUAL(outputs.dims(), cat_dims(dataset.samples(fold), dataset.tdim()));
    UTEST_CHECK_CLOSE(outputs.min(), 0.0, 1e-12);
}

UTEST_CASE(train_linear)
{
    auto loss = make_loss();
    auto solver = make_solver();
    auto dataset = make_dataset<gboost_linear_dataset_t>(10, 1, 120);

    auto wlinear = wlearner_lin1_t{};

    auto model = gboost_model_t{};
    UTEST_REQUIRE_NOTHROW(model.rounds(10));
    UTEST_REQUIRE_NOTHROW(model.patience(5));
    UTEST_REQUIRE_NOTHROW(model.subsample(90));
    UTEST_REQUIRE_NOTHROW(model.tune_steps(2));
    UTEST_REQUIRE_NOTHROW(model.tune_trials(5));
    UTEST_REQUIRE_NOTHROW(model.regularization(::nano::regularization::variance));
    UTEST_REQUIRE_NOTHROW(model.add(wlinear));

    auto result = train_result_t{};
    UTEST_REQUIRE_NOTHROW(result = model.train(*loss, dataset, *solver));
    ::check_result(dataset, result, model);
    ::check_features(*loss, dataset, model);

    const auto avg_te_error = ::check_predict(dataset, *loss,  model);
    UTEST_CHECK_CLOSE(result[0].avg_te_error(), avg_te_error, 1e-8);
}

UTEST_CASE(train_mixed)
{
    auto loss = make_loss();
    auto solver = make_solver();
    auto dataset = make_dataset<gboost_mixed_dataset_t>(10, 1, 130);

    auto wstump = wlearner_stump_t{};
    auto wlinear = wlearner_lin1_t{};

    auto model = gboost_model_t{};
    UTEST_REQUIRE_NOTHROW(model.rounds(10));
    UTEST_REQUIRE_NOTHROW(model.patience(5));
    UTEST_REQUIRE_NOTHROW(model.subsample(100));
    UTEST_REQUIRE_NOTHROW(model.tune_steps(3));
    UTEST_REQUIRE_NOTHROW(model.tune_trials(5));
    UTEST_REQUIRE_NOTHROW(model.regularization(::nano::regularization::none));
    UTEST_REQUIRE_NOTHROW(model.add(wstump));
    UTEST_REQUIRE_NOTHROW(model.add(wlinear));

    auto result = train_result_t{};
    UTEST_REQUIRE_NOTHROW(result = model.train(*loss, dataset, *solver));
    ::check_result(dataset, result, model);
    ::check_features(*loss, dataset, model);

    const auto avg_te_error = ::check_predict(dataset, *loss,  model);
    UTEST_CHECK_CLOSE(result[0].avg_te_error(), avg_te_error, 1e-8);
}

UTEST_END_MODULE()
