#include <fstream>
#include <utest/utest.h>
#include "fixture/gboost.h"
#include <nano/core/numeric.h>
#include <nano/gboost/model.h>

using namespace nano;

inline std::ostream& operator<<(std::ostream& stream, wscale type)
{
    return stream << scat(type);
}

inline std::ostream& operator<<(std::ostream& stream, importance type)
{
    return stream << scat(type);
}

class gboost_dataset_t : public fixture_dataset_t
{
public:

    gboost_dataset_t() = default;

    tensor_size_t groups() const override { return 3; }
    bool is_optional(tensor_size_t, tensor_size_t) const override { return false; }
};

class gboost_linear_dataset_t : public gboost_dataset_t
{
public:

    gboost_linear_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).full(
            make_affine_target<fun1_lin_t>(sample, gt_feature1(), 6, +1.0, -0.5) +
            make_affine_target<fun1_lin_t>(sample, gt_feature2(), 7, +2.0, -1.5) +
            make_affine_target<fun1_lin_t>(sample, gt_feature3(), 8, -1.0, +2.5));
    }

    tensor_size_t gt_feature1(bool discrete = false) const { return get_feature(discrete); }
    tensor_size_t gt_feature2(bool discrete = false) const { return get_feature(gt_feature1(), discrete); }
    tensor_size_t gt_feature3(bool discrete = false) const { return get_feature(gt_feature2(), discrete); }
};

class gboost_mixed_dataset_t : public gboost_dataset_t
{
public:

    gboost_mixed_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).full(
            make_affine_target<fun1_lin_t>(sample, gt_feature1(), 6, +1.0, -0.5) +
            make_stump_target(sample, gt_feature2(), 7, +3.5, +2.0, -1.5, 0) +
            make_stump_target(sample, gt_feature3(), 8, +2.5, -1.0, +2.5, 0));
    }

    tensor_size_t gt_feature1(bool discrete = false) const { return get_feature(discrete); }
    tensor_size_t gt_feature2(bool discrete = false) const { return get_feature(gt_feature1(), discrete); }
    tensor_size_t gt_feature3(bool discrete = false) const { return get_feature(gt_feature2(), discrete); }
};

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
        UTEST_CHECK_EQUAL(model.batch(), orig_model.batch());
        UTEST_CHECK_EQUAL(model.vAreg(), orig_model.vAreg());
        UTEST_CHECK_EQUAL(model.wscale(), orig_model.wscale());
        UTEST_CHECK_EQUAL(model.rounds(), orig_model.rounds());
        UTEST_CHECK_EQUAL(model.epsilon(), orig_model.epsilon());
        UTEST_CHECK_EQUAL(model.shrinkage(), orig_model.shrinkage());
        UTEST_CHECK_EQUAL(model.subsample(), orig_model.subsample());
        return model;
    }
}

static void check_predict(const dataset_t& dataset, const gboost_model_t& model)
{
    const auto samples = make_samples(dataset);
    const auto targets = dataset.targets(samples);

    tensor4d_t outputs;
    UTEST_REQUIRE_NOTHROW(outputs = model.predict(dataset, samples));
    UTEST_CHECK_EQUAL(outputs.dims(), cat_dims(samples.size(), dataset.tdims()));
    UTEST_CHECK_EIGEN_CLOSE(targets.vector(), outputs.vector(), 1e-3);

    // check that the predictions shouldn't change at all when reloading the model
    const auto imodel = ::check_stream(model);

    tensor4d_t soutputs;
    UTEST_REQUIRE_NOTHROW(soutputs = imodel.predict(dataset, samples));
    UTEST_CHECK_EQUAL(outputs.dims(), soutputs.dims());
    UTEST_CHECK_EIGEN_CLOSE(outputs.vector(), soutputs.vector(), 1e-8);
}

static void check_features(const dataset_t& dataset, const loss_t& loss, const gboost_model_t& model)
{
    const auto trials = 3;
    const auto solver = make_solver();
    const auto samples = make_samples(dataset);

    auto features = model.features();
    feature_info_t::sort_by_index(features);
    UTEST_REQUIRE_EQUAL(features.size(), 3U);
    UTEST_CHECK_EQUAL(features[0].feature(), 5);
    UTEST_CHECK_EQUAL(features[1].feature(), 7);
    UTEST_CHECK_EQUAL(features[2].feature(), 9);
    UTEST_CHECK_GREATER_EQUAL(features[0].count(), 1);
    UTEST_CHECK_GREATER_EQUAL(features[1].count(), 1);
    UTEST_CHECK_GREATER_EQUAL(features[2].count(), 1);
    UTEST_CHECK_CLOSE(features[0].importance(), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(features[1].importance(), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(features[2].importance(), 0.0, 1e-12);

    for (const auto type : enum_values<importance>())
    {
        auto features = model.features(loss, dataset, samples, *solver, type, trials);
        feature_info_t::sort_by_index(features);
        UTEST_REQUIRE_EQUAL(features.size(), 3U);
        UTEST_CHECK_EQUAL(features[0].feature(), 5);
        UTEST_CHECK_EQUAL(features[1].feature(), 7);
        UTEST_CHECK_EQUAL(features[2].feature(), 9);
        UTEST_CHECK_GREATER_EQUAL(features[0].count(), 1);
        UTEST_CHECK_GREATER_EQUAL(features[1].count(), 1);
        UTEST_CHECK_GREATER_EQUAL(features[2].count(), 1);
        UTEST_CHECK_GREATER_EQUAL(features[0].importance(), 0.5);
        UTEST_CHECK_GREATER_EQUAL(features[1].importance(), 0.5);
        UTEST_CHECK_GREATER_EQUAL(features[2].importance(), 0.5);
    }
}

UTEST_BEGIN_MODULE(test_gboost_model)

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
    const auto dataset = make_dataset<gboost_linear_dataset_t>();
    const auto samples = make_samples(dataset);

    tensor4d_t outputs;
    const auto model = gboost_model_t{};
    UTEST_REQUIRE_THROW(outputs = model.predict(dataset, samples), std::runtime_error);
}

UTEST_CASE(train_linear)
{
    const auto loss = make_loss();
    const auto solver = make_solver();
    const auto dataset = make_dataset<gboost_linear_dataset_t>();
    const auto samples = make_samples(dataset);

    auto wlinear = wlearner_lin1_t{};

    auto model = gboost_model_t{};
    UTEST_REQUIRE_NOTHROW(model.rounds(20));
    UTEST_REQUIRE_NOTHROW(model.epsilon(1e-8));
    UTEST_REQUIRE_NOTHROW(model.shrinkage(1.0));
    UTEST_REQUIRE_NOTHROW(model.subsample(0.9));
    UTEST_REQUIRE_NOTHROW(model.wscale(::nano::wscale::gboost));
    UTEST_REQUIRE_NOTHROW(model.add(wlinear));

    UTEST_REQUIRE_NOTHROW(model.fit(*loss, dataset, samples, *solver));
    ::check_predict(dataset, model);
    ::check_features(dataset, *loss, model);
}

UTEST_CASE(train_mixed)
{
    const auto loss = make_loss();
    const auto solver = make_solver();
    const auto dataset = make_dataset<gboost_mixed_dataset_t>(10, 1, 100);
    const auto samples = make_samples(dataset);

    auto wstump = wlearner_stump_t{};
    auto wlinear = wlearner_lin1_t{};

    auto model = gboost_model_t{};
    UTEST_REQUIRE_NOTHROW(model.rounds(10));
    UTEST_REQUIRE_NOTHROW(model.epsilon(1e-8));
    UTEST_REQUIRE_NOTHROW(model.shrinkage(1.0));
    UTEST_REQUIRE_NOTHROW(model.subsample(1.0));
    UTEST_REQUIRE_NOTHROW(model.wscale(::nano::wscale::tboost));
    UTEST_REQUIRE_NOTHROW(model.add(wstump));
    UTEST_REQUIRE_NOTHROW(model.add(wlinear));

    UTEST_REQUIRE_NOTHROW(model.fit(*loss, dataset, samples, *solver));
    ::check_predict(dataset, model);
    ::check_features(dataset, *loss, model);
}

UTEST_END_MODULE()
