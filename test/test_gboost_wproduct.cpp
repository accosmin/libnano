#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"

using namespace nano;

class wproduct_dataset_t : public fixture_dataset_t
{
public:

    wproduct_dataset_t() = default;

    [[nodiscard]] virtual strings_t ids() const = 0;
    [[nodiscard]] virtual indices_t features() const = 0;
    [[nodiscard]] tensor_size_t groups() const override { return 1; }

    void check_wlearner(const wlearner_product_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.features(), features());
        UTEST_CHECK_EQUAL(wlearner.protos().size(), 4U);

        const auto ids = this->ids();
        const auto& terms = wlearner.terms();
        UTEST_REQUIRE_EQUAL(terms.size(), ids.size());
        UTEST_CHECK_EQUAL(static_cast<tensor_size_t>(terms.size()), wlearner.degree());
        for (const auto& term : terms)
        {
            std::cout << "term.m_id = " << term.m_id << std::endl;
            UTEST_CHECK(std::find(ids.begin(), ids.end(), term.m_id) != ids.end());
        }
    }
};

class lin1_dataset_t : public wproduct_dataset_t
{
public:

    lin1_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_affine_target<fun1_lin_t>(sample, feature(), 6, -7.1, +2.4, 0));
    }

    [[nodiscard]] strings_t ids() const override { return {"lin1"}; }
    [[nodiscard]] indices_t features() const override { return std::array<tensor_size_t, 1>{{feature()}}; }
    [[nodiscard]] tensor_size_t feature(bool discrete = false) const { return get_feature(discrete); }
};

class logXlin_dataset_t : public wproduct_dataset_t
{
public:

    logXlin_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_affine_target<fun1_lin_t>(sample, feat1(), 6, -7.1, +2.4, 0) *
            make_affine_target<fun1_log_t>(sample, feat2(), 6, +0.7, -1.1, 0));
    }

    [[nodiscard]] strings_t ids() const override { return {"lin1", "log1"}; }
    [[nodiscard]] indices_t features() const override { return std::array<tensor_size_t, 2>{{feat2(), feat1()}}; }
    [[nodiscard]] tensor_size_t feat1(bool discrete = false) const { return get_feature(discrete); }
    [[nodiscard]] tensor_size_t feat2(bool discrete = false) const { return get_feature(feat1(), discrete); }
};

static auto make_wproduct(::nano::wlearner type, int degree)
{
    auto wlearner = make_wlearner<wlearner_product_t>(type);
    UTEST_REQUIRE_NOTHROW(wlearner.degree(degree));
    UTEST_REQUIRE_NOTHROW(wlearner.add("lin1"));
    UTEST_REQUIRE_NOTHROW(wlearner.add(make_wlearner<wlearner_affine_t<fun1_log_t>>(type)));
    UTEST_REQUIRE_NOTHROW(wlearner.add(make_wlearner<wlearner_affine_t<fun1_cos_t>>(type)));
    UTEST_REQUIRE_NOTHROW(wlearner.add(make_wlearner<wlearner_affine_t<fun1_sin_t>>(type)));
    UTEST_CHECK_EQUAL(wlearner.terms().size(), 0U);
    UTEST_CHECK_EQUAL(wlearner.protos().size(), 4U);
    return wlearner;
}

UTEST_BEGIN_MODULE(test_gboost_wproduct)

UTEST_CASE(fitting_lin1)
{
    const auto dataset = make_dataset<lin1_dataset_t>();
    const auto datasetx1 = make_dataset<lin1_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<lin1_dataset_t>(dataset.feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<lin1_dataset_t>>();

    for (const auto type : {static_cast<::nano::wlearner>(-1), ::nano::wlearner::discrete})
    {
        auto wlearner = make_wproduct(type, 1);
        check_fit_throws(wlearner, dataset);
    }

    for (const auto type : {::nano::wlearner::real})
    {
        auto wlearner = make_wproduct(type, 1);
        check_no_fit(wlearner, datasetx3);
        check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
    }
}

UTEST_CASE(fitting_logXlin)
{
    const auto dataset = make_dataset<logXlin_dataset_t>();
    const auto datasetx1 = make_dataset<logXlin_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<logXlin_dataset_t>(dataset.feat1(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<logXlin_dataset_t>>();

    for (const auto type : {static_cast<::nano::wlearner>(-1), ::nano::wlearner::discrete})
    {
        auto wlearner = make_wproduct(type, 2);
        check_fit_throws(wlearner, dataset);
    }

    for (const auto type : {::nano::wlearner::real})
    {
        auto wlearner = make_wproduct(type, 2);
        check_no_fit(wlearner, datasetx3);
        check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
    }
}


UTEST_END_MODULE()
