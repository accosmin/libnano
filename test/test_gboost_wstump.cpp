#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"

using namespace nano;

class wstump_dataset_t : public fixture_dataset_t
{
public:

    wstump_dataset_t() = default;

    [[nodiscard]] tensor_size_t groups() const override
    {
        return 2;
    }

    void make_scale(const tensor_size_t sample) override
    {
        scale(sample).constant(
            static_cast<scalar_t>(sample + 1) /
            static_cast<scalar_t>(samples()));
    }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_stump_target(sample, feature(), 5, 2.5, +3.0, -2.1, 0));

        target(sample).array() *= scale(sample).array();
    }

    void check_wlearner(const wlearner_stump_t& wlearner) const
    {
        const auto tables = (wlearner.type() == ::nano::wlearner::real) ? rtables() : dtables();
        UTEST_CHECK_EQUAL(wlearner.feature(), feature());
        UTEST_CHECK_EQUAL(wlearner.tables().dims(), tables.dims());
        UTEST_CHECK_CLOSE(wlearner.threshold(), threshold(), 1e-8);
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables.array(), 1e-8);
    }

    [[nodiscard]] scalar_t threshold() const { return 2.5; }
    [[nodiscard]] tensor_size_t feature(bool discrete = false) const { return get_feature(discrete); }
    [[nodiscard]] tensor4d_t rtables() const { return {make_dims(2, 1, 1, 1), std::array<scalar_t, 2>{{+3.0, -2.1}}}; }
    [[nodiscard]] tensor4d_t dtables() const { return {make_dims(2, 1, 1, 1), std::array<scalar_t, 2>{{+1.0, -1.0}}}; }
};

UTEST_BEGIN_MODULE(test_gboost_wstump)

UTEST_CASE(fitting)
{
    const auto dataset = make_dataset<wstump_dataset_t>();
    const auto datasetx1 = make_dataset<wstump_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wstump_dataset_t>(dataset.feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<wstump_dataset_t>>();

    for (const auto type : {static_cast<::nano::wlearner>(-1)})
    {
        auto wlearner = make_wlearner<wlearner_stump_t>(type);
        check_fit_throws(wlearner, dataset);
    }

    for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
    {
        auto wlearner = make_wlearner<wlearner_stump_t>(type);
        check_no_fit(wlearner, datasetx3);
        check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
    }
}

UTEST_END_MODULE()
