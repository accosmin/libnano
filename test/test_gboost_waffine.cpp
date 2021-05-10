#include <utest/utest.h>
#include "fixture/gboost.h"
#include <nano/core/numeric.h>

using namespace nano;

template <typename tfun1>
class waffine_dataset_t : public fixture_dataset_t
{
public:

    waffine_dataset_t() = default;

    tensor_size_t groups() const override
    {
        return 1;
    }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).full(
            make_affine_target<tfun1>(sample, gt_feature(), 6, weight(), bias(), 0));
    }

    void check_wlearner(const wlearner_affine_t<tfun1>& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), gt_feature());
        UTEST_REQUIRE_EQUAL(wlearner.tables().dims(), make_dims(2, 1, 1, 1));
        UTEST_CHECK_CLOSE(wlearner.tables()(0), weight(), 1e-8);
        UTEST_CHECK_CLOSE(wlearner.tables()(1), bias(), 1e-8);
    }

    scalar_t bias() const { return -7.1; }
    scalar_t weight() const { return +3.5; }
    tensor_size_t gt_feature(bool discrete = false) const { return get_feature(discrete); }
};

template <typename tfun1>
static void check_fitting()
{
    const auto dataset = make_dataset<waffine_dataset_t<tfun1>>();
    const auto datasetx1 = make_dataset<waffine_dataset_t<tfun1>>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<waffine_dataset_t<tfun1>>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<waffine_dataset_t<tfun1>>>();

    auto wlearner = make_wlearner<wlearner_affine_t<tfun1>>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
}

UTEST_BEGIN_MODULE(test_gboost_waffine)

UTEST_CASE(fitting_lin)
{
    check_fitting<fun1_lin_t>();
}

UTEST_CASE(fitting_log)
{
    check_fitting<fun1_log_t>();
}

UTEST_CASE(fitting_cos)
{
    check_fitting<fun1_cos_t>();
}

UTEST_CASE(fitting_sin)
{
    check_fitting<fun1_sin_t>();
}

UTEST_END_MODULE()
