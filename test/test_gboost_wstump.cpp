#include <utest/utest.h>
#include "fixture/gboost.h"
#include <nano/core/numeric.h>

using namespace nano;

class wstump_dataset_t : public fixture_dataset_t
{
public:

    wstump_dataset_t() = default;

    tensor_size_t groups() const override
    {
        return 2;
    }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).full(
            make_stump_target(sample, gt_feature(), 5, 2.5, +3.0, -2.1, 0));
    }

    void check_wlearner(const wlearner_stump_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), gt_feature());
        UTEST_CHECK_EQUAL(wlearner.tables().dims(), tables().dims());
        UTEST_CHECK_CLOSE(wlearner.threshold(), threshold(), 1e-8);
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables().array(), 1e-8);
    }

    scalar_t threshold() const { return 2.5; }
    tensor_size_t gt_feature(bool discrete = false) const { return get_feature(discrete); }
    tensor4d_t tables() const { return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), +3.0, -2.1); }
};

UTEST_BEGIN_MODULE(test_gboost_wstump)

UTEST_CASE(fitting)
{
    const auto dataset = make_dataset<wstump_dataset_t>();
    const auto datasetx1 = make_dataset<wstump_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wstump_dataset_t>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<wstump_dataset_t>>();

    auto wlearner = make_wlearner<wlearner_stump_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
}

UTEST_END_MODULE()
