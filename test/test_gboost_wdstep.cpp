#include <utest/utest.h>
#include "fixture/gboost.h"
#include <nano/core/numeric.h>

using namespace nano;

template <tensor_size_t fvalue>
class wdstep_dataset_t : public fixture_dataset_t
{
public:

    wdstep_dataset_t() = default;

    tensor_size_t groups() const override
    {
        return 1;
    }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).full(
            make_dstep_target(sample, gt_feature(), 3, 5.0, fvalue, 0));
    }

    void check_wlearner(const wlearner_dstep_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.fvalues(), 3);
        UTEST_CHECK_EQUAL(wlearner.fvalue(), fvalue);
        UTEST_CHECK_EQUAL(wlearner.feature(), gt_feature());
        UTEST_CHECK_EQUAL(wlearner.tables().dims(), tables().dims());
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables().array(), 1e-8);
    }

    tensor_size_t the_discrete_feature() const { return gt_feature(); }
    tensor_size_t gt_feature(bool discrete = true) const { return get_feature(discrete); }
    tensor4d_t tables() const
    {
        const auto table0 = fvalue == 0 ? 5.0 : 0.0;
        const auto table1 = fvalue == 1 ? 5.0 : 0.0;
        const auto table2 = fvalue == 2 ? 5.0 : 0.0;
        return make_tensor<scalar_t>(make_dims(3, 1, 1, 1), table0, table1, table2);
    }
};

UTEST_BEGIN_MODULE(test_gboost_wdstep)

UTEST_CASE(fitting0)
{
    const auto dataset = make_dataset<wdstep_dataset_t<0>>();
    const auto datasetx1 = make_dataset<wdstep_dataset_t<0>>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wdstep_dataset_t<0>>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_discrete_features_dataset_t<wdstep_dataset_t<0>>>();
    const auto datasetx4 = make_dataset<different_discrete_feature_dataset_t<wdstep_dataset_t<0>>>();

    auto wlearner = make_wlearner<wlearner_dstep_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3, datasetx4);
}

UTEST_CASE(fitting1)
{
    const auto dataset = make_dataset<wdstep_dataset_t<1>>();
    const auto datasetx1 = make_dataset<wdstep_dataset_t<1>>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wdstep_dataset_t<1>>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_discrete_features_dataset_t<wdstep_dataset_t<1>>>();
    const auto datasetx4 = make_dataset<different_discrete_feature_dataset_t<wdstep_dataset_t<1>>>();

    auto wlearner = make_wlearner<wlearner_dstep_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3, datasetx4);
}

UTEST_CASE(fitting2)
{
    const auto dataset = make_dataset<wdstep_dataset_t<2>>();
    const auto datasetx1 = make_dataset<wdstep_dataset_t<2>>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wdstep_dataset_t<2>>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_discrete_features_dataset_t<wdstep_dataset_t<2>>>();
    const auto datasetx4 = make_dataset<different_discrete_feature_dataset_t<wdstep_dataset_t<2>>>();

    auto wlearner = make_wlearner<wlearner_dstep_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3, datasetx4);
}

UTEST_END_MODULE()
