#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"
#include <nano/gboost/wlearner_affine.h>

using namespace nano;

class wlinear_dataset_t : public fixture_dataset_t
{
public:

    wlinear_dataset_t() = default;

    [[nodiscard]] tensor_size_t groups() const override
    {
        return 1;
    }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(make_linear_target(sample, gt_feature(), 6, gt_weight(), gt_bias(), 0));
    }

    [[nodiscard]] scalar_t gt_bias() const { return -7.1; }
    [[nodiscard]] scalar_t gt_weight() const { return +3.5; }
    [[nodiscard]] tensor_size_t gt_feature(bool discrete = false) const { return get_feature(discrete); }
};

UTEST_BEGIN_MODULE(test_gboost_waffine)

UTEST_CASE(fitting)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<wlinear_dataset_t>();

    for (const auto type : {::nano::wlearner::real})
    {
        // check fitting
        auto wlearner = make_wlearner<wlearner_lin1_t>(type);
        check_fit(dataset, fold, wlearner);

        UTEST_CHECK_EQUAL(wlearner.odim(), dataset.tdim());
        UTEST_CHECK_EQUAL(wlearner.feature(), dataset.gt_feature());

        UTEST_REQUIRE_EQUAL(wlearner.tables().dims(), make_dims(2, 1, 1, 1));
        UTEST_CHECK_CLOSE(wlearner.tables()(0), dataset.gt_weight(), 1e-8);
        UTEST_CHECK_CLOSE(wlearner.tables()(1), dataset.gt_bias(), 1e-8);

        // check scaling
        check_scale(dataset, fold, wlearner);

        // check model loading and saving from and to binary streams
        const auto iwlearner = stream_wlearner(wlearner);
        UTEST_CHECK_EQUAL(wlearner.feature(), iwlearner.feature());
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), iwlearner.tables().array(), 1e-8);
    }
}

UTEST_CASE(no_fitting)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<wlinear_dataset_t>();
    const auto datasetx = make_dataset<no_continuous_features_dataset_t<wlinear_dataset_t>>();

    for (const auto type : {::nano::wlearner::discrete, static_cast<::nano::wlearner>(-1)})
    {
        auto wlearner = make_wlearner<wlearner_lin1_t>(type);
        check_fit_throws(dataset, fold, wlearner);
    }

    for (const auto type : {::nano::wlearner::real})
    {
        auto wlearner = make_wlearner<wlearner_lin1_t>(type);
        check_no_fit(datasetx, fold, wlearner);
    }
}

UTEST_CASE(predict)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<wlinear_dataset_t>();
    const auto datasetx1 = make_dataset<wlinear_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wlinear_dataset_t>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<wlinear_dataset_t>>();

    auto wlearner = make_wlearner<wlearner_lin1_t>(::nano::wlearner::real);
    check_predict_throws(dataset, fold, wlearner);
    check_predict_throws(datasetx1, fold, wlearner);
    check_predict_throws(datasetx2, fold, wlearner);
    check_predict_throws(datasetx3, fold, wlearner);

    check_fit(dataset, fold, wlearner);

    check_predict(dataset, fold, wlearner);
    check_predict_throws(datasetx1, fold, wlearner);
    check_predict_throws(datasetx2, fold, wlearner);
    check_predict_throws(datasetx3, fold, wlearner);
}

UTEST_CASE(split)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<wlinear_dataset_t>();

    auto wlearner = make_wlearner<wlearner_lin1_t>(::nano::wlearner::real);
    check_split_throws(dataset, fold, make_indices(dataset, fold), wlearner);
    check_split_throws(dataset, fold, make_invalid_indices(dataset, fold), wlearner);

    check_fit(dataset, fold, wlearner);

    check_split(dataset, wlearner);
    check_split_throws(dataset, fold, make_invalid_indices(dataset, fold), wlearner);
}

UTEST_END_MODULE()
