#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"

using namespace nano;

template <typename tfun1>
class waffine_dataset_t : public fixture_dataset_t
{
public:

    waffine_dataset_t() = default;

    [[nodiscard]] tensor_size_t groups() const override
    {
        return 1;
    }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_affine_target<tfun1>(sample, gt_feature(), 6, gt_weight(), gt_bias(), 0));
    }

    [[nodiscard]] scalar_t gt_bias() const { return -7.1; }
    [[nodiscard]] scalar_t gt_weight() const { return +3.5; }
    [[nodiscard]] tensor_size_t gt_feature(bool discrete = false) const { return get_feature(discrete); }
};

template <typename tdataset>
class product_dataset_t : public tdataset
{
public:

    product_dataset_t() = default;

    bool load() override
    {
        m_base.resize(cat_dims(tdataset::samples(), tdataset::tdim()));
        m_base.random(-1.0, +1.0);

        return tdataset::load();
    }

    void make_target(const tensor_size_t sample) override
    {
        tdataset::make_target(sample);
        tdataset::target(sample).array() *= m_base.array(sample);
    }

private:

    // attributes
    tensor4d_t      m_base;     ///<
};


template <typename tfun1>
static void check_fitting()
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<waffine_dataset_t<tfun1>>();

    for (const auto type : {::nano::wlearner::real})
    {
        // check fitting
        auto wlearner = make_wlearner<wlearner_affine_t<tfun1>>(type);
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

template <typename tfun1>
static void check_predict()
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<waffine_dataset_t<tfun1>>();
    const auto datasetx1 = make_dataset<waffine_dataset_t<tfun1>>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<waffine_dataset_t<tfun1>>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<waffine_dataset_t<tfun1>>>();

    auto wlearner = make_wlearner<wlearner_affine_t<tfun1>>(::nano::wlearner::real);
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

UTEST_CASE(no_fitting)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<waffine_dataset_t<fun1_lin_t>>();
    const auto datasetx = make_dataset<no_continuous_features_dataset_t<waffine_dataset_t<fun1_lin_t>>>();

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

UTEST_CASE(predict_lin)
{
    check_predict<fun1_lin_t>();
}

UTEST_CASE(predict_log)
{
    check_predict<fun1_log_t>();
}

UTEST_CASE(predict_cos)
{
    check_predict<fun1_cos_t>();
}

UTEST_CASE(predict_sin)
{
    check_predict<fun1_sin_t>();
}

UTEST_CASE(split)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<waffine_dataset_t<fun1_lin_t>>();

    auto wlearner = make_wlearner<wlearner_lin1_t>(::nano::wlearner::real);
    check_split_throws(dataset, fold, make_indices(dataset, fold), wlearner);
    check_split_throws(dataset, fold, make_invalid_indices(dataset, fold), wlearner);

    check_fit(dataset, fold, wlearner);

    check_split(dataset, wlearner);
    check_split_throws(dataset, fold, make_invalid_indices(dataset, fold), wlearner);
}

UTEST_END_MODULE()
