#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"
#include <nano/gboost/wlearner_table.h>

using namespace nano;

class wtable_dataset_t : public fixture_dataset_t
{
public:

    wtable_dataset_t() = default;

    [[nodiscard]] tensor_size_t groups() const override
    {
        return 3;
    }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(make_table_target(sample, feature(), 3, 5.0, 0));
    }

    [[nodiscard]] tensor_size_t the_discrete_feature() const { return feature(); }
    [[nodiscard]] tensor_size_t feature(bool discrete = true) const { return get_feature(discrete); }
    [[nodiscard]] tensor4d_t rtables() const { return {make_dims(3, 1, 1, 1), std::array<scalar_t, 3>{{-5.0, +0.0, +5.0}}}; }
    [[nodiscard]] tensor4d_t dtables() const { return {make_dims(3, 1, 1, 1), std::array<scalar_t, 3>{{-1.0, +0.0, +1.0}}}; }
};

UTEST_BEGIN_MODULE(test_gboost_wtable)

UTEST_CASE(fitting)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<wtable_dataset_t>();

    for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
    {
        // check fitting
        auto wlearner = make_wlearner<wlearner_table_t>(type);
        check_fit(dataset, fold, wlearner);

        const auto tables = (type == ::nano::wlearner::real) ? dataset.rtables() : dataset.dtables();

        UTEST_CHECK_EQUAL(wlearner.odim(), dataset.tdim());
        UTEST_CHECK_EQUAL(wlearner.feature(), dataset.feature());
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables.array(), 1e-8);

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
    const auto dataset = make_dataset<wtable_dataset_t>();
    const auto datasetx = make_dataset<no_discrete_features_dataset_t<wtable_dataset_t>>();

    for (const auto type : {static_cast<::nano::wlearner>(-1)})
    {
        auto wlearner = make_wlearner<wlearner_table_t>(type);
        check_fit_throws(dataset, fold, wlearner);
    }

    for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
    {
        auto wlearner = make_wlearner<wlearner_table_t>(type);
        check_no_fit(datasetx, fold, wlearner);
    }
}

UTEST_CASE(predict)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<wtable_dataset_t>();
    const auto datasetx1 = make_dataset<wtable_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wtable_dataset_t>(dataset.feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_discrete_features_dataset_t<wtable_dataset_t>>();
    const auto datasetx4 = make_dataset<different_discrete_feature_dataset_t<wtable_dataset_t>>();

    for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
    {
        auto wlearner = make_wlearner<wlearner_table_t>(type);
        check_predict_throws(dataset, fold, wlearner);
        check_predict_throws(datasetx1, fold, wlearner);
        check_predict_throws(datasetx2, fold, wlearner);
        check_predict_throws(datasetx3, fold, wlearner);
        check_predict_throws(datasetx4, fold, wlearner);

        check_fit(dataset, fold, wlearner);

        check_predict(dataset, fold, wlearner);
        check_predict_throws(datasetx1, fold, wlearner);
        check_predict_throws(datasetx2, fold, wlearner);
        check_predict_throws(datasetx3, fold, wlearner);
        check_predict_throws(datasetx4, fold, wlearner);
    }
}

UTEST_CASE(split)
{
    const auto fold = make_fold();
    const auto dataset = make_dataset<wtable_dataset_t>();

    for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
    {
        auto wlearner = make_wlearner<wlearner_table_t>(type);
        check_split_throws(dataset, fold, make_indices(dataset, fold), wlearner);

        check_fit(dataset, fold, wlearner);

        check_split(dataset, wlearner);
        check_split_throws(dataset, fold, make_invalid_indices(dataset, fold), wlearner);
    }
}

UTEST_END_MODULE()
