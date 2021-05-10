#include <utest/utest.h>
#include "fixture/gboost.h"
#include <nano/core/numeric.h>

using namespace nano;

inline std::ostream& operator<<(std::ostream& stream, hinge type)
{
    return stream << scat(type);
}

class whinge_dataset_t : public fixture_dataset_t
{
public:

    whinge_dataset_t() = default;

    tensor_size_t groups() const override
    {
        return 1;
    }

    void check_wlearner(const wlearner_hinge_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.hinge(), hinge());
        UTEST_CHECK_EQUAL(wlearner.feature(), gt_feature());
        UTEST_CHECK_EQUAL(wlearner.tables().dims(), tables().dims());
        UTEST_CHECK_CLOSE(wlearner.threshold(), threshold(), 1e-8);
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables().array(), 1e-8);
    }

    scalar_t threshold() const { return 2.5; }
    tensor_size_t gt_feature(bool discrete = false) const { return get_feature(discrete); }
    virtual ::nano::hinge hinge() const = 0;
    virtual tensor4d_t tables() const = 0;
};

class whinge_left_dataset_t : public whinge_dataset_t
{
public:

    whinge_left_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).full(
            make_hinge_target(sample, gt_feature(), 5, 2.5, +3.0, ::nano::hinge::left, 0));
    }

    ::nano::hinge hinge() const override
    {
        return ::nano::hinge::left;
    }

    tensor4d_t tables() const override
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), +3.0, -3.0 * threshold());
    }
};

class whinge_right_dataset_t : public whinge_dataset_t
{
public:

    whinge_right_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).full(
            make_hinge_target(sample, gt_feature(), 5, 2.5, -2.1, ::nano::hinge::right, 0));
    }

    ::nano::hinge hinge() const override
    {
        return ::nano::hinge::right;
    }

    tensor4d_t tables() const override
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), -2.1, +2.1 * threshold());
    }
};

UTEST_BEGIN_MODULE(test_gboost_whinge)

UTEST_CASE(fitting_left)
{
    const auto dataset = make_dataset<whinge_left_dataset_t>();
    const auto datasetx1 = make_dataset<whinge_left_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<whinge_left_dataset_t>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<whinge_left_dataset_t>>();

    auto wlearner = make_wlearner<wlearner_hinge_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
}

UTEST_CASE(fitting_right)
{
    const auto dataset = make_dataset<whinge_right_dataset_t>();
    const auto datasetx1 = make_dataset<whinge_right_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<whinge_right_dataset_t>(dataset.gt_feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<whinge_right_dataset_t>>();

    auto wlearner = make_wlearner<wlearner_hinge_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
}

UTEST_END_MODULE()
