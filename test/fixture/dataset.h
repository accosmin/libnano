#include <utest/utest.h>
#include <nano/dataset.h>

using namespace nano;

template <typename tscalar, size_t trank>
static auto check_inputs(const dataset_t& dataset, tensor_size_t index,
    const feature_t& gt_feature, const tensor_mem_t<tscalar, trank>& gt_data, const mask_cmap_t& gt_mask)
{
    dataset.visit_inputs(index, [&] (const auto& feature, const auto& data, const auto& mask)
    {
        UTEST_CHECK_EQUAL(feature, gt_feature);
        if constexpr (std::is_same<decltype(data), const tensor_cmap_t<tscalar, trank>&>::value)
        {
            UTEST_CHECK_TENSOR_CLOSE(data, gt_data, 1e-12);
            UTEST_CHECK_TENSOR_EQUAL(mask, gt_mask);
        }
        else
        {
            UTEST_CHECK(false);
        }
    });
}

template <typename tscalar, size_t trank>
static auto check_target(const dataset_t& dataset,
    const feature_t& gt_feature, const tensor_mem_t<tscalar, trank>& gt_data, const mask_cmap_t& gt_mask)
{
    dataset.visit_target([&] (const auto& feature, const auto& data, const auto& mask)
    {
        UTEST_CHECK_EQUAL(feature, gt_feature);
        if constexpr (std::is_same<decltype(data), const tensor_cmap_t<tscalar, trank>&>::value)
        {
            UTEST_CHECK_TENSOR_CLOSE(data, gt_data, 1e-12);
            UTEST_CHECK_TENSOR_EQUAL(mask, gt_mask);
        }
        else
        {
            UTEST_CHECK(false);
        }
    });
}
