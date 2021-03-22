#include <utest/utest.h>
#include <nano/dataset/stats.h>

using namespace nano;

template <typename tvalue, size_t trank>
static auto const_tensor(tvalue value, tensor_dims_t<trank> dims)
{
    tensor_mem_t<tvalue, trank> values(dims);
    values.constant(value);
    return values;
}

template <template <typename, size_t> class tstorage, typename tscalar>
static void check_sclass_stats(
    const feature_t& feature, const tensor_t<tstorage, tscalar, 1>& data, indices_cmap_t samples, mask_cmap_t mask,
    const indices_t& gt_class_counts)
{
    const auto stats = feature_sclass_stats_t::make(feature, make_iterator(data, mask, samples));

    UTEST_CHECK_TENSOR_EQUAL(stats.m_class_counts, gt_class_counts);
}

template <template <typename, size_t> class tstorage, typename tscalar>
static void check_mclass_stats(
    const feature_t& feature, const tensor_t<tstorage, tscalar, 2>& data, indices_cmap_t samples, mask_cmap_t mask,
    const indices_t& gt_class_counts)
{
    const auto stats = feature_mclass_stats_t::make(feature, make_iterator(data, mask, samples));

    UTEST_CHECK_TENSOR_EQUAL(stats.m_class_counts, gt_class_counts);
}

template <template <typename, size_t> class tstorage, typename tscalar>
static void check_scalar_stats(
    const feature_t& feature, const tensor_t<tstorage, tscalar, 4>& data, indices_cmap_t samples, mask_cmap_t mask,
    tensor_size_t gt_count, scalar_t gt_min, scalar_t gt_max, scalar_t gt_mean, scalar_t gt_stdev,
    scalar_t epsilon = 1e-12)
{
    const auto stats = feature_scalar_stats_t::make(feature, make_iterator(data, mask, samples));

    const auto expected_min = const_tensor<scalar_t>(gt_min, feature.dims());
    const auto expected_max = const_tensor<scalar_t>(gt_max, feature.dims());
    const auto expected_mean = const_tensor<scalar_t>(gt_mean, feature.dims());
    const auto expected_stdev = const_tensor<scalar_t>(gt_stdev, feature.dims());

    UTEST_CHECK_EQUAL(stats.m_count, gt_count);
    UTEST_CHECK_TENSOR_CLOSE(stats.m_min, expected_min, epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.m_max, expected_max, epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.m_mean, expected_mean, epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.m_stdev, expected_stdev, epsilon);
}

UTEST_BEGIN_MODULE(test_dataset_stats)

UTEST_CASE(scalar)
{
    for (const auto dims : {make_dims(3, 1, 2), make_dims(1, 1, 1)})
    {
        const auto feature = feature_t{"feature"}.scalar(feature_type::float32, dims);

        const auto samples = arange(0, 42);
        auto mask = make_mask(make_dims(samples.size()));

        tensor_mem_t<scalar_t, 4> values(cat_dims(samples.size(), dims));
        values.constant(std::numeric_limits<scalar_t>::quiet_NaN());
        {
            const auto min = std::numeric_limits<scalar_t>::max();
            const auto max = std::numeric_limits<scalar_t>::lowest();
            check_scalar_stats(feature, values, samples, mask, 0, min, max, 0.0, 0.0);
        }
        {
            values.tensor(0).constant(1.0);
            setbit(mask, 0);
            check_scalar_stats(feature, values, samples, mask, 1, 1.0, 1.0, 1.0, 0.0);
        }
        {
            for (tensor_size_t sample = 1; sample < samples.size(); sample += 3)
            {
                values.tensor(sample).constant(static_cast<scalar_t>(sample));
                setbit(mask, sample);
            }
            check_scalar_stats(feature, values, samples, mask, 15, 1.0, 40.0, 19.2, 13.09961831505);
        }
    }
}

UTEST_CASE(sclass)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto samples = arange(0, 42);
    auto mask = make_mask(make_dims(samples.size()));

    tensor_mem_t<uint8_t, 1> values(samples.size());
    values.zero();
    {
        check_sclass_stats(feature, values, samples, mask, make_tensor<tensor_size_t>(make_dims(3), 0, 0, 0));
    }
    {
        for (tensor_size_t sample = 1; sample < samples.size(); sample += 7)
        {
            values(sample) = static_cast<uint8_t>(sample % 3);
            setbit(mask, sample);
        }
        check_sclass_stats(feature, values, samples, mask, make_tensor<tensor_size_t>(make_dims(3), 2, 2, 2));
    }
}

UTEST_CASE(mclass)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto samples = arange(0, 42);
    auto mask = make_mask(make_dims(samples.size()));

    tensor_mem_t<uint8_t, 2> values(samples.size(), feature.classes());
    values.zero();
    {
        check_mclass_stats(feature, values, samples, mask, make_tensor<tensor_size_t>(make_dims(3), 0, 0, 0));
    }
    {
        values.tensor(3) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1);
        values.tensor(5) = make_tensor<uint8_t>(make_dims(3), 1, 1, 1);
        values.tensor(8) = make_tensor<uint8_t>(make_dims(3), 0, 0, 1);
        setbit(mask, 3);
        setbit(mask, 5);
        setbit(mask, 8);
        check_mclass_stats(feature, values, samples, mask, make_tensor<tensor_size_t>(make_dims(3), 1, 2, 3));
    }
}

UTEST_END_MODULE()
