#include <utest/utest.h>
#include <nano/dataset/stats.h>

using namespace nano;

template <typename tvalue, size_t trank>
static auto const_tensor(tvalue value, tensor_dims_t<trank> dims)
{
    tensor_mem_t<tvalue, trank> values(dims);
    values.full(value);
    return values;
}

template <template <typename, size_t> class tstorage, typename tscalar>
static void check_sclass_stats(
    const feature_t& feature, const tensor_t<tstorage, tscalar, 1>& data, indices_cmap_t samples, mask_cmap_t mask,
    tensor_size_t expected_samples, const indices_t& expected_class_counts)
{
    const auto stats = sclass_stats_t::make(feature, make_iterator(data, mask, samples));

    UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
    UTEST_CHECK_TENSOR_EQUAL(stats.class_counts(), expected_class_counts);

    // TODO: check sample weights
}

template <template <typename, size_t> class tstorage, typename tscalar>
static void check_mclass_stats(
    const feature_t& feature, const tensor_t<tstorage, tscalar, 2>& data, indices_cmap_t samples, mask_cmap_t mask,
    tensor_size_t expected_samples, const indices_t& expected_class_counts)
{
    const auto stats = mclass_stats_t::make(feature, make_iterator(data, mask, samples));

    UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
    UTEST_CHECK_TENSOR_EQUAL(stats.class_counts(), expected_class_counts);

    // TODO: check sample weights
}

template <template <typename, size_t> class tstorage, typename tscalar>
static void check_scalar_stats(
    const feature_t& feature, const tensor_t<tstorage, tscalar, 4>& data, indices_cmap_t samples, mask_cmap_t mask,
    tensor_size_t expected_samples, scalar_t expected_min, scalar_t expected_max, scalar_t expected_mean, scalar_t expected_stdev,
    scalar_t epsilon = 1e-12)
{
    const auto stats = scalar_stats_t::make(feature, make_iterator(data, mask, samples));

    const auto gt_min = const_tensor<scalar_t>(expected_min, feature.dims());
    const auto gt_max = const_tensor<scalar_t>(expected_max, feature.dims());
    const auto gt_mean = const_tensor<scalar_t>(expected_mean, feature.dims());
    const auto gt_stdev = const_tensor<scalar_t>(expected_stdev, feature.dims());

    UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
    UTEST_CHECK_TENSOR_CLOSE(stats.min(), gt_min.reshape(-1), epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.max(), gt_max.reshape(-1), epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.mean(), gt_mean.reshape(-1), epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.stdev(), gt_stdev.reshape(-1), epsilon);
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
        values.full(std::numeric_limits<scalar_t>::quiet_NaN());
        {
            const auto min = std::numeric_limits<scalar_t>::max();
            const auto max = std::numeric_limits<scalar_t>::lowest();
            check_scalar_stats(feature, values, samples, mask, 0, min, max, 0.0, 0.0);
        }
        {
            values.tensor(0).full(1.0);
            setbit(mask, 0);
            check_scalar_stats(feature, values, samples, mask, 1, 1.0, 1.0, 1.0, 0.0);
        }
        {
            for (tensor_size_t sample = 1; sample < samples.size(); sample += 3)
            {
                values.tensor(sample).full(static_cast<scalar_t>(sample));
                setbit(mask, sample);
            }
            check_scalar_stats(feature, values, samples, mask, 15, 1.0, 40.0, 19.2, 13.09961831505);
        }
    }
}

UTEST_CASE(sclass)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto samples = arange(0, 20);
    auto mask = make_mask(make_dims(samples.size()));

    tensor_mem_t<uint8_t, 1> values(samples.size());
    values.zero();
    {
        check_sclass_stats(feature, values, samples, mask,
            0, make_indices(0, 0, 0));
    }
    {
        values(0) = static_cast<uint8_t>(0); setbit(mask, 0);

        check_sclass_stats(feature, values, samples, mask,
            1, make_indices(1, 0, 0));
    }
    {
        values(1) = static_cast<uint8_t>(1); setbit(mask, 1);
        values(3) = static_cast<uint8_t>(2); setbit(mask, 3);
        values(5) = static_cast<uint8_t>(0); setbit(mask, 5);
        values(6) = static_cast<uint8_t>(1); setbit(mask, 6);
        values(9) = static_cast<uint8_t>(1); setbit(mask, 9);

        check_sclass_stats(feature, values, samples, mask,
            6, make_indices(2, 3, 1));
    }
    {
        values(10) = static_cast<uint8_t>(2); setbit(mask, 10);
        values(11) = static_cast<uint8_t>(2); setbit(mask, 11);
        values(13) = static_cast<uint8_t>(2); setbit(mask, 13);
        values(15) = static_cast<uint8_t>(0); setbit(mask, 15);
        values(16) = static_cast<uint8_t>(1); setbit(mask, 16);
        values(19) = static_cast<uint8_t>(1); setbit(mask, 19);

        check_sclass_stats(feature, values, samples, mask,
            12, make_indices(3, 5, 4));
    }
}

UTEST_CASE(mclass)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto samples = arange(0, 22);
    auto mask = make_mask(make_dims(samples.size()));

    tensor_mem_t<uint8_t, 2> values(samples.size(), feature.classes());
    values.zero();
    {
        check_mclass_stats(feature, values, samples, mask,
            0, make_indices(0, 0, 0, 0, 0, 0));
    }
    {
        values.tensor(3) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 3);
        values.tensor(5) = make_tensor<uint8_t>(make_dims(3), 1, 1, 1); setbit(mask, 5);
        values.tensor(8) = make_tensor<uint8_t>(make_dims(3), 0, 0, 1); setbit(mask, 8);

        check_mclass_stats(feature, values, samples, mask,
            3, make_indices(0, 0, 0, 1, 1, 1));
    }
    {
        values.tensor(11) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 11);
        values.tensor(12) = make_tensor<uint8_t>(make_dims(3), 1, 1, 1); setbit(mask, 12);
        values.tensor(13) = make_tensor<uint8_t>(make_dims(3), 1, 0, 1); setbit(mask, 13);
        values.tensor(14) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 14);

        check_mclass_stats(feature, values, samples, mask,
            7, make_indices(0, 0, 0, 1, 4, 2));
    }
    {
        values.tensor(15) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0); setbit(mask, 15);
        values.tensor(16) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0); setbit(mask, 16);
        values.tensor(17) = make_tensor<uint8_t>(make_dims(3), 0, 0, 1); setbit(mask, 17);
        values.tensor(18) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 18);
        values.tensor(19) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 19);
        values.tensor(20) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0); setbit(mask, 20);
        values.tensor(21) = make_tensor<uint8_t>(make_dims(3), 0, 1, 0); setbit(mask, 21);

        check_mclass_stats(feature, values, samples, mask,
            14, make_indices(3, 0, 1, 2, 6, 2));
    }
}

UTEST_END_MODULE()
