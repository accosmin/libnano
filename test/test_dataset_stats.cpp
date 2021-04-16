#include <utest/utest.h>
#include <nano/dataset/stats.h>

using namespace nano;

template <typename tvalue, size_t trank>
static auto const_tensor(tensor_dims_t<trank> dims, tvalue value)
{
    tensor_mem_t<tvalue, trank> data(dims);
    data.full(value);
    return data;
}

template <typename tscalar, size_t trank>
static void check_sclass_stats(
    const feature_t& feature, dataset_iterator_t<tscalar, trank> it,
    tensor_size_t expected_samples, const indices_t& expected_class_counts, tensor1d_t expected_weights,
    scalar_t epsilon = 1e-12)
{
    const auto stats = sclass_stats_t::make(feature, it);

    UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
    UTEST_CHECK_TENSOR_EQUAL(stats.class_counts(), expected_class_counts);
    {
        if (expected_samples > 0)
        {
            expected_weights.array() *= expected_samples / expected_weights.sum();
        }
        const auto weights = stats.sample_weights(feature, it);

        UTEST_CHECK_EQUAL(weights.size(), it.size());
        UTEST_CHECK_CLOSE(weights.sum(), expected_samples, epsilon);
        UTEST_CHECK_TENSOR_CLOSE(weights, expected_weights, epsilon);
    }
    {
        // sample weights for incompatible features
        const auto weights0 = stats.sample_weights(feature_t{""}.sclass(42), it);
        const auto expected_weights0 = const_tensor<scalar_t>(make_dims(it.size()), 0.0);
        UTEST_CHECK_TENSOR_CLOSE(weights0, expected_weights0, epsilon);
    }
}

template <typename tscalar, size_t trank>
static void check_mclass_stats(
    const feature_t& feature, dataset_iterator_t<tscalar, trank> it,
    tensor_size_t expected_samples, const indices_t& expected_class_counts, tensor1d_t expected_weights,
    scalar_t epsilon = 1e-12)
{
    const auto stats = mclass_stats_t::make(feature, it);

    UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
    UTEST_CHECK_TENSOR_EQUAL(stats.class_counts(), expected_class_counts);
    {
        if (expected_samples > 0)
        {
            expected_weights.array() *= expected_samples / expected_weights.sum();
        }
        const auto weights = stats.sample_weights(feature, it);

        UTEST_CHECK_EQUAL(weights.size(), it.size());
        UTEST_CHECK_CLOSE(weights.sum(), expected_samples, epsilon);
        UTEST_CHECK_TENSOR_CLOSE(weights, expected_weights, epsilon);
    }
    {
        // sample weights for incompatible features
        const auto weights0 = stats.sample_weights(feature_t{""}.sclass(42), it);
        const auto expected_weights0 = const_tensor<scalar_t>(make_dims(it.size()), 0.0);
        UTEST_CHECK_TENSOR_CLOSE(weights0, expected_weights0, epsilon);
    }
}

template <typename tscalar, size_t trank>
static void check_scalar_stats(
    const feature_t& feature, dataset_iterator_t<tscalar, trank> it,
    tensor_size_t expected_samples, scalar_t expected_min, scalar_t expected_max, scalar_t expected_mean, scalar_t expected_stdev,
    scalar_t epsilon = 1e-12)
{
    const auto stats = scalar_stats_t::make(feature, it);

    const auto gt_min = const_tensor<scalar_t>(feature.dims(), expected_min);
    const auto gt_max = const_tensor<scalar_t>(feature.dims(), expected_max);
    const auto gt_mean = const_tensor<scalar_t>(feature.dims(), expected_mean);
    const auto gt_stdev = const_tensor<scalar_t>(feature.dims(), expected_stdev);

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
        const auto samples = arange(0, 42);
        const auto feature = feature_t{"feature"}.scalar(feature_type::float32, dims);

        auto mask = make_mask(make_dims(samples.size()));
        auto data = const_tensor<scalar_t>(cat_dims(samples.size(), dims), std::numeric_limits<scalar_t>::quiet_NaN());
        const auto it = make_iterator(data, mask, samples);
        {
            const auto min = std::numeric_limits<scalar_t>::max();
            const auto max = std::numeric_limits<scalar_t>::lowest();
            check_scalar_stats(feature, it, 0, min, max, 0.0, 0.0);
        }
        {
            data.tensor(0).full(1.0);
            setbit(mask, 0);
            check_scalar_stats(feature, it, 1, 1.0, 1.0, 1.0, 0.0);
        }
        {
            for (tensor_size_t sample = 1; sample < samples.size(); sample += 3)
            {
                data.tensor(sample).full(static_cast<scalar_t>(sample));
                setbit(mask, sample);
            }
            check_scalar_stats(feature, it, 15, 1.0, 40.0, 19.2, 13.09961831505);
        }
    }
}

UTEST_CASE(sclass)
{
    const auto samples = arange(0, 20);
    const auto feature = feature_t{"feature"}.sclass(3);

    auto mask = make_mask(make_dims(samples.size()));
    auto data = const_tensor<uint8_t>(make_dims(samples.size()), 0x00);
    const auto it = make_iterator(data, mask, samples);
    {
        check_sclass_stats(feature, it, 0, make_indices(0, 0, 0),
            make_tensor<scalar_t>(make_dims(20),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
    {
        data(0) = static_cast<uint8_t>(0); setbit(mask, 0);

        check_sclass_stats(feature, it, 1, make_indices(1, 0, 0),
            make_tensor<scalar_t>(make_dims(20),
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
    {
        data(1) = static_cast<uint8_t>(1); setbit(mask, 1);
        data(3) = static_cast<uint8_t>(2); setbit(mask, 3);
        data(5) = static_cast<uint8_t>(0); setbit(mask, 5);
        data(6) = static_cast<uint8_t>(1); setbit(mask, 6);
        data(9) = static_cast<uint8_t>(1); setbit(mask, 9);

        check_sclass_stats(feature, it, 6, make_indices(2, 3, 1),
            make_tensor<scalar_t>(make_dims(20),
            1.0/2.0, 1.0/3.0, 0.0, 1.0/1.0, 0.0, 1.0/2.0, 1.0/3.0, 0.0, 0.0, 1.0/3.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
    {
        data(10) = static_cast<uint8_t>(2); setbit(mask, 10);
        data(11) = static_cast<uint8_t>(2); setbit(mask, 11);
        data(13) = static_cast<uint8_t>(2); setbit(mask, 13);
        data(15) = static_cast<uint8_t>(0); setbit(mask, 15);
        data(16) = static_cast<uint8_t>(1); setbit(mask, 16);
        data(19) = static_cast<uint8_t>(1); setbit(mask, 19);

        check_sclass_stats(feature, it, 12, make_indices(3, 5, 4),
            make_tensor<scalar_t>(make_dims(20),
            1.0/3.0, 1.0/5.0, 0.0, 1.0/4.0, 0.0, 1.0/3.0, 1.0/5.0, 0.0, 0.0, 1.0/5.0,
            1.0/4.0, 1.0/4.0, 0.0, 1.0/4.0, 0.0, 1.0/3.0, 1.0/5.0, 0.0, 0.0, 1.0/5.0));
    }
}

UTEST_CASE(mclass)
{
    const auto samples = arange(0, 22);
    const auto feature = feature_t{"feature"}.sclass(3);

    auto mask = make_mask(make_dims(samples.size()));
    auto data = const_tensor<uint8_t>(make_dims(samples.size(), feature.classes()), 0x00);
    const auto it = make_iterator(data, mask, samples);
    {
        check_mclass_stats(feature, it, 0, make_indices(0, 0, 0, 0, 0, 0),
            make_tensor<scalar_t>(make_dims(22),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0));
    }
    {
        data.tensor(3) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 3);
        data.tensor(5) = make_tensor<uint8_t>(make_dims(3), 1, 1, 1); setbit(mask, 5);
        data.tensor(8) = make_tensor<uint8_t>(make_dims(3), 0, 0, 1); setbit(mask, 8);

        check_mclass_stats(feature, it, 3, make_indices(0, 0, 0, 1, 1, 1),
            make_tensor<scalar_t>(make_dims(22),
            0.0, 0.0, 0.0, 1.0/1.0, 0.0, 1.0/1.0, 0.0, 0.0, 1.0/1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0));
    }
    {
        data.tensor(11) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 11);
        data.tensor(12) = make_tensor<uint8_t>(make_dims(3), 1, 1, 1); setbit(mask, 12);
        data.tensor(13) = make_tensor<uint8_t>(make_dims(3), 1, 0, 1); setbit(mask, 13);
        data.tensor(14) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 14);

        check_mclass_stats(feature, it, 7, make_indices(0, 0, 0, 1, 4, 2),
            make_tensor<scalar_t>(make_dims(22),
            0.0, 0.0, 0.0, 1.0/4.0, 0.0, 1.0/2.0, 0.0, 0.0, 1.0/1.0, 0.0,
            0.0, 1.0/4.0, 1.0/2.0, 1.0/4.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0));
    }
    {
        data.tensor(15) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0); setbit(mask, 15);
        data.tensor(16) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0); setbit(mask, 16);
        data.tensor(17) = make_tensor<uint8_t>(make_dims(3), 0, 0, 1); setbit(mask, 17);
        data.tensor(18) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 18);
        data.tensor(19) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1); setbit(mask, 19);
        data.tensor(20) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0); setbit(mask, 20);
        data.tensor(21) = make_tensor<uint8_t>(make_dims(3), 0, 1, 0); setbit(mask, 21);

        check_mclass_stats(feature, it, 14, make_indices(3, 0, 1, 2, 6, 2),
            make_tensor<scalar_t>(make_dims(22),
            0.0, 0.0, 0.0, 1.0/6.0, 0.0, 1.0/2.0, 0.0, 0.0, 1.0/2.0, 0.0,
            0.0, 1.0/6.0, 1.0/2.0, 1.0/6.0, 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/2.0, 1.0/6.0, 1.0/6.0,
            1.0/3.0, 1.0/1.0));
    }
}

UTEST_END_MODULE()
