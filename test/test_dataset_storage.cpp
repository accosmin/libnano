#include <utest/utest.h>
#include <nano/dataset/storage.h>

using namespace nano;

template <typename tvalue, size_t trank>
static auto make_tensor(tvalue value, tensor_dims_t<trank> dims)
{
    tensor_mem_t<tvalue, trank> values(dims);
    values.constant(value);
    return values;
}

template <typename ttensor>
static void check_scalar_stats(
    const feature_storage_t& storage, const ttensor& values, const indices_cmap_t& samples, const mask_cmap_t& mask,
    tensor_size_t gt_count, scalar_t gt_min, scalar_t gt_max, scalar_t gt_mean, scalar_t gt_stdev,
    scalar_t epsilon = 1e-12)
{
    feature_scalar_stats_t stats;
    UTEST_CHECK_NOTHROW(stats = storage.stats(values.tensor(), samples, mask));

    const auto expected_min = make_tensor<scalar_t>(gt_min, storage.dims());
    const auto expected_max = make_tensor<scalar_t>(gt_max, storage.dims());
    const auto expected_mean = make_tensor<scalar_t>(gt_mean, storage.dims());
    const auto expected_stdev = make_tensor<scalar_t>(gt_stdev, storage.dims());

    UTEST_CHECK_EQUAL(stats.m_count, gt_count);
    UTEST_CHECK_TENSOR_CLOSE(stats.m_min, expected_min, epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.m_max, expected_max, epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.m_mean, expected_mean, epsilon);
    UTEST_CHECK_TENSOR_CLOSE(stats.m_stdev, expected_stdev, epsilon);
}

template <typename ttensor>
static void check_sclass_stats(
    const feature_storage_t& storage, const ttensor& values, const indices_cmap_t& samples, const mask_cmap_t& mask,
    const indices_t& gt_class_counts)
{
    feature_sclass_stats_t stats;
    UTEST_CHECK_NOTHROW(stats = storage.stats(values.tensor(), samples, mask));

    UTEST_CHECK_TENSOR_EQUAL(stats.m_class_counts, gt_class_counts);
}

template <typename ttensor>
static void check_mclass_stats(
    const feature_storage_t& storage, const ttensor& values, const indices_cmap_t& samples, const mask_cmap_t& mask,
    const indices_t& gt_class_counts)
{
    feature_mclass_stats_t stats;
    UTEST_CHECK_NOTHROW(stats = storage.stats(values.tensor(), samples, mask));

    UTEST_CHECK_TENSOR_EQUAL(stats.m_class_counts, gt_class_counts);
}

UTEST_BEGIN_MODULE(test_dataset_storage)

UTEST_CASE(mask)
{
    for (const tensor_size_t samples : {1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33})
    {
        auto mask = make_mask(make_dims(samples));
        UTEST_CHECK_EQUAL(mask.size(), ((samples + 7) / 8));
        UTEST_CHECK(optional(mask, samples));

        for (auto sample = tensor_size_t{0}; sample < samples; ++ sample)
        {
            UTEST_CHECK(!getbit(mask, sample));
        }

        for (auto sample = tensor_size_t{0}; sample < samples; sample += 3)
        {
            setbit(mask, sample);
        }
        UTEST_CHECK(optional(mask, samples) == (samples > 1));

        for (auto sample = tensor_size_t{0}; sample < samples; ++ sample)
        {
            const auto bit = (sample % 3) == 0;
            UTEST_CHECK(getbit(mask, sample) == bit);
        }

        for (auto sample = tensor_size_t{0}; sample < samples; ++ sample)
        {
            setbit(mask, sample);
        }
        UTEST_CHECK(!optional(mask, samples));

        for (auto sample = tensor_size_t{0}; sample < samples; ++ sample)
        {
            UTEST_CHECK(getbit(mask, sample));
        }
    }
}

UTEST_CASE(scalar_stats)
{
    for (const auto dims : {make_dims(3, 1, 2), make_dims(1, 1, 1)})
    {
        const auto feature = feature_t{"feature"}.scalar(feature_type::float32, dims);

        const auto storage = feature_storage_t{feature};
        UTEST_CHECK_EQUAL(storage.dims(), dims);
        UTEST_CHECK_EQUAL(storage.classes(), 0);
        UTEST_CHECK_EQUAL(storage.name(), "feature");
        UTEST_CHECK_EQUAL(storage.feature(), feature);

        const auto samples = arange(0, 42);
        auto mask = make_mask(make_dims(samples.size()));

        tensor_mem_t<scalar_t, 4> values(cat_dims(samples.size(), dims));
        values.constant(std::numeric_limits<scalar_t>::quiet_NaN());
        {
            const auto min = std::numeric_limits<scalar_t>::max();
            const auto max = std::numeric_limits<scalar_t>::lowest();
            check_scalar_stats(storage, values, samples, mask, 0, min, max, 0.0, 0.0);
        }
        {
            UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), 0, make_tensor<int>(1, dims)));
            setbit(mask, 0);
            check_scalar_stats(storage, values, samples, mask, 1, 1.0, 1.0, 1.0, 0.0);
        }
        {
            for (tensor_size_t sample = 1; sample < samples.size(); sample += 3)
            {
                if (::nano::size(dims) == 1)
                {
                    if (sample > 11)
                    {
                        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, scat(sample)));
                    }
                    else
                    {
                        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, sample));
                    }
                }
                else
                {
                    UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, make_tensor(sample, dims)));
                }
                setbit(mask, sample);
            }
            check_scalar_stats(storage, values, samples, mask, 15, 1.0, 40.0, 19.2, 13.09961831505);
        }
    }
}

UTEST_CASE(sclass_stats)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto storage = feature_storage_t{feature};
    UTEST_CHECK_EQUAL(storage.classes(), 3);
    UTEST_CHECK_EQUAL(storage.name(), "feature");
    UTEST_CHECK_EQUAL(storage.feature(), feature);

    const auto samples = arange(0, 42);
    auto mask = make_mask(make_dims(samples.size()));

    tensor_mem_t<uint8_t, 1> values(samples.size());
    values.zero();
    {
        check_sclass_stats(storage, values, samples, mask, indices_t{make_dims(3), {0, 0, 0}});
    }
    {
        for (tensor_size_t sample = 1; sample < samples.size(); sample += 7)
        {
            if (sample > 17)
            {
                UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, scat(sample % 3)));
            }
            else
            {
                UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, sample % 3));
            }
            setbit(mask, sample);
        }
        check_sclass_stats(storage, values, samples, mask, indices_t{make_dims(3), {2, 2, 2}});
    }
}

UTEST_CASE(mclass_stats)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto storage = feature_storage_t{feature};
    UTEST_CHECK_EQUAL(storage.classes(), 3);
    UTEST_CHECK_EQUAL(storage.name(), "feature");
    UTEST_CHECK_EQUAL(storage.feature(), feature);

    const auto samples = arange(0, 42);
    auto mask = make_mask(make_dims(samples.size()));

    tensor_mem_t<uint8_t, 2> values(samples.size(), feature.classes());
    values.zero();
    {
        check_mclass_stats(storage, values, samples, mask, indices_t{make_dims(3), {0, 0, 0}});
    }
    {
        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), 3, indices_t{make_dims(3), {0, 1, 1}}));
        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), 5, indices_t{make_dims(3), {1, 1, 1}}));
        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), 8, indices_t{make_dims(3), {0, 0, 1}}));
        setbit(mask, 3);
        setbit(mask, 5);
        setbit(mask, 8);
        check_mclass_stats(storage, values, samples, mask, indices_t{make_dims(3), {1, 2, 3}});
    }
}

UTEST_CASE(storage_scalar)
{
    for (const auto dims : {make_dims(3, 1, 2), make_dims(1, 1, 1)})
    {
        const auto feature = feature_t{"feature"}.scalar(feature_type::float32, dims);

        const auto storage = feature_storage_t{feature};
        UTEST_CHECK_EQUAL(storage.dims(), dims);
        UTEST_CHECK_EQUAL(storage.classes(), 0);
        UTEST_CHECK_EQUAL(storage.name(), "feature");
        UTEST_CHECK_EQUAL(storage.feature(), feature);

        tensor_mem_t<scalar_t, 4> values(cat_dims(42, dims));
        values.constant(std::numeric_limits<scalar_t>::quiet_NaN());

        for (tensor_size_t sample : {0, 11})
        {
            const auto value = 14.6f;
            const auto expected_value = make_tensor<scalar_t>(value, dims);

            // check if possible to set with scalar
            if (::nano::size(dims) == 1)
            {
                UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, value));
                UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, std::to_string(value)));
            }
            else
            {
                UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, value), std::runtime_error);
            }

            // should be possible to set with compatible tensor
            const auto values3d = make_tensor(value, dims);
            UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, values3d));

            const auto values1d = make_tensor(value, make_dims(::nano::size(dims)));
            UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, values1d));

            // cannot set with incompatible tensor
            {
                const auto values_nok = make_tensor(value, make_dims(::nano::size(dims) + 1));
                UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
            }
            {
                const auto [dim0, dim1, dim2] = dims;
                const auto values_nok = make_tensor(value, make_dims(dim0, dim1 + 1, dim2));
                UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
            }

            // cannot set with invalid string
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, "N/A"), std::runtime_error);

            // the expected feature value should be there
            UTEST_CHECK_TENSOR_CLOSE(values.tensor(sample), expected_value, 1e-12);
        }
    }
}

UTEST_CASE(storage_sclass)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto storage = feature_storage_t{feature};
    UTEST_CHECK_EQUAL(storage.classes(), 3);
    UTEST_CHECK_EQUAL(storage.name(), "feature");
    UTEST_CHECK_EQUAL(storage.feature(), feature);

    tensor_mem_t<uint8_t, 1> values(42);
    values.zero();
    for (tensor_size_t sample : {2, 7})
    {
        const auto value = feature.classes() - 1;
        const auto expected_value = value;

        // cannot set multi-label indices
        for (const auto& values_nok : {
            make_tensor(value, make_dims(1)),
            make_tensor(value, make_dims(feature.classes())),
        })
        {
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
        }

        // cannot set multivariate scalars
        for (const auto& values_nok : {
            make_tensor(value, make_dims(1, 1, 1)),
            make_tensor(value, make_dims(2, 1, 3)),
        })
        {
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
        }

        // cannot set with out-of-bounds class indices
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, -1), std::runtime_error);
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, feature.classes()), std::runtime_error);
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, feature.classes() + 1), std::runtime_error);

        // check if possible to set with valid class index
        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, value));

        // the expected feature value should be there
        UTEST_CHECK_EQUAL(values(sample), expected_value);
    }
}

UTEST_CASE(storage_mclass)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto storage = feature_storage_t{feature};
    UTEST_CHECK_EQUAL(storage.classes(), 3);
    UTEST_CHECK_EQUAL(storage.name(), "feature");
    UTEST_CHECK_EQUAL(storage.feature(), feature);

    tensor_mem_t<uint8_t, 2> values(42, feature.classes());
    values.zero();
    for (tensor_size_t sample : {11, 17})
    {
        const auto value = tensor_mem_t<uint16_t, 1>{make_dims(feature.classes()), {1, 0, 1}};
        const auto expected_value = tensor_mem_t<uint8_t, 1>{make_dims(feature.classes()), {1, 0, 1}};

        // cannot set multi-label indices of invalid size
        for (const auto& values_nok : {
            make_tensor(0, make_dims(feature.classes() - 1)),
            make_tensor(0, make_dims(feature.classes() + 1)),
        })
        {
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
        }

        // cannot set scalars or strings
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, 1), std::runtime_error);
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, "2"), std::runtime_error);

        // cannot set multivariate scalars
        for (const auto& values_nok : {
            make_tensor(1, make_dims(1, 1, 1)),
            make_tensor(1, make_dims(2, 1, 3)),
        })
        {
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
        }

        // check if possible to set with valid class hits
        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, value));

        // the expected feature value should be there
        UTEST_CHECK_TENSOR_EQUAL(values.tensor(sample), expected_value);
    }
}

UTEST_END_MODULE()
