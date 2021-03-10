#include <utest/utest.h>
#include <nano/dataset/memory_dataset.h>

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

/*
static auto make_features()
{
    return features_t
    {
        feature_t{"i8"}.scalar(feature_type::int8),
        feature_t{"i16"}.scalar(feature_type::int16),
        feature_t{"i32"}.scalar(feature_type::int32),
        feature_t{"i64"}.scalar(feature_type::int64),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"f64"}.scalar(feature_type::float64),

        feature_t{"ui8_struct"}.scalar(feature_type::uint8, make_dims(3, 8, 8)),
        feature_t{"ui16_struct"}.scalar(feature_type::uint16, make_dims(3, 8, 8)),
        feature_t{"ui32_struct"}.scalar(feature_type::uint32, make_dims(3, 8, 8)),
        feature_t{"ui64_struct"}.scalar(feature_type::uint64, make_dims(3, 8, 8)),

        feature_t{"sclass2"}.sclass(2),
        feature_t{"sclass10"}.sclass(10),

        feature_t{"mclass10"}.mclass(10),
    };
}

class fixture_dataset_t final : public memory_dataset_t
{
public:

    fixture_dataset_t(tensor_size_t samples, features_t features, size_t target) :
        m_samples(samples),
        m_features(std::move(features)),
        m_target(target)
    {
    }

    void load() override
    {
        resize(m_samples, m_features, m_target);

        // scalars
        for (tensor_size_t feature = 0; feature < 6; ++ feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; sample += feature + 1)
            {
                check_set_scalar(sample, feature, sample + feature, make_dims(1, 1, 1));
            }
        }

        // structured
        for (tensor_size_t feature = 6; feature < 10; ++ feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; sample += feature + 1)
            {
                check_set_scalar(sample, feature, sample % feature, make_dims(3, 8, 8));
            }
        }

        // single label
        for (tensor_size_t sample = 0, feature = 10; sample < m_samples; sample += 2)
        {
            check_set_sclass(sample, feature, sample % 2, 2);
        }
        for (tensor_size_t sample = 0, feature = 11; sample < m_samples; sample += 3)
        {
            check_set_sclass(sample, feature, sample % 10, 10);
        }

        // multi label
        for (tensor_size_t sample = 0, feature = 12; sample < m_samples; sample += 4)
        {
            check_set_mclass(sample, feature, sample % 10, 10);
        }
    }

private:

    template <typename tvalue>
    void check_set(tensor_size_t sample, tensor_size_t feature, const tvalue& value)
    {
        set(sample, feature, value);
    }

    tensor_size_t   m_samples{0};
    features_t      m_features;
    size_t          m_target;
};

static auto make_dataset(tensor_size_t samples, const features_t& features, size_t target)
{
    auto dataset = fixture_dataset_t{samples, features, target};
    UTEST_CHECK_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    return dataset;
}*/

UTEST_BEGIN_MODULE(test_dataset_memory)

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
        {
            const auto sample = 0;
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
    {
        const auto sample = 2;
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
    {
        const auto sample = 11;
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

/*
UTEST_CASE(dataset)
{
    const auto mask0 = mask_t{make_dims(6), {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC0}};
    const auto mask1 = mask_t{make_dims(6), {0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0x80}};
    const auto mask2 = mask_t{make_dims(6), {0x92, 0x49, 0x24, 0x92, 0x49, 0x00}};
    const auto mask3 = mask_t{make_dims(6), {0x88, 0x88, 0x88, 0x88, 0x88, 0x80}};
    const auto mask4 = mask_t{make_dims(6), {0x84, 0x21, 0x08, 0x42, 0x10, 0x80}};
    const auto mask5 = mask_t{make_dims(6), {0x82, 0x08, 0x20, 0x82, 0x08, 0x00}};
    const auto mask6 = mask_t{make_dims(6), {0x81, 0x02, 0x04, 0x08, 0x10, 0x00}};
    const auto mask7 = mask_t{make_dims(6), {0x80, 0x80, 0x80, 0x80, 0x80, 0x80}};
    const auto mask8 = mask_t{make_dims(6), {0x80, 0x40, 0x20, 0x10, 0x08, 0x00}};
    const auto mask9 = mask_t{make_dims(6), {0x80, 0x20, 0x08, 0x02, 0x00, 0x80}};
    const auto mask10 = mask1;
    const auto mask11 = mask2;
    const auto mask12 = mask3;

    const auto samples = ::nano::arange(0, 42);

    const auto features = make_features();
    for (const auto target : {size_t{0U}, size_t{6U}, size_t{10U}, size_t{12U}, string_t::npos})
    {
        const auto dataset = make_dataset(42, features, target);

        UTEST_CHECK_TENSOR_EQUAL(mask0, dataset.mask(0));
        UTEST_CHECK_TENSOR_EQUAL(mask1, dataset.mask(1));
        UTEST_CHECK_TENSOR_EQUAL(mask2, dataset.mask(2));
        UTEST_CHECK_TENSOR_EQUAL(mask3, dataset.mask(3));
        UTEST_CHECK_TENSOR_EQUAL(mask4, dataset.mask(4));
        UTEST_CHECK_TENSOR_EQUAL(mask5, dataset.mask(5));
        UTEST_CHECK_TENSOR_EQUAL(mask6, dataset.mask(6));
        UTEST_CHECK_TENSOR_EQUAL(mask7, dataset.mask(7));
        UTEST_CHECK_TENSOR_EQUAL(mask8, dataset.mask(8));
        UTEST_CHECK_TENSOR_EQUAL(mask9, dataset.mask(9));
        UTEST_CHECK_TENSOR_EQUAL(mask10, dataset.mask(10));
        UTEST_CHECK_TENSOR_EQUAL(mask11, dataset.mask(11));
        UTEST_CHECK_TENSOR_EQUAL(mask12, dataset.mask(12));

        check_scalar_stats(dataset.storage(0), samples, dataset.mask(0), 42, 0, 41, 20.5, 12.267844146385);
        check_scalar_stats(dataset.storage(1), samples, dataset.mask(1), 21, 1, 41, 21.0, 12.409673645991);
        check_scalar_stats(dataset.storage(2), samples, dataset.mask(2), 14, 2, 41, 21.5, 12.549900398011);
        check_scalar_stats(dataset.storage(3), samples, dataset.mask(3), 11, 3, 43, 23.0, 13.266499161422);
        check_scalar_stats(dataset.storage(4), samples, dataset.mask(4), 9, 4, 44, 24.0, 13.693063937629);
        check_scalar_stats(dataset.storage(5), samples, dataset.mask(5), 7, 5, 41, 23.0, 12.961481396816);
        check_scalar_stats(dataset.storage(6), samples, dataset.mask(6), 6, 0, 5, 2.5, 1.870828693387);
        check_scalar_stats(dataset.storage(7), samples, dataset.mask(7), 6, 0, 5, 2.5, 1.870828693387);
        check_scalar_stats(dataset.storage(8), samples, dataset.mask(8), 5, 0, 4, 2.0, 1.581138830084);
        check_scalar_stats(dataset.storage(9), samples, dataset.mask(9), 5, 0, 4, 2.0, 1.581138830084);
        check_sclass_stats(dataset.storage(10), samples, dataset.mask(10), indices_t{make_dims(2), {21, 0}});
        check_sclass_stats(dataset.storage(11), samples, dataset.mask(11), indices_t{make_dims(10), {2, 1, 1, 2, 1, 1, 2, 1, 1, 2}});
        check_mclass_stats(dataset.storage(12), samples, dataset.mask(12), indices_t{make_dims(10), {11, 0, 2, 0, 2, 0, 2, 0, 2, 0}});

        // TODO: check feature indices selection
        // TODO: check feature access with both iterators

        // TODO: create in-memory dataset with various feature types (sclass, mclass, scalar or structured) w/o optional
        // TODO: check that the flatten & the feature iterators work as expected
        // TODO: check that feature normalization works
        // TODO: check that feature extraction works (e.g sign(x), sign(x)*log(1+x^2), polynomial expansion)
    }
}*/

UTEST_END_MODULE()
