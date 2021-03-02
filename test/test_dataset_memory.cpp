#include <utest/utest.h>
#include <nano/dataset/memory.h>

using namespace nano;

template <typename tvalue, size_t trank>
static auto make_tensor(tvalue value, tensor_dims_t<trank> dims)
{
    tensor_mem_t<tvalue, trank> values(dims);
    values.constant(value);
    return values;
}

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

static void check_scalar_stats(
    const feature_storage_t& fs, const indices_cmap_t& samples, const mask_cmap_t& mask,
    tensor_size_t gt_count, scalar_t gt_min, scalar_t gt_max, scalar_t gt_mean, scalar_t gt_stdev,
    scalar_t epsilon = 1e-12)
{
    UTEST_CHECK_THROW(fs.sclass_stats(samples, mask), std::runtime_error);
    UTEST_CHECK_THROW(fs.mclass_stats(samples, mask), std::runtime_error);

    feature_scalar_stats_t stats;
    UTEST_CHECK_NOTHROW(stats = fs.scalar_stats(samples, mask));

    UTEST_CHECK_EQUAL(stats.m_min.dims(), fs.dims());
    UTEST_CHECK_EQUAL(stats.m_max.dims(), fs.dims());
    UTEST_CHECK_EQUAL(stats.m_mean.dims(), fs.dims());
    UTEST_CHECK_EQUAL(stats.m_stdev.dims(), fs.dims());

    UTEST_CHECK_EQUAL(stats.m_count, gt_count);
    UTEST_CHECK_CLOSE(stats.m_min.min(), gt_min, epsilon);
    UTEST_CHECK_CLOSE(stats.m_min.max(), gt_min, epsilon);
    UTEST_CHECK_CLOSE(stats.m_max.min(), gt_max, epsilon);
    UTEST_CHECK_CLOSE(stats.m_max.max(), gt_max, epsilon);
    UTEST_CHECK_CLOSE(stats.m_mean.min(), gt_mean, epsilon);
    UTEST_CHECK_CLOSE(stats.m_mean.max(), gt_mean, epsilon);
    UTEST_CHECK_CLOSE(stats.m_stdev.min(), gt_stdev, epsilon);
    UTEST_CHECK_CLOSE(stats.m_stdev.max(), gt_stdev, epsilon);
}

static void check_sclass_stats(
    const feature_storage_t& fs, const indices_cmap_t& samples, const mask_cmap_t& mask,
    const indices_t& gt_class_counts)
{
    UTEST_CHECK_THROW(fs.scalar_stats(samples, mask), std::runtime_error);
    UTEST_CHECK_THROW(fs.mclass_stats(samples, mask), std::runtime_error);

    feature_sclass_stats_t stats;
    UTEST_CHECK_NOTHROW(stats = fs.sclass_stats(samples, mask));

    UTEST_CHECK_TENSOR_EQUAL(stats.m_class_counts, gt_class_counts);
}

static void check_mclass_stats(
    const feature_storage_t& fs, const indices_cmap_t& samples, const mask_cmap_t& mask,
    const indices_t& gt_class_counts)
{
    UTEST_CHECK_THROW(fs.scalar_stats(samples, mask), std::runtime_error);
    UTEST_CHECK_THROW(fs.sclass_stats(samples, mask), std::runtime_error);

    feature_mclass_stats_t stats;
    UTEST_CHECK_NOTHROW(stats = fs.mclass_stats(samples, mask));

    UTEST_CHECK_TENSOR_EQUAL(stats.m_class_counts, gt_class_counts);
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

    template <typename tvalue>
    void check_set_scalar(tensor_size_t sample, tensor_size_t feature, tvalue value, tensor3d_dims_t dims)
    {
        // check if possible to set with scalar
        if (::nano::size(dims) == 1)
        {
            UTEST_REQUIRE_NOTHROW(check_set(sample, feature, value));
            UTEST_REQUIRE_NOTHROW(check_set(sample, feature, std::to_string(value)));
        }
        else
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, value), std::runtime_error);
        }

        // should be possible to set with compatible tensor
        const auto values = make_tensor<tvalue, 3>(value, dims);
        UTEST_REQUIRE_NOTHROW(check_set(sample, feature, values));

        const auto values1d = make_tensor<tvalue, 1>(value, make_dims(::nano::size(dims)));
        UTEST_REQUIRE_NOTHROW(check_set(sample, feature, values1d));

        // cannot set with incompatible tensor
        {
            const auto values_nok = make_tensor<tvalue, 1>(value, make_dims(::nano::size(dims) + 1));
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }
        {
            const auto [dim0, dim1, dim2] = dims;
            const auto values_nok = make_tensor<tvalue, 3>(value, make_dims(dim0, dim1 + 1, dim2));
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // cannot set with invalid string
        UTEST_REQUIRE_THROW(check_set(sample, feature, "N/A"), std::runtime_error);

        // cannot set if the sample index is invalid
        for (const auto invalid_sample : {-1, 10001})
        {
            if (::nano::size(dims) == 1)
            {
                UTEST_REQUIRE_THROW(check_set(invalid_sample, feature, value), std::runtime_error);
            }
            UTEST_REQUIRE_THROW(check_set(invalid_sample, feature, values), std::runtime_error);
        }
    }

    template <typename tvalue>
    void check_set_sclass(tensor_size_t sample, tensor_size_t feature, tvalue value, tensor_size_t classes)
    {
        // cannot set multi-label indices
        for (const auto& values_nok : {
            make_tensor<tvalue, 1>(value, make_dims(1)),
            make_tensor<tvalue, 1>(value, make_dims(classes)),
        })
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // cannot set multivariate scalars
        for (const auto& values_nok : {
            make_tensor<tvalue, 3>(value, make_dims(1, 1, 1)),
            make_tensor<tvalue, 3>(value, make_dims(2, 1, 3)),
        })
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // cannot set with out-of-bounds class indices
        UTEST_REQUIRE_THROW(check_set(sample, feature, static_cast<tvalue>(-1)), std::runtime_error);
        UTEST_REQUIRE_THROW(check_set(sample, feature, static_cast<tvalue>(classes)), std::runtime_error);
        UTEST_REQUIRE_THROW(check_set(sample, feature, static_cast<tvalue>(classes + 1)), std::runtime_error);

        // check if possible to set with valid class index
        UTEST_REQUIRE_NOTHROW(check_set(sample, feature, value));

        // cannot set if the sample index is invalid
        for (const auto invalid_sample : {-1, 10001})
        {
            UTEST_REQUIRE_THROW(check_set(invalid_sample, feature, value), std::runtime_error);
        }
    }

    template <typename tvalue>
    void check_set_mclass(tensor_size_t sample, tensor_size_t feature, tvalue value, tensor_size_t classes)
    {
        // cannot set multi-label indices of invalid size
        for (const auto& values_nok : {
            make_tensor<tvalue, 1>(0, make_dims(classes + 1)),
            make_tensor<tvalue, 1>(0, make_dims(classes + 2)),
        })
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // cannot set scalars
        UTEST_REQUIRE_THROW(check_set(sample, feature, value), std::runtime_error);

        // cannot set multivariate scalars
        for (const auto& values_nok : {
            make_tensor<tvalue, 3>(value, make_dims(1, 1, 1)),
            make_tensor<tvalue, 3>(value, make_dims(2, 1, 3)),
        })
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // check if possible to set with valid class hits
        tensor_mem_t<tvalue, 1> values{classes};
        values.zero();
        values(0) = 1;
        values(static_cast<tensor_size_t>(value)) = 1;
        UTEST_REQUIRE_NOTHROW(check_set(sample, feature, values));

        // cannot set if the sample index is invalid
        for (const auto invalid_sample : {-1, 10001})
        {
            UTEST_REQUIRE_THROW(check_set(invalid_sample, feature, values), std::runtime_error);
        }
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
}

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

UTEST_CASE(fs_default)
{
    const auto storage = feature_storage_t{};
    UTEST_CHECK_EQUAL(storage.samples(), 0);
}

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
}

UTEST_END_MODULE()
