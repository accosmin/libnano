#include <utest/utest.h>
#include <nano/dataset/generator.h>

using namespace nano;

static auto make_features()
{
    return features_t
    {
        feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}),
        feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)),
    };
}

class fixture_dataset_t final : public memory_dataset_t
{
public:

    fixture_dataset_t(tensor_size_t samples, size_t target) :
        m_samples(samples),
        m_features(make_features()),
        m_target(target)
    {
    }

    void load() override
    {
        resize(m_samples, m_features, m_target);

        tensor_mem_t<tensor_size_t, 1> hits(3);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 3)
        {
            hits(0) = sample % 2;
            hits(1) = 1 - (sample % 2);
            hits(2) = (sample % 6) == 0;
            set(sample, 0, hits);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 1, sample % 2);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 2, sample);
        }

        tensor_mem_t<tensor_size_t, 3> values(2, 1, 2);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            values.constant(sample);
            values(0) = sample + 1;
            set(sample, 3, values);
        }
    }

    static constexpr auto nan = std::numeric_limits<scalar_t>::quiet_NaN();

    static auto expected_select0() { return make_tensor<int32_t>(make_dims(10), 0, -1, -1, 1, -1, -1, 0, -1, -1, 1); }
    static auto expected_select1() { return make_tensor<int32_t>(make_dims(10), 1, -1, -1, 0, -1, -1, 1, -1, -1, 0); }
    static auto expected_select2() { return make_tensor<int32_t>(make_dims(10), 1, -1, -1, 0, -1, -1, 1, -1, -1, 0); }
    static auto expected_select3() { return make_tensor<int32_t>(make_dims(10), 0, 1, 0, 1, 0, 1, 0, 1, 0, 1); }
    static auto expected_select4() { return make_tensor<scalar_t>(make_dims(10), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9); }
    static auto expected_select5() { return make_tensor<scalar_t>(make_dims(10, 2, 1, 2),
        1.0, 0.0, 0.0, 0.0, nan, nan, nan, nan,
        3.0, 2.0, 2.0, 2.0, nan, nan, nan, nan,
        5.0, 4.0, 4.0, 4.0, nan, nan, nan, nan,
        7.0, 6.0, 6.0, 6.0, nan, nan, nan, nan,
        9.0, 8.0, 8.0, 8.0, nan, nan, nan, nan); }
    static auto expected_select6() { return make_tensor<scalar_t>(make_dims(10), 1.0, nan, 3.0, nan, 5.0, nan, 7.0, nan, 9.0, nan); }
    static auto expected_select7() { return make_tensor<scalar_t>(make_dims(10), 0.0, nan, 2.0, nan, 4.0, nan, 6.0, nan, 8.0, nan); }
    static auto expected_select8() { return make_tensor<scalar_t>(make_dims(10), 0.0, nan, 2.0, nan, 4.0, nan, 6.0, nan, 8.0, nan); }
    static auto expected_select9() { return make_tensor<scalar_t>(make_dims(10), 0.0, nan, 2.0, nan, 4.0, nan, 6.0, nan, 8.0, nan); }

private:

    tensor_size_t   m_samples{0};
    features_t      m_features;
    size_t          m_target;
};

static auto make_dataset(tensor_size_t samples, size_t target)
{
    auto dataset = fixture_dataset_t{samples, target};
    UTEST_CHECK_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    return dataset;
}

static void check_sclass(const dataset_generator_t& generator, tensor_size_t feature, indices_cmap_t samples, const sclass_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    UTEST_CHECK_NOTHROW(generator.select(feature, samples, sclass_buffer));
    UTEST_CHECK_THROW(generator.select(feature, samples, scalar_buffer), std::runtime_error);
    UTEST_CHECK_THROW(generator.select(feature, samples, struct_buffer), std::runtime_error);
    UTEST_CHECK_TENSOR_EQUAL(sclass_buffer, expected);
}

static void check_scalar(const dataset_generator_t& generator, tensor_size_t feature, indices_cmap_t samples, const scalar_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    UTEST_CHECK_THROW(generator.select(feature, samples, sclass_buffer), std::runtime_error);
    UTEST_CHECK_NOTHROW(generator.select(feature, samples, scalar_buffer));
    UTEST_CHECK_THROW(generator.select(feature, samples, struct_buffer), std::runtime_error);
    UTEST_CHECK_TENSOR_CLOSE(scalar_buffer, expected, 1e-12);
}

static void check_struct(const dataset_generator_t& generator, tensor_size_t feature, indices_cmap_t samples, const struct_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    UTEST_CHECK_THROW(generator.select(feature, samples, sclass_buffer), std::runtime_error);
    UTEST_CHECK_THROW(generator.select(feature, samples, scalar_buffer), std::runtime_error);
    UTEST_CHECK_NOTHROW(generator.select(feature, samples, struct_buffer));
    UTEST_CHECK_TENSOR_CLOSE(struct_buffer, expected, 1e-12);
}

template <typename... tindices>
static auto make_indices(tindices... indices)
{
    return make_tensor<tensor_size_t>(make_dims(static_cast<tensor_size_t>(sizeof...(indices))), indices...);
}

// TODO: check that the flatten & the feature iterators work as expected
// TODO: check that feature scaling scaling works
// TODO: check that feature extraction works (e.g sign(x), sign(x)*log(1+x^2), polynomial expansion)

UTEST_BEGIN_MODULE(test_dataset_generator)

UTEST_CASE(unsupervised)
{
    const auto samples = ::nano::arange(0, 10);
    const auto alrange = make_range(0, samples.size());
    const auto dataset = make_dataset(samples.size(), string_t::npos);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_CHECK_EQUAL(generator.features(), 10);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3_m0"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"mclass3_m1"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"mclass3_m2"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));
    UTEST_CHECK_EQUAL(generator.feature(6), feature_t{"u8_struct_0"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(7), feature_t{"u8_struct_1"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(8), feature_t{"u8_struct_2"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(9), feature_t{"u8_struct_3"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));

    {
        const auto features = make_indices(0, 5);
        const auto expected = make_indices(0, 3);
        UTEST_CHECK_TENSOR_EQUAL(generator.original(features), expected);
    }
    {
        const auto features = make_indices(0, 2, 3, 4, 5);
        const auto expected = make_indices(0, 1, 2, 3);
        UTEST_CHECK_TENSOR_EQUAL(generator.original(features), expected);
    }
    for (auto ex : {execution::par, execution::seq})
    {
        select_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = make_select_stats(generator, ex));
        UTEST_CHECK_TENSOR_EQUAL(stats.m_sclass_features, make_indices(0, 1, 2, 3));
        UTEST_CHECK_TENSOR_EQUAL(stats.m_scalar_features, make_indices(4, 6, 7, 8, 9));
        UTEST_CHECK_TENSOR_EQUAL(stats.m_struct_features, make_indices(5));
    }

    check_sclass(generator, 0, samples, dataset.expected_select0());
    check_sclass(generator, 1, samples, dataset.expected_select1());
    check_sclass(generator, 2, samples, dataset.expected_select2());
    check_sclass(generator, 3, samples, dataset.expected_select3());
    check_scalar(generator, 4, samples, dataset.expected_select4());
    check_struct(generator, 5, samples, dataset.expected_select5());
    check_scalar(generator, 6, samples, dataset.expected_select6());
    check_scalar(generator, 7, samples, dataset.expected_select7());
    check_scalar(generator, 8, samples, dataset.expected_select8());
    check_scalar(generator, 9, samples, dataset.expected_select9());

    {
        tensor2d_t flatten_buffer;
        tensor2d_cmap_t flatten_cmap;
        UTEST_CHECK_EQUAL(generator.columns(), 10);
        UTEST_CHECK_NOTHROW(flatten_cmap = generator.flatten(alrange, flatten_buffer));

        const auto expected = make_tensor<scalar_t>(make_dims(10, 10),
            -1, +1, +1, +1, -1, 0, 1, 0, 0, 0,
            +0, +0, +0, -1, +1, 1, 0, 0, 0, 0,
            +0, +0, +0, +1, -1, 2, 3, 2, 2, 2,
            +1, -1, -1, -1, +1, 3, 0, 0, 0, 0,
            +0, +0, +0, +1, -1, 4, 5, 4, 4, 4,
            +0, +0, +0, -1, +1, 5, 0, 0, 0, 0,
            -1, +1, +1, +1, -1, 6, 7, 6, 6, 6,
            +0, +0, +0, -1, +1, 7, 0, 0, 0, 0,
            +0, +0, +0, +1, -1, 8, 9, 8, 8, 8,
            +1, -1, -1, -1, +1, 9, 0, 0, 0, 0);
        UTEST_CHECK_TENSOR_CLOSE(flatten_cmap, expected, 1e-12);
    }
    for (auto ex : {execution::par, execution::seq})
    {
        const auto expected_min = make_tensor<scalar_t>(make_dims(10), -1, -1, -1, -1, -1, 0, 0, 0, 0, 0);
        const auto expected_max = make_tensor<scalar_t>(make_dims(10), +1, +1, +1, +1, +1, 9, 9, 8, 8, 8);
        const auto expected_mean = make_tensor<scalar_t>(make_dims(10), 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 2.5, 2.0, 2.0, 2.0);
        const auto expected_stdev = make_tensor<scalar_t>(make_dims(10),
            0.666666666667, 0.666666666667, 0.666666666667, 1.054092553389, 1.054092553389,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000);

        flatten_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = make_flatten_stats(generator, ex, 3));
        UTEST_CHECK_EQUAL(stats.m_count, 10);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_min, expected_min, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_max, expected_max, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_mean, expected_mean, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_stdev, expected_stdev, 1e-12);
    }

    {
        tensor4d_t targets_buffer;
        tensor4d_cmap_t targets_cmap;
        UTEST_CHECK_EQUAL(generator.target(), feature_t{});
        UTEST_CHECK_EQUAL(generator.target_dims(), make_dims(0, 0, 0));
        UTEST_CHECK_THROW(targets_cmap = generator.targets(alrange, targets_buffer), std::runtime_error);
    }
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_CHECK_THROW(stats = make_targets_stats(generator, ex, 3), std::runtime_error);
    }

    // TODO: check caching
}

UTEST_CASE(classification)
{
    const auto samples = ::nano::arange(0, 10);
    const auto alrange = make_range(0, samples.size());
    const auto dataset = make_dataset(samples.size(), 1U);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_CHECK_EQUAL(generator.features(), 9);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3_m0"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"mclass3_m1"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"mclass3_m2"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"u8_struct_0"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(6), feature_t{"u8_struct_1"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(7), feature_t{"u8_struct_2"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(8), feature_t{"u8_struct_3"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));

    {
        const auto features = make_indices(0, 5);
        const auto expected = make_indices(0, 2);
        UTEST_CHECK_TENSOR_EQUAL(generator.original(features), expected);
    }
    {
        const auto features = make_indices(0, 2, 3, 4, 5);
        const auto expected = make_indices(0, 1, 2);
        UTEST_CHECK_TENSOR_EQUAL(generator.original(features), expected);
    }
    for (auto ex : {execution::par, execution::seq})
    {
        select_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = make_select_stats(generator, ex));
        UTEST_CHECK_TENSOR_EQUAL(stats.m_sclass_features, make_indices(0, 1, 2));
        UTEST_CHECK_TENSOR_EQUAL(stats.m_scalar_features, make_indices(3, 5, 6, 7, 8));
        UTEST_CHECK_TENSOR_EQUAL(stats.m_struct_features, make_indices(4));
    }

    check_sclass(generator, 0, samples, dataset.expected_select0());
    check_sclass(generator, 1, samples, dataset.expected_select1());
    check_sclass(generator, 2, samples, dataset.expected_select2());
    check_scalar(generator, 3, samples, dataset.expected_select4());
    check_struct(generator, 4, samples, dataset.expected_select5());
    check_scalar(generator, 5, samples, dataset.expected_select6());
    check_scalar(generator, 6, samples, dataset.expected_select7());
    check_scalar(generator, 7, samples, dataset.expected_select8());
    check_scalar(generator, 8, samples, dataset.expected_select9());

    {
        tensor2d_t flatten_buffer;
        tensor2d_cmap_t flatten_cmap;
        UTEST_CHECK_EQUAL(generator.columns(), 8);
        UTEST_CHECK_NOTHROW(flatten_cmap = generator.flatten(alrange, flatten_buffer));

        const auto expected = make_tensor<scalar_t>(make_dims(10, 8),
            -1, +1, +1, 0, 1, 0, 0, 0,
            +0, +0, +0, 1, 0, 0, 0, 0,
            +0, +0, +0, 2, 3, 2, 2, 2,
            +1, -1, -1, 3, 0, 0, 0, 0,
            +0, +0, +0, 4, 5, 4, 4, 4,
            +0, +0, +0, 5, 0, 0, 0, 0,
            -1, +1, +1, 6, 7, 6, 6, 6,
            +0, +0, +0, 7, 0, 0, 0, 0,
            +0, +0, +0, 8, 9, 8, 8, 8,
            +1, -1, -1, 9, 0, 0, 0, 0);
        UTEST_CHECK_TENSOR_CLOSE(flatten_cmap, expected, 1e-12);
    }
    for (auto ex : {execution::par, execution::seq})
    {
        const auto expected_min = make_tensor<scalar_t>(make_dims(8), -1, -1, -1, 0, 0, 0, 0, 0);
        const auto expected_max = make_tensor<scalar_t>(make_dims(8), +1, +1, +1, 9, 9, 8, 8, 8);
        const auto expected_mean = make_tensor<scalar_t>(make_dims(8), 0.0, 0.0, 0.0, 4.5, 2.5, 2.0, 2.0, 2.0);
        const auto expected_stdev = make_tensor<scalar_t>(make_dims(8),
            0.666666666667, 0.666666666667, 0.666666666667,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000);

        flatten_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = make_flatten_stats(generator, ex, 3));
        UTEST_CHECK_EQUAL(stats.m_count, 10);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_min, expected_min, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_max, expected_max, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_mean, expected_mean, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_stdev, expected_stdev, 1e-12);
    }

    {
        tensor4d_t targets_buffer;
        tensor4d_cmap_t targets_cmap;
        UTEST_CHECK_EQUAL(generator.target(), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
        UTEST_CHECK_EQUAL(generator.target_dims(), make_dims(2, 1, 1));
        UTEST_CHECK_NOTHROW(targets_cmap = generator.targets(alrange, targets_buffer));

        const auto expected = make_tensor<scalar_t>(make_dims(10, 2, 1, 1),
            +1, -1,
            -1, +1,
            +1, -1,
            -1, +1,
            +1, -1,
            -1, +1,
            +1, -1,
            -1, +1,
            +1, -1,
            -1, +1);
        UTEST_CHECK_TENSOR_CLOSE(targets_cmap, expected, 1e-12);
    }

    // TODO: check select, flatten and target stats
    // TODO: check caching
}

UTEST_END_MODULE()
