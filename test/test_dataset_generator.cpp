#include <utest/utest.h>
#include <nano/dataset/generators.h>

using namespace nano;

static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

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

std::ostream& operator<<(std::ostream& stream, const mclass_stats_t::class_counts_t& class_counts)
{
    for (const auto [class_hits, counts] : class_counts)
    {
        stream << "class_hits=" << class_hits << ", counts=" << counts << std::endl;
    }
    return stream;
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
            set(sample, 1, (sample % 3 == 0) ? 0 : 1);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 2, sample);
        }

        tensor_mem_t<tensor_size_t, 3> values(2, 1, 2);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            values.full(sample);
            values(0) = sample + 1;
            set(sample, 3, values);
        }
    }

    static auto expected_select0()
    {
        return make_tensor<int8_t>(make_dims(10, 3),
            0, 1, 1, -1, -1, -1, -1, -1, -1,
            1, 0, 0, -1, -1, -1, -1, -1, -1,
            0, 1, 1, -1, -1, -1, -1, -1, -1,
            1, 0, 0);
    }
    static auto expected_select1()
    {
        return make_tensor<int32_t>(make_dims(10),
            0, 1, 1, 0, 1, 1, 0, 1, 1, 0);
    }
    static auto expected_select2()
    {
        return make_tensor<scalar_t>(make_dims(10),
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    }
    static auto expected_select3()
    {
        return make_tensor<scalar_t>(make_dims(10, 2, 1, 2),
            1.0, 0.0, 0.0, 0.0, NaN, NaN, NaN, NaN,
            3.0, 2.0, 2.0, 2.0, NaN, NaN, NaN, NaN,
            5.0, 4.0, 4.0, 4.0, NaN, NaN, NaN, NaN,
            7.0, 6.0, 6.0, 6.0, NaN, NaN, NaN, NaN,
            9.0, 8.0, 8.0, 8.0, NaN, NaN, NaN, NaN);
    }

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

static auto make_sample_ranges(const dataset_generator_t& generator)
{
    const auto samples = generator.samples().size();

    return std::vector<tensor_range_t>
    {
        make_range(0, samples),
        make_range(0, samples / 2),
        make_range(samples / 2, samples)
    };
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
static void check_select0(const dataset_generator_t& generator,
    tensor_size_t feature, tensor_range_t sample_range, const tensor_t<tstorage, tscalar, trank>& expected)
{
    tensor_t<tstorage, tscalar, trank> buffer;
    decltype(generator.select(feature, sample_range, buffer)) storage;

    UTEST_CHECK_NOTHROW(storage = generator.select(feature, sample_range, buffer));
    UTEST_CHECK_TENSOR_CLOSE(storage, expected.slice(sample_range), 1e-12);

    const auto shuffle = generator.shuffle(feature);
    const auto samples = arange(0, generator.samples().size());
    UTEST_REQUIRE_EQUAL(shuffle.size(), samples.size());
    UTEST_CHECK(std::is_permutation(shuffle.begin(), shuffle.end(), samples.begin()));
    UTEST_CHECK_NOT_EQUAL(shuffle, samples);
    UTEST_CHECK_NOTHROW(storage = generator.select(feature, sample_range, buffer));
    UTEST_CHECK_TENSOR_CLOSE(storage, expected.indexed(shuffle.slice(sample_range)), 1e-12);

    generator.unshuffle();
    UTEST_CHECK_NOTHROW(storage = generator.select(feature, sample_range, buffer));
    UTEST_CHECK_TENSOR_CLOSE(storage, expected.slice(sample_range), 1e-12);

    generator.drop(feature);
    tensor_t<tstorage, tscalar, trank> expected_dropped = expected.slice(sample_range);
    UTEST_CHECK_NOTHROW(storage = generator.select(feature, sample_range, buffer));
    switch (generator.feature(feature).type())
    {
    case feature_type::sclass:  expected_dropped.full(-1); break;
    case feature_type::mclass:  expected_dropped.full(-1); break;
    default:                    expected_dropped.full(static_cast<tscalar>(NaN)); break;
    }
    UTEST_CHECK_TENSOR_CLOSE(storage, expected_dropped, 1e-12);

    generator.undrop();
    UTEST_CHECK_NOTHROW(storage = generator.select(feature, sample_range, buffer));
    UTEST_CHECK_TENSOR_CLOSE(storage, expected.slice(sample_range), 1e-12);
}

static void check_select(const dataset_generator_t& generator, tensor_size_t feature, const sclass_mem_t& expected)
{
    mclass_mem_t mclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    for (const auto sample_range : make_sample_ranges(generator))
    {
        UTEST_CHECK_THROW(generator.select(feature, sample_range, mclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(feature, sample_range, scalar_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(feature, sample_range, struct_buffer), std::runtime_error);
        check_select0(generator, feature, sample_range, expected);
    }
}

static void check_select(const dataset_generator_t& generator, tensor_size_t feature, const mclass_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    for (const auto sample_range : make_sample_ranges(generator))
    {
        UTEST_CHECK_THROW(generator.select(feature, sample_range, sclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(feature, sample_range, scalar_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(feature, sample_range, struct_buffer), std::runtime_error);
        check_select0(generator, feature, sample_range, expected);
    }
}

static void check_select(const dataset_generator_t& generator, tensor_size_t feature, const scalar_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    mclass_mem_t mclass_buffer;
    struct_mem_t struct_buffer;

    for (const auto sample_range : make_sample_ranges(generator))
    {
        UTEST_CHECK_THROW(generator.select(feature, sample_range, sclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(feature, sample_range, mclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(feature, sample_range, struct_buffer), std::runtime_error);
        check_select0(generator, feature, sample_range, expected);
    }
}

static void check_select(const dataset_generator_t& generator, tensor_size_t feature, const struct_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    mclass_mem_t mclass_buffer;
    scalar_mem_t scalar_buffer;

    for (const auto sample_range : make_sample_ranges(generator))
    {
        UTEST_CHECK_THROW(generator.select(feature, sample_range, sclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(feature, sample_range, mclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(feature, sample_range, scalar_buffer), std::runtime_error);
        check_select0(generator, feature, sample_range, expected);
    }
}

static void check_flatten(const dataset_generator_t& generator,
    const tensor2d_t& expected_flatten, const indices_t& expected_column2features, scalar_t eps = 1e-12)
{
    tensor2d_t flatten_buffer;
    tensor2d_cmap_t flatten_cmap;

    for (const auto sample_range : make_sample_ranges(generator))
    {
        UTEST_REQUIRE_EQUAL(generator.columns(), expected_flatten.size<1>());
        UTEST_CHECK_NOTHROW(flatten_cmap = generator.flatten(sample_range, flatten_buffer));
        UTEST_CHECK_TENSOR_CLOSE(flatten_cmap, expected_flatten.slice(sample_range), eps);
    }

    UTEST_REQUIRE_EQUAL(generator.columns(), expected_column2features.size());
    for (tensor_size_t column = 0; column < generator.columns(); ++ column)
    {
        UTEST_CHECK_EQUAL(generator.column2feature(column), expected_column2features(column));
    }
}

static void check_select_stats(const dataset_generator_t& generator,
    const indices_t& expected_sclass_features,
    const indices_t& expected_mclass_features,
    const indices_t& expected_scalar_features,
    const indices_t& expected_struct_features)
{
    for (auto ex : {execution::par, execution::seq})
    {
        select_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = generator.select_stats(ex));
        UTEST_CHECK_TENSOR_EQUAL(stats.m_sclass_features, expected_sclass_features);
        UTEST_CHECK_TENSOR_EQUAL(stats.m_mclass_features, expected_mclass_features);
        UTEST_CHECK_TENSOR_EQUAL(stats.m_scalar_features, expected_scalar_features);
        UTEST_CHECK_TENSOR_EQUAL(stats.m_struct_features, expected_struct_features);
    }
}

static void check_flatten_stats0(const dataset_generator_t& generator,
    tensor_size_t expected_count,
    const tensor1d_t& expected_min, const tensor1d_t& expected_max,
    const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev, scalar_t eps = 1e-12)
{
    for (auto ex : {execution::par, execution::seq})
    {
        flatten_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = generator.flatten_stats(ex, 3));
        UTEST_CHECK_EQUAL(stats.m_count, expected_count);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_min, expected_min, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_max, expected_max, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_mean, expected_mean, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.m_stdev, expected_stdev, eps);
    }
}

static void check_flatten_stats(const dataset_generator_t& generator,
    tensor_size_t expected_count,
    const tensor1d_t& expected_min, const tensor1d_t& expected_max,
    const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev)
{
    check_flatten_stats0(generator, expected_count, expected_min, expected_max, expected_mean, expected_stdev);

    generator.shuffle(1);
    check_flatten_stats0(generator, expected_count, expected_min, expected_max, expected_mean, expected_stdev);

    generator.shuffle(0);
    check_flatten_stats0(generator, expected_count, expected_min, expected_max, expected_mean, expected_stdev);

    generator.unshuffle();
    check_flatten_stats0(generator, expected_count, expected_min, expected_max, expected_mean, expected_stdev);
}

static void check_targets(const dataset_generator_t& generator,
    const feature_t& expected_target, tensor3d_dims_t expected_target_dims,
    const tensor4d_t& expected_targets, scalar_t eps = 1e-12)
{
    const auto alrange = make_range(0, expected_targets.size<0>());

    tensor4d_t targets_buffer;
    tensor4d_cmap_t targets_cmap;
    UTEST_CHECK_EQUAL(generator.target(), expected_target);
    UTEST_CHECK_EQUAL(generator.target_dims(), expected_target_dims);
    UTEST_REQUIRE_NOTHROW(targets_cmap = generator.targets(alrange, targets_buffer));
    UTEST_CHECK_TENSOR_CLOSE(targets_cmap, expected_targets, eps);
}

static void check_targets_stats(const dataset_generator_t& generator,
    const indices_t& expected_class_counts,
    const tensor1d_t& expected_sample_weights, scalar_t eps = 1e-12)
{
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_REQUIRE_NOTHROW(stats = generator.targets_stats(ex, 3));
        UTEST_REQUIRE_NOTHROW(std::get<sclass_stats_t>(stats));
        UTEST_CHECK_TENSOR_EQUAL(std::get<sclass_stats_t>(stats).m_class_counts, expected_class_counts);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(stats), expected_sample_weights, eps);

        std::get<sclass_stats_t>(stats).m_class_counts(0) = 0;
        UTEST_CHECK_NOTHROW(generator.sample_weights(stats));

        std::get<sclass_stats_t>(stats).m_class_counts.resize(42);
        std::get<sclass_stats_t>(stats).m_class_counts.zero();
        UTEST_CHECK_THROW(generator.sample_weights(stats), std::runtime_error);
    }
}

static void check_targets_stats(const dataset_generator_t& generator,
    const mclass_stats_t::class_counts_t& expected_class_counts,
    const tensor1d_t& expected_sample_weights, scalar_t eps = 1e-12)
{
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_REQUIRE_NOTHROW(stats = generator.targets_stats(ex, 3));
        UTEST_REQUIRE_NOTHROW(std::get<mclass_stats_t>(stats));
        UTEST_CHECK_EQUAL(std::get<mclass_stats_t>(stats).m_class_counts, expected_class_counts);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(stats), expected_sample_weights, eps);

        /*std::get<mclass_stats_t>(stats).m_class_counts.begin()->second = 0;
        UTEST_CHECK_NOTHROW(generator.sample_weights(stats));

        std::get<mclass_stats_t>(stats).m_class_counts[string_t(42U, '0')] = 0;
        UTEST_CHECK_THROW(generator.sample_weights(stats), std::runtime_error);*/
    }
}

static void check_targets_stats(const dataset_generator_t& generator,
    tensor_size_t expected_count,
    const tensor1d_t& expected_min, const tensor1d_t& expected_max,
    const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev, scalar_t eps = 1e-12)
{
    tensor1d_t expected_sample_weights = tensor1d_t{generator.samples().size()};
    expected_sample_weights.full(1.0);

    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_REQUIRE_NOTHROW(stats = generator.targets_stats(ex, 3));
        UTEST_REQUIRE_NOTHROW(std::get<scalar_stats_t>(stats));
        UTEST_CHECK_EQUAL(std::get<scalar_stats_t>(stats).m_count, expected_count);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).m_min, expected_min, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).m_max, expected_max, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).m_mean, expected_mean, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).m_stdev, expected_stdev, eps);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(stats), expected_sample_weights, eps);
    }
}

template <typename... tindices>
static auto make_indices(tindices... indices)
{
    return make_tensor<tensor_size_t>(make_dims(static_cast<tensor_size_t>(sizeof...(indices))), indices...);
}

// TODO: check caching
// TODO: check that feature scaling scaling works
// TODO: check that feature extraction works (e.g sign(x), sign(x)*log(1+x^2), polynomial expansion)

UTEST_BEGIN_MODULE(test_dataset_generator)

UTEST_CASE(unsupervised)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), string_t::npos);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_CHECK_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));

    check_select(generator, 0, dataset.expected_select0());
    check_select(generator, 1, dataset.expected_select1());
    check_select(generator, 2, dataset.expected_select2());
    check_select(generator, 3, dataset.expected_select3());
    check_select_stats(generator, make_indices(1), make_indices(0), make_indices(2), make_indices(3));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 10),
        -1, +1, +1, +1, -1, 0, 1, 0, 0, 0,
        +0, +0, +0, -1, +1, 1, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 2, 3, 2, 2, 2,
        +1, -1, -1, +1, -1, 3, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 4, 5, 4, 4, 4,
        +0, +0, +0, -1, +1, 5, 0, 0, 0, 0,
        -1, +1, +1, +1, -1, 6, 7, 6, 6, 6,
        +0, +0, +0, -1, +1, 7, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 8, 9, 8, 8, 8,
        +1, -1, -1, +1, -1, 9, 0, 0, 0, 0),
        make_indices(0, 0, 0, 1, 1,  2, 3, 3, 3, 3));

    generator.drop(0);
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 10),
        +0, +0, +0, +1, -1, 0, 1, 0, 0, 0,
        +0, +0, +0, -1, +1, 1, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 2, 3, 2, 2, 2,
        +0, +0, +0, +1, -1, 3, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 4, 5, 4, 4, 4,
        +0, +0, +0, -1, +1, 5, 0, 0, 0, 0,
        +0, +0, +0, +1, -1, 6, 7, 6, 6, 6,
        +0, +0, +0, -1, +1, 7, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 8, 9, 8, 8, 8,
        +0, +0, +0, +1, -1, 9, 0, 0, 0, 0),
        make_indices(0, 0, 0, 1, 1,  2, 3, 3, 3, 3));

    generator.drop(2);
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 10),
        +0, +0, +0, +1, -1, 0, 1, 0, 0, 0,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 0, 3, 2, 2, 2,
        +0, +0, +0, +1, -1, 0, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 0, 5, 4, 4, 4,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, 0,
        +0, +0, +0, +1, -1, 0, 7, 6, 6, 6,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 0, 9, 8, 8, 8,
        +0, +0, +0, +1, -1, 0, 0, 0, 0, 0),
        make_indices(0, 0, 0, 1, 1,  2, 3, 3, 3, 3));

    generator.undrop();
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 10),
        -1, +1, +1, +1, -1, 0, 1, 0, 0, 0,
        +0, +0, +0, -1, +1, 1, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 2, 3, 2, 2, 2,
        +1, -1, -1, +1, -1, 3, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 4, 5, 4, 4, 4,
        +0, +0, +0, -1, +1, 5, 0, 0, 0, 0,
        -1, +1, +1, +1, -1, 6, 7, 6, 6, 6,
        +0, +0, +0, -1, +1, 7, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 8, 9, 8, 8, 8,
        +1, -1, -1, +1, -1, 9, 0, 0, 0, 0),
        make_indices(0, 0, 0, 1, 1,  2, 3, 3, 3, 3));

    check_flatten_stats(
        generator, 10,
        make_tensor<scalar_t>(make_dims(10), -1, -1, -1, -1, -1, 0, 0, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(10), +1, +1, +1, +1, +1, 9, 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(10), 0, 0, 0, -0.2, +0.2, 4.5, 2.5, 2, 2, 2),
        make_tensor<scalar_t>(make_dims(10),
            0.666666666667, 0.666666666667, 0.666666666667, 1.032795558989, 1.032795558989,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000));

    {
        tensor4d_t targets_buffer;
        tensor4d_cmap_t targets_cmap;
        const auto alrange = make_range(0, samples.size());
        UTEST_CHECK_EQUAL(generator.target(), feature_t{});
        UTEST_CHECK_EQUAL(generator.target_dims(), make_dims(0, 0, 0));
        UTEST_CHECK_THROW(targets_cmap = generator.targets(alrange, targets_buffer), std::runtime_error);
    }
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = generator.targets_stats(ex, 3));
        UTEST_CHECK_EQUAL(std::holds_alternative<scalar_stats_t>(stats), false);
        UTEST_CHECK_EQUAL(std::holds_alternative<sclass_stats_t>(stats), false);
        UTEST_CHECK_EQUAL(std::holds_alternative<mclass_stats_t>(stats), false);
    }
}

UTEST_CASE(sclassification)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), 1U);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_CHECK_EQUAL(generator.features(), 3);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));

    check_select(generator, 0, dataset.expected_select0());
    check_select(generator, 1, dataset.expected_select2());
    check_select(generator, 2, dataset.expected_select3());
    check_select_stats(generator, indices_t{}, make_indices(0), make_indices(1), make_indices(2));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 8),
        -1, +1, +1, 0, 1, 0, 0, 0,
        +0, +0, +0, 1, 0, 0, 0, 0,
        +0, +0, +0, 2, 3, 2, 2, 2,
        +1, -1, -1, 3, 0, 0, 0, 0,
        +0, +0, +0, 4, 5, 4, 4, 4,
        +0, +0, +0, 5, 0, 0, 0, 0,
        -1, +1, +1, 6, 7, 6, 6, 6,
        +0, +0, +0, 7, 0, 0, 0, 0,
        +0, +0, +0, 8, 9, 8, 8, 8,
        +1, -1, -1, 9, 0, 0, 0, 0),
        make_indices(0, 0, 0, 1, 2, 2, 2, 2));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(8), -1, -1, -1, 0, 0, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(8), +1, +1, +1, 9, 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(8), 0.0, 0.0, 0.0, 4.5, 2.5, 2.0, 2.0, 2.0),
        make_tensor<scalar_t>(make_dims(8),
            0.666666666667, 0.666666666667, 0.666666666667,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000));

    check_targets(generator, feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}), make_dims(2, 1, 1),
        make_tensor<scalar_t>(make_dims(10, 2, 1, 1),
            +1, -1, -1, +1, -1, +1, +1, -1, -1, +1,
            -1, +1, +1, -1, -1, +1, -1, +1, +1, -1));
    check_targets_stats(generator, make_indices(4, 6), make_tensor<scalar_t>(make_dims(10),
        5.0 / 4.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 4.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 4.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 4.0));
}

UTEST_CASE(mclassification)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), 0U);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_CHECK_EQUAL(generator.features(), 3);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));

    check_select(generator, 0, dataset.expected_select1());
    check_select(generator, 1, dataset.expected_select2());
    check_select(generator, 2, dataset.expected_select3());
    check_select_stats(generator, make_indices(0), indices_t{}, make_indices(1), make_indices(2));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 7),
        +1, -1, 0, 1, 0, 0, 0,
        -1, +1, 1, 0, 0, 0, 0,
        -1, +1, 2, 3, 2, 2, 2,
        +1, -1, 3, 0, 0, 0, 0,
        -1, +1, 4, 5, 4, 4, 4,
        -1, +1, 5, 0, 0, 0, 0,
        +1, -1, 6, 7, 6, 6, 6,
        -1, +1, 7, 0, 0, 0, 0,
        -1, +1, 8, 9, 8, 8, 8,
        +1, -1, 9, 0, 0, 0, 0),
        make_indices(0, 0, 1, 2, 2, 2, 2));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(7), -1, -1, 0, 0, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(7), +1, +1, 9, 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(7), -0.2, +0.2, 4.5, 2.5, 2.0, 2.0, 2.0),
        make_tensor<scalar_t>(make_dims(7),
            1.032795558989, 1.032795558989,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000));

    check_targets(generator, feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}), make_dims(3, 1, 1),
        make_tensor<scalar_t>(make_dims(10, 3, 1, 1),
            -1.0, +1.0, +1.0, NaN, NaN, NaN, NaN, NaN, NaN,
            +1.0, -1.0, -1.0, NaN, NaN, NaN, NaN, NaN, NaN,
            -1.0, +1.0, +1.0, NaN, NaN, NaN, NaN, NaN, NaN,
            +1.0, -1.0, -1.0));
    check_targets_stats(generator,
        {   {"011", 2},
            {"100", 2}},
        make_tensor<scalar_t>(make_dims(10), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
}

UTEST_CASE(regression)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), 2U);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_CHECK_EQUAL(generator.features(), 3);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));

    check_select(generator, 0, dataset.expected_select0());
    check_select(generator, 1, dataset.expected_select1());
    check_select(generator, 2, dataset.expected_select3());
    check_select_stats(generator, make_indices(1), make_indices(0), indices_t{}, make_indices(2));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 9),
        -1, +1, +1, +1, -1, 1, 0, 0, 0,
        +0, +0, +0, -1, +1, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 3, 2, 2, 2,
        +1, -1, -1, +1, -1, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 5, 4, 4, 4,
        +0, +0, +0, -1, +1, 0, 0, 0, 0,
        -1, +1, +1, +1, -1, 7, 6, 6, 6,
        +0, +0, +0, -1, +1, 0, 0, 0, 0,
        +0, +0, +0, -1, +1, 9, 8, 8, 8,
        +1, -1, -1, +1, -1, 0, 0, 0, 0),
        make_indices(0, 0, 0, 1, 1, 2, 2, 2, 2));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(9), -1, -1, -1, -1, -1, 0, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(9), +1, +1, +1, +1, +1, 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(9), 0, 0, 0, -0.2, 0.2, 2.5, 2, 2, 2),
        make_tensor<scalar_t>(make_dims(9),
            0.666666666667, 0.666666666667, 0.666666666667, 1.032795558989, 1.032795558989,
            3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000));

    check_targets(generator, feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)), make_dims(1, 1, 1),
        make_tensor<scalar_t>(make_dims(10, 1, 1, 1), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    check_targets_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(1), 0),
        make_tensor<scalar_t>(make_dims(1), 9),
        make_tensor<scalar_t>(make_dims(1), 4.5),
        make_tensor<scalar_t>(make_dims(1), 3.027650354097));
}

UTEST_CASE(mvregression)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), 3U);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_CHECK_EQUAL(generator.features(), 3);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));

    check_select(generator, 0, dataset.expected_select0());
    check_select(generator, 1, dataset.expected_select1());
    check_select(generator, 2, dataset.expected_select2());
    check_select_stats(generator, make_indices(1), make_indices(0), make_indices(2), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 6),
        -1, +1, +1, +1, -1, 0,
        +0, +0, +0, -1, +1, 1,
        +0, +0, +0, -1, +1, 2,
        +1, -1, -1, +1, -1, 3,
        +0, +0, +0, -1, +1, 4,
        +0, +0, +0, -1, +1, 5,
        -1, +1, +1, +1, -1, 6,
        +0, +0, +0, -1, +1, 7,
        +0, +0, +0, -1, +1, 8,
        +1, -1, -1, +1, -1, 9),
        make_indices(0, 0, 0, 1, 1, 2));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(6), -1, -1, -1, -1, -1, 0),
        make_tensor<scalar_t>(make_dims(6), +1, +1, +1, +1, +1, 9),
        make_tensor<scalar_t>(make_dims(6), 0, 0, 0, -0.2, +0.2, 4.5),
        make_tensor<scalar_t>(make_dims(6),
            0.666666666667, 0.666666666667, 0.666666666667, 1.032795558989, 1.032795558989, 3.027650354097));

    check_targets(generator, feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)), make_dims(2, 1, 2),
        make_tensor<scalar_t>(make_dims(10, 2, 1, 2),
            1, 0, 0, 0, NaN, NaN, NaN, NaN,
            3, 2, 2, 2, NaN, NaN, NaN, NaN,
            5, 4, 4, 4, NaN, NaN, NaN, NaN,
            7, 6, 6, 6, NaN, NaN, NaN, NaN,
            9, 8, 8, 8, NaN, NaN, NaN, NaN));
    check_targets_stats(generator, 5,
        make_tensor<scalar_t>(make_dims(4), 1, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(4), 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(4), 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(4), 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168));
}

UTEST_END_MODULE()
