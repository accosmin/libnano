#include <utest/utest.h>
#include <nano/dataset/generators.h>

using namespace nano;

static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

template <typename tscalar>
std::ostream& operator<<(std::ostream& stream, const std::vector<tscalar>& values)
{
    for (const auto& value : values)
    {
        stream << value << ',';
    }
    return stream;
}

static auto make_features()
{
    return features_t
    {
        feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}),
        feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)),
        feature_t{"f64"}.scalar(feature_type::float64),
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

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 4, 1 - sample);
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
    static auto expected_select4()
    {
        return make_tensor<scalar_t>(make_dims(10),
            1, 0, -1, -2, -3, -4, -5, -6, -7, -8);
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
    tensor_size_t expected_samples,
    const tensor1d_t& expected_min, const tensor1d_t& expected_max,
    const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev, scalar_t eps = 1e-12)
{
    for (auto ex : {execution::par, execution::seq})
    {
        flatten_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = generator.flatten_stats(ex, 3));
        UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
        UTEST_CHECK_TENSOR_CLOSE(stats.min(), expected_min, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.max(), expected_max, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.mean(), expected_mean, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.stdev(), expected_stdev, eps);
    }
}

static void check_flatten_stats(const dataset_generator_t& generator,
    tensor_size_t expected_samples,
    const tensor1d_t& expected_min, const tensor1d_t& expected_max,
    const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev)
{
    check_flatten_stats0(generator, expected_samples, expected_min, expected_max, expected_mean, expected_stdev);

    generator.shuffle(1);
    check_flatten_stats0(generator, expected_samples, expected_min, expected_max, expected_mean, expected_stdev);

    generator.shuffle(0);
    check_flatten_stats0(generator, expected_samples, expected_min, expected_max, expected_mean, expected_stdev);

    generator.unshuffle();
    check_flatten_stats0(generator, expected_samples, expected_min, expected_max, expected_mean, expected_stdev);
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

static void check_targets_sclass_stats(const dataset_generator_t& generator,
    const indices_t& expected_class_counts,
    const tensor1d_t& expected_sample_weights, scalar_t eps = 1e-12)
{
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_REQUIRE_NOTHROW(stats = generator.targets_stats(ex, 3));
        UTEST_REQUIRE_NOTHROW(std::get<sclass_stats_t>(stats));
        UTEST_CHECK_TENSOR_EQUAL(std::get<sclass_stats_t>(stats).class_counts(), expected_class_counts);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(stats), expected_sample_weights, eps);

        stats = sclass_stats_t{42};
        UTEST_CHECK_THROW(generator.sample_weights(stats), std::runtime_error);

        stats = mclass_stats_t{expected_class_counts.size()};
        UTEST_CHECK_THROW(generator.sample_weights(stats), std::runtime_error);
    }
}

static void check_targets_mclass_stats(const dataset_generator_t& generator,
    const indices_t& expected_class_counts,
    const tensor1d_t& expected_sample_weights, scalar_t eps = 1e-12)
{
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_REQUIRE_NOTHROW(stats = generator.targets_stats(ex, 3));
        UTEST_REQUIRE_NOTHROW(std::get<mclass_stats_t>(stats));
        UTEST_CHECK_EQUAL(std::get<mclass_stats_t>(stats).class_counts(), expected_class_counts);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(stats), expected_sample_weights, eps);

        stats = mclass_stats_t{42};
        UTEST_CHECK_THROW(generator.sample_weights(stats), std::runtime_error);

        stats = sclass_stats_t{expected_class_counts.size() / 2};
        UTEST_CHECK_THROW(generator.sample_weights(stats), std::runtime_error);
    }
}

static void check_targets_scalar_stats(const dataset_generator_t& generator,
    tensor_size_t expected_samples,
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
        UTEST_CHECK_EQUAL(std::get<scalar_stats_t>(stats).samples(), expected_samples);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).min(), expected_min, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).max(), expected_max, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).mean(), expected_mean, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).stdev(), expected_stdev, eps);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(stats), expected_sample_weights, eps);
    }
}

UTEST_BEGIN_MODULE(test_dataset_generator)

UTEST_CASE(unsupervised)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), string_t::npos);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 5);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));

    check_select(generator, 0, dataset.expected_select0());
    check_select(generator, 1, dataset.expected_select1());
    check_select(generator, 2, dataset.expected_select2());
    check_select(generator, 3, dataset.expected_select3());
    check_select(generator, 4, dataset.expected_select4());
    check_select_stats(generator, make_indices(1), make_indices(0), make_indices(2, 4), make_indices(3));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 11),
        -1, +1, +1, +1, -1, 0, 1, 0, 0, 0, +1,
        +0, +0, +0, -1, +1, 1, 0, 0, 0, 0, +0,
        +0, +0, +0, -1, +1, 2, 3, 2, 2, 2, -1,
        +1, -1, -1, +1, -1, 3, 0, 0, 0, 0, -2,
        +0, +0, +0, -1, +1, 4, 5, 4, 4, 4, -3,
        +0, +0, +0, -1, +1, 5, 0, 0, 0, 0, -4,
        -1, +1, +1, +1, -1, 6, 7, 6, 6, 6, -5,
        +0, +0, +0, -1, +1, 7, 0, 0, 0, 0, -6,
        +0, +0, +0, -1, +1, 8, 9, 8, 8, 8, -7,
        +1, -1, -1, +1, -1, 9, 0, 0, 0, 0, -8),
        make_indices(0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4));

    generator.drop(0);
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 11),
        +0, +0, +0, +1, -1, 0, 1, 0, 0, 0, +1,
        +0, +0, +0, -1, +1, 1, 0, 0, 0, 0, +0,
        +0, +0, +0, -1, +1, 2, 3, 2, 2, 2, -1,
        +0, +0, +0, +1, -1, 3, 0, 0, 0, 0, -2,
        +0, +0, +0, -1, +1, 4, 5, 4, 4, 4, -3,
        +0, +0, +0, -1, +1, 5, 0, 0, 0, 0, -4,
        +0, +0, +0, +1, -1, 6, 7, 6, 6, 6, -5,
        +0, +0, +0, -1, +1, 7, 0, 0, 0, 0, -6,
        +0, +0, +0, -1, +1, 8, 9, 8, 8, 8, -7,
        +0, +0, +0, +1, -1, 9, 0, 0, 0, 0, -8),
        make_indices(0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4));

    generator.drop(2);
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 11),
        +0, +0, +0, +1, -1, 0, 1, 0, 0, 0, +1,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, 0, +0,
        +0, +0, +0, -1, +1, 0, 3, 2, 2, 2, -1,
        +0, +0, +0, +1, -1, 0, 0, 0, 0, 0, -2,
        +0, +0, +0, -1, +1, 0, 5, 4, 4, 4, -3,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, 0, -4,
        +0, +0, +0, +1, -1, 0, 7, 6, 6, 6, -5,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, 0, -6,
        +0, +0, +0, -1, +1, 0, 9, 8, 8, 8, -7,
        +0, +0, +0, +1, -1, 0, 0, 0, 0, 0, -8),
        make_indices(0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4));

    generator.undrop();
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 11),
        -1, +1, +1, +1, -1, 0, 1, 0, 0, 0, +1,
        +0, +0, +0, -1, +1, 1, 0, 0, 0, 0, +0,
        +0, +0, +0, -1, +1, 2, 3, 2, 2, 2, -1,
        +1, -1, -1, +1, -1, 3, 0, 0, 0, 0, -2,
        +0, +0, +0, -1, +1, 4, 5, 4, 4, 4, -3,
        +0, +0, +0, -1, +1, 5, 0, 0, 0, 0, -4,
        -1, +1, +1, +1, -1, 6, 7, 6, 6, 6, -5,
        +0, +0, +0, -1, +1, 7, 0, 0, 0, 0, -6,
        +0, +0, +0, -1, +1, 8, 9, 8, 8, 8, -7,
        +1, -1, -1, +1, -1, 9, 0, 0, 0, 0, -8),
        make_indices(0, 0, 0, 1, 1,  2, 3, 3, 3, 3, 4));

    check_flatten_stats(
        generator, 10,
        make_tensor<scalar_t>(make_dims(11), -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -8),
        make_tensor<scalar_t>(make_dims(11), +1, +1, +1, +1, +1, 9, 9, 8, 8, 8, +1),
        make_tensor<scalar_t>(make_dims(11), 0, 0, 0, -0.2, +0.2, 4.5, 2.5, 2, 2, 2, -3.5),
        make_tensor<scalar_t>(make_dims(11),
            0.666666666667, 0.666666666667, 0.666666666667, 1.032795558989, 1.032795558989,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000, 3.027650354097));

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

    UTEST_REQUIRE_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));

    check_select(generator, 0, dataset.expected_select0());
    check_select(generator, 1, dataset.expected_select2());
    check_select(generator, 2, dataset.expected_select3());
    check_select(generator, 3, dataset.expected_select4());
    check_select_stats(generator, indices_t{}, make_indices(0), make_indices(1, 3), make_indices(2));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 9),
        -1, +1, +1, 0, 1, 0, 0, 0, +1,
        +0, +0, +0, 1, 0, 0, 0, 0, +0,
        +0, +0, +0, 2, 3, 2, 2, 2, -1,
        +1, -1, -1, 3, 0, 0, 0, 0, -2,
        +0, +0, +0, 4, 5, 4, 4, 4, -3,
        +0, +0, +0, 5, 0, 0, 0, 0, -4,
        -1, +1, +1, 6, 7, 6, 6, 6, -5,
        +0, +0, +0, 7, 0, 0, 0, 0, -6,
        +0, +0, +0, 8, 9, 8, 8, 8, -7,
        +1, -1, -1, 9, 0, 0, 0, 0, -8),
        make_indices(0, 0, 0, 1, 2, 2, 2, 2, 3));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(9), -1, -1, -1, 0, 0, 0, 0, 0, -8),
        make_tensor<scalar_t>(make_dims(9), +1, +1, +1, 9, 9, 8, 8, 8, +1),
        make_tensor<scalar_t>(make_dims(9), 0.0, 0.0, 0.0, 4.5, 2.5, 2.0, 2.0, 2.0, -3.5),
        make_tensor<scalar_t>(make_dims(9),
            0.666666666667, 0.666666666667, 0.666666666667,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000, 3.027650354097));

    check_targets(generator, feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}), make_dims(2, 1, 1),
        make_tensor<scalar_t>(make_dims(10, 2, 1, 1),
            +1, -1, -1, +1, -1, +1, +1, -1, -1, +1,
            -1, +1, +1, -1, -1, +1, -1, +1, +1, -1));
    check_targets_sclass_stats(generator,
        make_indices(4, 6),
        make_tensor<scalar_t>(make_dims(10),
            5.0 / 4.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 4.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 4.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 4.0));
}

UTEST_CASE(mclassification)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), 0U);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));

    check_select(generator, 0, dataset.expected_select1());
    check_select(generator, 1, dataset.expected_select2());
    check_select(generator, 2, dataset.expected_select3());
    check_select(generator, 3, dataset.expected_select4());
    check_select_stats(generator, make_indices(0), indices_t{}, make_indices(1, 3), make_indices(2));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 8),
        +1, -1, 0, 1, 0, 0, 0, +1,
        -1, +1, 1, 0, 0, 0, 0, +0,
        -1, +1, 2, 3, 2, 2, 2, -1,
        +1, -1, 3, 0, 0, 0, 0, -2,
        -1, +1, 4, 5, 4, 4, 4, -3,
        -1, +1, 5, 0, 0, 0, 0, -4,
        +1, -1, 6, 7, 6, 6, 6, -5,
        -1, +1, 7, 0, 0, 0, 0, -6,
        -1, +1, 8, 9, 8, 8, 8, -7,
        +1, -1, 9, 0, 0, 0, 0, -8),
        make_indices(0, 0, 1, 2, 2, 2, 2, 3));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(8), -1, -1, 0, 0, 0, 0, 0, -8),
        make_tensor<scalar_t>(make_dims(8), +1, +1, 9, 9, 8, 8, 8, +1),
        make_tensor<scalar_t>(make_dims(8), -0.2, +0.2, 4.5, 2.5, 2.0, 2.0, 2.0, -3.5),
        make_tensor<scalar_t>(make_dims(8),
            1.032795558989, 1.032795558989,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000, 3.027650354097));

    check_targets(generator, feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}), make_dims(3, 1, 1),
        make_tensor<scalar_t>(make_dims(10, 3, 1, 1),
            -1.0, +1.0, +1.0, NaN, NaN, NaN, NaN, NaN, NaN,
            +1.0, -1.0, -1.0, NaN, NaN, NaN, NaN, NaN, NaN,
            -1.0, +1.0, +1.0, NaN, NaN, NaN, NaN, NaN, NaN,
            +1.0, -1.0, -1.0));
    check_targets_mclass_stats(generator,
        make_indices(0, 2, 0, 0, 2, 0),
        make_tensor<scalar_t>(make_dims(10), 1, 0, 0, 1, 0, 0, 1, 0, 0, 1));
}

UTEST_CASE(regression)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), 2U);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>(execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));

    check_select(generator, 0, dataset.expected_select0());
    check_select(generator, 1, dataset.expected_select1());
    check_select(generator, 2, dataset.expected_select3());
    check_select(generator, 3, dataset.expected_select4());
    check_select_stats(generator, make_indices(1), make_indices(0), make_indices(3), make_indices(2));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 10),
        -1, +1, +1, +1, -1, 1, 0, 0, 0, +1,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, +0,
        +0, +0, +0, -1, +1, 3, 2, 2, 2, -1,
        +1, -1, -1, +1, -1, 0, 0, 0, 0, -2,
        +0, +0, +0, -1, +1, 5, 4, 4, 4, -3,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, -4,
        -1, +1, +1, +1, -1, 7, 6, 6, 6, -5,
        +0, +0, +0, -1, +1, 0, 0, 0, 0, -6,
        +0, +0, +0, -1, +1, 9, 8, 8, 8, -7,
        +1, -1, -1, +1, -1, 0, 0, 0, 0, -8),
        make_indices(0, 0, 0, 1, 1, 2, 2, 2, 2, 3));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(10), -1, -1, -1, -1, -1, 0, 0, 0, 0, -8),
        make_tensor<scalar_t>(make_dims(10), +1, +1, +1, +1, +1, 9, 8, 8, 8, +1),
        make_tensor<scalar_t>(make_dims(10), 0, 0, 0, -0.2, 0.2, 2.5, 2, 2, 2, -3.5),
        make_tensor<scalar_t>(make_dims(10),
            0.666666666667, 0.666666666667, 0.666666666667, 1.032795558989, 1.032795558989,
            3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000, 3.027650354097));

    check_targets(generator, feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)), make_dims(1, 1, 1),
        make_tensor<scalar_t>(make_dims(10, 1, 1, 1), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    check_targets_scalar_stats(generator, 10,
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

    UTEST_REQUIRE_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));

    check_select(generator, 0, dataset.expected_select0());
    check_select(generator, 1, dataset.expected_select1());
    check_select(generator, 2, dataset.expected_select2());
    check_select(generator, 3, dataset.expected_select4());
    check_select_stats(generator, make_indices(1), make_indices(0), make_indices(2, 3), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 7),
        -1, +1, +1, +1, -1, 0, +1,
        +0, +0, +0, -1, +1, 1, +0,
        +0, +0, +0, -1, +1, 2, -1,
        +1, -1, -1, +1, -1, 3, -2,
        +0, +0, +0, -1, +1, 4, -3,
        +0, +0, +0, -1, +1, 5, -4,
        -1, +1, +1, +1, -1, 6, -5,
        +0, +0, +0, -1, +1, 7, -6,
        +0, +0, +0, -1, +1, 8, -7,
        +1, -1, -1, +1, -1, 9, -8),
        make_indices(0, 0, 0, 1, 1, 2, 3));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(7), -1, -1, -1, -1, -1, 0, -8),
        make_tensor<scalar_t>(make_dims(7), +1, +1, +1, +1, +1, 9, +1),
        make_tensor<scalar_t>(make_dims(7), 0, 0, 0, -0.2, +0.2, 4.5, -3.5),
        make_tensor<scalar_t>(make_dims(7),
            0.666666666667, 0.666666666667, 0.666666666667, 1.032795558989, 1.032795558989, 3.027650354097, 3.027650354097));

    check_targets(generator, feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)), make_dims(2, 1, 2),
        make_tensor<scalar_t>(make_dims(10, 2, 1, 2),
            1, 0, 0, 0, NaN, NaN, NaN, NaN,
            3, 2, 2, 2, NaN, NaN, NaN, NaN,
            5, 4, 4, 4, NaN, NaN, NaN, NaN,
            7, 6, 6, 6, NaN, NaN, NaN, NaN,
            9, 8, 8, 8, NaN, NaN, NaN, NaN));
    check_targets_scalar_stats(generator, 5,
        make_tensor<scalar_t>(make_dims(4), 1, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(4), 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(4), 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(4), 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168));
}

UTEST_CASE(unsupervised_quadratic_scalar)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), string_t::npos);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<scalar2scalar_pairwise_generator_t<pairwise_product_t>>(
        execution::par, struct2scalar::off);

    UTEST_REQUIRE_EQUAL(generator.features(), 3);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"product(f32[0],f32[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"product(f32[0],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"product(f64[0],f64[0])"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(10), 0, 1, 4, 9, 16, 25, 36, 49, 64, 81));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(10), 0, 0, -2, -6, -12, -20, -30, -42, -56, -72));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(10), 1, 0, 1, 4, 9, 16, 25, 36, 49, 64));
    check_select_stats(generator, indices_t{}, indices_t{}, make_indices(0, 1, 2), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 3),
        0, 0, 1,
        1, 0, 0,
        4, -2, 1,
        9, -6, 4,
        16, -12, 9,
        25, -20, 16,
        36, -30, 25,
        49, -42, 36,
        64, -56, 49,
        81, -72, 64),
        make_indices(0, 1, 2));

    generator.drop(1);
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 3),
        0, 0, 1,
        1, 0, 0,
        4, 0, 1,
        9, 0, 4,
        16, 0, 9,
        25, 0, 16,
        36, 0, 25,
        49, 0, 36,
        64, 0, 49,
        81, 0, 64),
        make_indices(0, 1, 2));

    generator.undrop();
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 3),
        0, 0, 1,
        1, 0, 0,
        4, -2, 1,
        9, -6, 4,
        16, -12, 9,
        25, -20, 16,
        36, -30, 25,
        49, -42, 36,
        64, -56, 49,
        81, -72, 64),
        make_indices(0, 1, 2));

    check_flatten_stats(
        generator, 10,
        make_tensor<scalar_t>(make_dims(3), 0, -72, 0),
        make_tensor<scalar_t>(make_dims(3), 81, 0, 64),
        make_tensor<scalar_t>(make_dims(3), 28.5, -24.0, 20.5),
        make_tensor<scalar_t>(make_dims(3), 28.304887681577, 25.403411844344, 22.535157717073));
}

UTEST_CASE(unsupervised_quadratic_mixed)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), string_t::npos);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<scalar2scalar_pairwise_generator_t<pairwise_product_t>>(
        execution::par, struct2scalar::on, make_indices(0, 1, 3, 4));

    UTEST_REQUIRE_EQUAL(generator.features(), 15);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"product(u8s[0],u8s[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"product(u8s[0],u8s[1])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"product(u8s[0],u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"product(u8s[0],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"product(u8s[0],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"product(u8s[1],u8s[1])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(6), feature_t{"product(u8s[1],u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(7), feature_t{"product(u8s[1],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(8), feature_t{"product(u8s[1],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(9), feature_t{"product(u8s[2],u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(10), feature_t{"product(u8s[2],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(11), feature_t{"product(u8s[2],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(12), feature_t{"product(u8s[3],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(13), feature_t{"product(u8s[3],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(14), feature_t{"product(f64[0],f64[0])"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(10), 1, NaN, 9, NaN, 25, NaN, 49, NaN, 81, NaN));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(10), 0, NaN, 6, NaN, 20, NaN, 42, NaN, 72, NaN));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(10), 0, NaN, 6, NaN, 20, NaN, 42, NaN, 72, NaN));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(10), 0, NaN, 6, NaN, 20, NaN, 42, NaN, 72, NaN));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(10), 1, NaN, -3, NaN, -15, NaN, -35, NaN, -63, NaN));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 6, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 7, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 8, make_tensor<scalar_t>(make_dims(10), 0, NaN, -2, NaN, -12, NaN, -30, NaN, -56, NaN));
    check_select(generator, 9, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 10, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 11, make_tensor<scalar_t>(make_dims(10), 0, NaN, -2, NaN, -12, NaN, -30, NaN, -56, NaN));
    check_select(generator, 12, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 13, make_tensor<scalar_t>(make_dims(10), 0, NaN, -2, NaN, -12, NaN, -30, NaN, -56, NaN));
    check_select(generator, 14, make_tensor<scalar_t>(make_dims(10), 1, 0, 1, 4, 9, 16, 25, 36, 49, 64));
    check_select_stats(generator, indices_t{}, indices_t{}, arange(0, 15), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 15),
        1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        9, 6, 6, 6, -3, 4, 4, 4, -2, 4, 4, -2, 4, -2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
        25, 20, 20, 20, -15, 16, 16, 16, -12, 16, 16, -12, 16, -12, 9,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16,
        49, 42, 42, 42, -35, 36, 36, 36, -30, 36, 36, -30, 36, -30, 25,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36,
        81, 72, 72, 72, -63, 64, 64, 64, -56, 64, 64, -56, 64, -56, 49,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64),
        arange(0, 15));

    check_flatten_stats(
        generator, 10,
        make_tensor<scalar_t>(make_dims(15), 0, 0, 0, 0, -63, 0, 0, 0, -56, 0, 0, -56, 0, -56, 0),
        make_tensor<scalar_t>(make_dims(15), 81, 72, 72, 72, 1, 64, 64, 64, 0, 64, 64, 0, 64, 0, 64),
        make_tensor<scalar_t>(make_dims(15), 16.5, 14.0, 14.0, 14.0, -11.5, 12.0, 12.0, 12.0, -10.0, 12.0, 12.0, -10.0, 12.0, -10.0, 20.5),
        make_tensor<scalar_t>(make_dims(15),
            27.781888584712, 24.549270186029, 24.549270186029, 24.549270186029, 21.360659581993,
            21.664102412363, 21.664102412363, 21.664102412363, 18.808981306221, 21.664102412363,
            21.664102412363, 18.808981306221, 21.664102412363, 18.808981306221, 22.535157717073));
}

UTEST_CASE(select_scalar_components)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, indices_t{});
        const auto expected = std::vector<tensor_size_t>{2, 0, 4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(2));
        const auto expected = std::vector<tensor_size_t>{2, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(3));
        const auto expected = std::vector<tensor_size_t>{};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(2, 3, 4));
        const auto expected = std::vector<tensor_size_t>{2, 0, 4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, indices_t{});
        const auto expected = std::vector<tensor_size_t>{2, 0, 3, 0, 3, 1, 3, 2, 3, 3, 4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, make_indices(1, 4));
        const auto expected = std::vector<tensor_size_t>{4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, make_indices(1, 3,  4));
        const auto expected = std::vector<tensor_size_t>{3, 0, 3, 1, 3, 2, 3, 3, 4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
}

UTEST_END_MODULE()
