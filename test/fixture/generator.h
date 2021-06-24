#include <utest/utest.h>
#include <nano/generator.h>

using namespace nano;

static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

static auto make_samples(const dataset_generator_t& generator)
{
    const auto samples = generator.dataset().samples();

    return std::vector<indices_t>
    {
        arange(0, samples),
        arange(0, samples / 2),
        arange(samples / 2, samples)
    };
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
[[maybe_unused]] static void check_select0(const dataset_generator_t& generator,
    indices_cmap_t samples, tensor_size_t feature, const tensor_t<tstorage, tscalar, trank>& expected)
{
    tensor_t<tstorage, tscalar, trank> buffer;
    decltype(generator.select(samples, feature, buffer)) storage;

    UTEST_CHECK_NOTHROW(storage = generator.select(samples, feature, buffer));
    UTEST_CHECK_TENSOR_CLOSE(storage, expected.indexed(samples), 1e-12);

    generator.shuffle(feature);
    const auto shuffle = generator.shuffled(samples, feature);
    UTEST_REQUIRE_EQUAL(shuffle.size(), samples.size());
    UTEST_CHECK(std::is_permutation(shuffle.begin(), shuffle.end(), samples.begin()));
    //UTEST_CHECK_NOT_EQUAL(shuffle, samples);
    UTEST_CHECK_NOTHROW(storage = generator.select(samples, feature, buffer));
    UTEST_CHECK_TENSOR_CLOSE(storage, expected.indexed(shuffle), 1e-12);

    const auto shuffle2 = generator.shuffled(samples, feature);
    UTEST_CHECK_TENSOR_EQUAL(shuffle, shuffle2);

    generator.unshuffle();
    UTEST_CHECK_NOTHROW(storage = generator.select(samples, feature, buffer));
    UTEST_CHECK_TENSOR_CLOSE(storage, expected.indexed(samples), 1e-12);

    generator.drop(feature);
    tensor_t<tstorage, tscalar, trank> expected_dropped = expected.indexed(samples);
    UTEST_CHECK_NOTHROW(storage = generator.select(samples, feature, buffer));
    switch (generator.feature(feature).type())
    {
    case feature_type::sclass:  expected_dropped.full(-1); break; // NOLINT(bugprone-branch-clone)
    case feature_type::mclass:  expected_dropped.full(-1); break;
    default:                    expected_dropped.full(static_cast<tscalar>(NaN)); break;
    }
    UTEST_CHECK_TENSOR_CLOSE(storage, expected_dropped, 1e-12);

    generator.undrop();
    UTEST_CHECK_NOTHROW(storage = generator.select(samples, feature, buffer));
    UTEST_CHECK_TENSOR_CLOSE(storage, expected.indexed(samples), 1e-12);
}

[[maybe_unused]] static void check_select(const dataset_generator_t& generator, tensor_size_t feature, const sclass_mem_t& expected)
{
    mclass_mem_t mclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    for (const auto& samples : make_samples(generator))
    {
        UTEST_CHECK_THROW(generator.select(samples, feature, mclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(samples, feature, scalar_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(samples, feature, struct_buffer), std::runtime_error);
        check_select0(generator, samples, feature, expected);
    }
}

[[maybe_unused]] static void check_select(const dataset_generator_t& generator, tensor_size_t feature, const mclass_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    for (const auto& samples : make_samples(generator))
    {
        UTEST_CHECK_THROW(generator.select(samples, feature, sclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(samples, feature, scalar_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(samples, feature, struct_buffer), std::runtime_error);
        check_select0(generator, samples, feature, expected);
    }
}

[[maybe_unused]] static void check_select(const dataset_generator_t& generator, tensor_size_t feature, const scalar_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    mclass_mem_t mclass_buffer;
    struct_mem_t struct_buffer;

    for (const auto& samples : make_samples(generator))
    {
        UTEST_CHECK_THROW(generator.select(samples, feature, sclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(samples, feature, mclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(samples, feature, struct_buffer), std::runtime_error);
        check_select0(generator, samples, feature, expected);
    }
}

[[maybe_unused]] static void check_select(const dataset_generator_t& generator, tensor_size_t feature, const struct_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    mclass_mem_t mclass_buffer;
    scalar_mem_t scalar_buffer;

    for (const auto& samples : make_samples(generator))
    {
        UTEST_CHECK_THROW(generator.select(samples, feature, sclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(samples, feature, mclass_buffer), std::runtime_error);
        UTEST_CHECK_THROW(generator.select(samples, feature, scalar_buffer), std::runtime_error);
        check_select0(generator, samples, feature, expected);
    }
}

[[maybe_unused]] static void check_flatten(const dataset_generator_t& generator,
    const tensor2d_t& expected_flatten, const indices_t& expected_column2features, scalar_t eps = 1e-12)
{
    tensor2d_t flatten_buffer;
    tensor2d_cmap_t flatten_cmap;

    for (const auto& samples : make_samples(generator))
    {
        UTEST_REQUIRE_EQUAL(generator.columns(), expected_flatten.size<1>());
        UTEST_CHECK_NOTHROW(flatten_cmap = generator.flatten(samples, flatten_buffer));
        UTEST_CHECK_TENSOR_CLOSE(flatten_cmap, expected_flatten.indexed(samples), eps);
    }

    UTEST_REQUIRE_EQUAL(generator.columns(), expected_column2features.size());
    for (tensor_size_t column = 0; column < generator.columns(); ++ column)
    {
        UTEST_CHECK_EQUAL(generator.column2feature(column), expected_column2features(column));
    }
}

[[maybe_unused]] static void check_select_stats(const dataset_generator_t& generator,
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

[[maybe_unused]] static void check_flatten_stats0(const dataset_generator_t& generator,
    tensor_size_t expected_samples,
    const tensor1d_t& expected_min, const tensor1d_t& expected_max,
    const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev, scalar_t eps = 1e-12)
{
    const auto samples = arange(0, generator.dataset().samples());
    for (auto ex : {execution::par, execution::seq})
    {
        flatten_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = generator.flatten_stats(samples, ex, 3));
        UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
        UTEST_CHECK_TENSOR_CLOSE(stats.min(), expected_min, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.max(), expected_max, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.mean(), expected_mean, eps);
        UTEST_CHECK_TENSOR_CLOSE(stats.stdev(), expected_stdev, eps);
    }
}

[[maybe_unused]] static void check_flatten_stats(const dataset_generator_t& generator,
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

[[maybe_unused]] static void check_targets(const dataset_generator_t& generator,
    const feature_t& expected_target, tensor3d_dims_t expected_target_dims,
    const tensor4d_t& expected_targets, scalar_t eps = 1e-12)
{
    const auto samples = arange(0, expected_targets.size<0>());

    tensor4d_t targets_buffer;
    tensor4d_cmap_t targets_cmap;
    UTEST_CHECK_EQUAL(generator.target(), expected_target);
    UTEST_CHECK_EQUAL(generator.target_dims(), expected_target_dims);
    UTEST_REQUIRE_NOTHROW(targets_cmap = generator.targets(samples, targets_buffer));
    UTEST_CHECK_TENSOR_CLOSE(targets_cmap, expected_targets, eps);
}

[[maybe_unused]] static void check_targets_sclass_stats(const dataset_generator_t& generator,
    const indices_t& expected_class_counts,
    const tensor1d_t& expected_sample_weights, scalar_t eps = 1e-12)
{
    const auto samples = arange(0, generator.dataset().samples());
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_REQUIRE_NOTHROW(stats = generator.targets_stats(samples, ex, 3));
        UTEST_REQUIRE_NOTHROW(std::get<sclass_stats_t>(stats));
        UTEST_CHECK_TENSOR_EQUAL(std::get<sclass_stats_t>(stats).class_counts(), expected_class_counts);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(samples, stats), expected_sample_weights, eps);

        stats = sclass_stats_t{42};
        UTEST_CHECK_THROW(generator.sample_weights(samples, stats), std::runtime_error);

        stats = mclass_stats_t{expected_class_counts.size()};
        UTEST_CHECK_THROW(generator.sample_weights(samples, stats), std::runtime_error);
    }
}

[[maybe_unused]] static void check_targets_mclass_stats(const dataset_generator_t& generator,
    const indices_t& expected_class_counts,
    const tensor1d_t& expected_sample_weights, scalar_t eps = 1e-12)
{
    const auto samples = arange(0, generator.dataset().samples());
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_REQUIRE_NOTHROW(stats = generator.targets_stats(samples, ex, 3));
        UTEST_REQUIRE_NOTHROW(std::get<mclass_stats_t>(stats));
        UTEST_CHECK_EQUAL(std::get<mclass_stats_t>(stats).class_counts(), expected_class_counts);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(samples, stats), expected_sample_weights, eps);

        stats = mclass_stats_t{42};
        UTEST_CHECK_THROW(generator.sample_weights(samples, stats), std::runtime_error);

        stats = sclass_stats_t{expected_class_counts.size() / 2};
        UTEST_CHECK_THROW(generator.sample_weights(samples, stats), std::runtime_error);
    }
}

[[maybe_unused]] static void check_targets_scalar_stats(const dataset_generator_t& generator,
    tensor_size_t expected_samples,
    const tensor1d_t& expected_min, const tensor1d_t& expected_max,
    const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev, scalar_t eps = 1e-12)
{
    tensor1d_t expected_sample_weights = tensor1d_t{generator.dataset().samples()};
    expected_sample_weights.full(1.0);

    const auto samples = arange(0, generator.dataset().samples());
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_REQUIRE_NOTHROW(stats = generator.targets_stats(samples, ex, 3));
        UTEST_REQUIRE_NOTHROW(std::get<scalar_stats_t>(stats));
        UTEST_CHECK_EQUAL(std::get<scalar_stats_t>(stats).samples(), expected_samples);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).min(), expected_min, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).max(), expected_max, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).mean(), expected_mean, eps);
        UTEST_CHECK_TENSOR_CLOSE(std::get<scalar_stats_t>(stats).stdev(), expected_stdev, eps);
        UTEST_CHECK_TENSOR_CLOSE(generator.sample_weights(samples, stats), expected_sample_weights, eps);
    }
}
