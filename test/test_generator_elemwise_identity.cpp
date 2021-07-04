#include <utest/utest.h>
#include "fixture/generator.h"
#include <nano/generator/elemwise_identity.h>

using namespace nano;

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

class fixture_dataset_t final : public dataset_t
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
            hits(2) = ((sample % 6) == 0) ? 1 : 0;
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

static auto make_generator(const dataset_t& dataset)
{
    auto generator = dataset_generator_t{dataset};
    UTEST_CHECK_NOTHROW(generator.add<elemwise_generator_t<sclass_identity_t>>());
    UTEST_CHECK_NOTHROW(generator.add<elemwise_generator_t<mclass_identity_t>>());
    UTEST_CHECK_NOTHROW(generator.add<elemwise_generator_t<scalar_identity_t>>());
    UTEST_CHECK_NOTHROW(generator.add<elemwise_generator_t<struct_identity_t>>());
    UTEST_CHECK_NOTHROW(generator.fit(arange(0, dataset.samples()), execution::par));
    return generator;
}

UTEST_BEGIN_MODULE(test_generator_elemwise_identity)

UTEST_CASE(empty)
{
    const auto dataset = make_dataset(10, string_t::npos);
    const auto generator = dataset_generator_t{dataset};

    UTEST_CHECK_EQUAL(generator.columns(), 0);
    UTEST_CHECK_EQUAL(generator.features(), 0);
}

UTEST_CASE(unsupervised)
{
    const auto dataset = make_dataset(10, string_t::npos);
    const auto generator = make_generator(dataset);

    UTEST_REQUIRE_EQUAL(generator.features(), 5);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));

    check_select(generator, 0, fixture_dataset_t::expected_select1());
    check_select(generator, 1, fixture_dataset_t::expected_select0());
    check_select(generator, 2, fixture_dataset_t::expected_select2());
    check_select(generator, 3, fixture_dataset_t::expected_select4());
    check_select(generator, 4, fixture_dataset_t::expected_select3());
    check_select_stats(generator, make_indices(0), make_indices(1), make_indices(2, 3), make_indices(4));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 11),
        +1, -1, -1, +1, +1, 0, +1, 1, 0, 0, 0,
        -1, +1, +0, +0, +0, 1, +0, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, 2, -1, 3, 2, 2, 2,
        +1, -1, +1, -1, -1, 3, -2, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, 4, -3, 5, 4, 4, 4,
        -1, +1, +0, +0, +0, 5, -4, 0, 0, 0, 0,
        +1, -1, -1, +1, +1, 6, -5, 7, 6, 6, 6,
        -1, +1, +0, +0, +0, 7, -6, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, 8, -7, 9, 8, 8, 8,
        +1, -1, +1, -1, -1, 9, -8, 0, 0, 0, 0),
        make_indices(0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 4));

    generator.drop(0);
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 11),
        +0, +0, -1, +1, +1, 0, +1, 1, 0, 0, 0,
        +0, +0, +0, +0, +0, 1, +0, 0, 0, 0, 0,
        +0, +0, +0, +0, +0, 2, -1, 3, 2, 2, 2,
        +0, +0, +1, -1, -1, 3, -2, 0, 0, 0, 0,
        +0, +0, +0, +0, +0, 4, -3, 5, 4, 4, 4,
        +0, +0, +0, +0, +0, 5, -4, 0, 0, 0, 0,
        +0, +0, -1, +1, +1, 6, -5, 7, 6, 6, 6,
        +0, +0, +0, +0, +0, 7, -6, 0, 0, 0, 0,
        +0, +0, +0, +0, +0, 8, -7, 9, 8, 8, 8,
        +0, +0, +1, -1, -1, 9, -8, 0, 0, 0, 0),
        make_indices(0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 4));

    generator.drop(2);
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 11),
        +0, +0, -1, +1, +1, 0, +1, 1, 0, 0, 0,
        +0, +0, +0, +0, +0, 0, +0, 0, 0, 0, 0,
        +0, +0, +0, +0, +0, 0, -1, 3, 2, 2, 2,
        +0, +0, +1, -1, -1, 0, -2, 0, 0, 0, 0,
        +0, +0, +0, +0, +0, 0, -3, 5, 4, 4, 4,
        +0, +0, +0, +0, +0, 0, -4, 0, 0, 0, 0,
        +0, +0, -1, +1, +1, 0, -5, 7, 6, 6, 6,
        +0, +0, +0, +0, +0, 0, -6, 0, 0, 0, 0,
        +0, +0, +0, +0, +0, 0, -7, 9, 8, 8, 8,
        +0, +0, +1, -1, -1, 0, -8, 0, 0, 0, 0),
        make_indices(0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 4));

    generator.undrop();
    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 11),
        +1, -1, -1, +1, +1, 0, +1, 1, 0, 0, 0,
        -1, +1, +0, +0, +0, 1, +0, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, 2, -1, 3, 2, 2, 2,
        +1, -1, +1, -1, -1, 3, -2, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, 4, -3, 5, 4, 4, 4,
        -1, +1, +0, +0, +0, 5, -4, 0, 0, 0, 0,
        +1, -1, -1, +1, +1, 6, -5, 7, 6, 6, 6,
        -1, +1, +0, +0, +0, 7, -6, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, 8, -7, 9, 8, 8, 8,
        +1, -1, +1, -1, -1, 9, -8, 0, 0, 0, 0),
        make_indices(0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 4));

    check_flatten_stats(
        generator, 10,
        make_tensor<scalar_t>(make_dims(11), -1, -1, -1, -1, -1, 0, -8, 0, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(11), +1, +1, +1, +1, +1, 9, +1, 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(11), -0.2, +0.2, 0, 0, 0, 4.5, -3.5, 2.5, 2, 2, 2),
        make_tensor<scalar_t>(make_dims(11),
            1.032795558989, 1.032795558989, 0.666666666667, 0.666666666667, 0.666666666667,
            3.027650354097, 3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000));

    const auto samples = arange(0, generator.dataset().samples());
    {
        tensor4d_t targets_buffer;
        tensor4d_cmap_t targets_cmap;
        UTEST_CHECK_EQUAL(generator.target(), feature_t{});
        UTEST_CHECK_EQUAL(generator.target_dims(), make_dims(0, 0, 0));
        UTEST_CHECK_THROW(targets_cmap = generator.targets(samples, targets_buffer), std::runtime_error);
    }
    for (auto ex : {execution::par, execution::seq})
    {
        targets_stats_t stats;
        UTEST_CHECK_NOTHROW(stats = generator.targets_stats(samples, ex, 3));
        UTEST_CHECK_EQUAL(std::holds_alternative<scalar_stats_t>(stats), false);
        UTEST_CHECK_EQUAL(std::holds_alternative<sclass_stats_t>(stats), false);
        UTEST_CHECK_EQUAL(std::holds_alternative<mclass_stats_t>(stats), false);
    }
}

UTEST_CASE(sclassification)
{
    const auto dataset = make_dataset(10, 1U);
    const auto generator = make_generator(dataset);

    UTEST_REQUIRE_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));

    check_select(generator, 0, fixture_dataset_t::expected_select0());
    check_select(generator, 1, fixture_dataset_t::expected_select2());
    check_select(generator, 2, fixture_dataset_t::expected_select4());
    check_select(generator, 3, fixture_dataset_t::expected_select3());
    check_select_stats(generator, indices_t{}, make_indices(0), make_indices(1, 2), make_indices(3));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 9),
        -1, +1, +1, 0, +1, 1, 0, 0, 0,
        +0, +0, +0, 1, +0, 0, 0, 0, 0,
        +0, +0, +0, 2, -1, 3, 2, 2, 2,
        +1, -1, -1, 3, -2, 0, 0, 0, 0,
        +0, +0, +0, 4, -3, 5, 4, 4, 4,
        +0, +0, +0, 5, -4, 0, 0, 0, 0,
        -1, +1, +1, 6, -5, 7, 6, 6, 6,
        +0, +0, +0, 7, -6, 0, 0, 0, 0,
        +0, +0, +0, 8, -7, 9, 8, 8, 8,
        +1, -1, -1, 9, -8, 0, 0, 0, 0),
        make_indices(0, 0, 0, 1, 2, 3, 3, 3, 3));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(9), -1, -1, -1, 0, -8, 0, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(9), +1, +1, +1, 9, +1, 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(9), 0.0, 0.0, 0.0, 4.5, -3.5, 2.5, 2.0, 2.0, 2.0),
        make_tensor<scalar_t>(make_dims(9),
            0.666666666667, 0.666666666667, 0.666666666667,
            3.027650354097, 3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000));

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
    const auto dataset = make_dataset(10, 0U);
    const auto generator = make_generator(dataset);

    UTEST_REQUIRE_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));

    check_select(generator, 0, fixture_dataset_t::expected_select1());
    check_select(generator, 1, fixture_dataset_t::expected_select2());
    check_select(generator, 2, fixture_dataset_t::expected_select4());
    check_select(generator, 3, fixture_dataset_t::expected_select3());
    check_select_stats(generator, make_indices(0), indices_t{}, make_indices(1, 2), make_indices(3));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 8),
        +1, -1, 0, +1, 1, 0, 0, 0,
        -1, +1, 1, +0, 0, 0, 0, 0,
        -1, +1, 2, -1, 3, 2, 2, 2,
        +1, -1, 3, -2, 0, 0, 0, 0,
        -1, +1, 4, -3, 5, 4, 4, 4,
        -1, +1, 5, -4, 0, 0, 0, 0,
        +1, -1, 6, -5, 7, 6, 6, 6,
        -1, +1, 7, -6, 0, 0, 0, 0,
        -1, +1, 8, -7, 9, 8, 8, 8,
        +1, -1, 9, -8, 0, 0, 0, 0),
        make_indices(0, 0, 1, 2, 3, 3, 3, 3));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(8), -1, -1, 0, -8, 0, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(8), +1, +1, 9, +1, 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(8), -0.2, +0.2, 4.5, -3.5, 2.5, 2.0, 2.0, 2.0),
        make_tensor<scalar_t>(make_dims(8),
            1.032795558989, 1.032795558989,
            3.027650354097, 3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000));

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
    const auto dataset = make_dataset(10, 2U);
    const auto generator = make_generator(dataset);

    UTEST_REQUIRE_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));

    check_select(generator, 0, fixture_dataset_t::expected_select1());
    check_select(generator, 1, fixture_dataset_t::expected_select0());
    check_select(generator, 2, fixture_dataset_t::expected_select4());
    check_select(generator, 3, fixture_dataset_t::expected_select3());
    check_select_stats(generator, make_indices(0), make_indices(1), make_indices(2), make_indices(3));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 10),
        +1, -1, -1, +1, +1, +1, 1, 0, 0, 0,
        -1, +1, +0, +0, +0, +0, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, -1, 3, 2, 2, 2,
        +1, -1, +1, -1, -1, -2, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, -3, 5, 4, 4, 4,
        -1, +1, +0, +0, +0, -4, 0, 0, 0, 0,
        +1, -1, -1, +1, +1, -5, 7, 6, 6, 6,
        -1, +1, +0, +0, +0, -6, 0, 0, 0, 0,
        -1, +1, +0, +0, +0, -7, 9, 8, 8, 8,
        +1, -1, +1, -1, -1, -8, 0, 0, 0, 0),
        make_indices(0, 0, 1, 1, 1, 2, 3, 3, 3, 3));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(10), -1, -1, -1, -1, -1, -8, 0, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(10), +1, +1, +1, +1, +1, +1, 9, 8, 8, 8),
        make_tensor<scalar_t>(make_dims(10), -0.2, +0.2, 0, 0, 0, -3.5, 2.5, 2, 2, 2),
        make_tensor<scalar_t>(make_dims(10),
            1.032795558989, 1.032795558989, 0.666666666667, 0.666666666667, 0.666666666667,
            3.027650354097, 3.374742788553, 2.981423970000, 2.981423970000, 2.981423970000));

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
    const auto dataset = make_dataset(10, 3U);
    const auto generator = make_generator(dataset);

    UTEST_REQUIRE_EQUAL(generator.features(), 4);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"f64"}.scalar(feature_type::float64, make_dims(1, 1, 1)));

    check_select(generator, 0, fixture_dataset_t::expected_select1());
    check_select(generator, 1, fixture_dataset_t::expected_select0());
    check_select(generator, 2, fixture_dataset_t::expected_select2());
    check_select(generator, 3, fixture_dataset_t::expected_select4());
    check_select_stats(generator, make_indices(0), make_indices(1), make_indices(2, 3), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 7),
        +1, -1, -1, +1, +1, 0, +1,
        -1, +1, +0, +0, +0, 1, +0,
        -1, +1, +0, +0, +0, 2, -1,
        +1, -1, +1, -1, -1, 3, -2,
        -1, +1, +0, +0, +0, 4, -3,
        -1, +1, +0, +0, +0, 5, -4,
        +1, -1, -1, +1, +1, 6, -5,
        -1, +1, +0, +0, +0, 7, -6,
        -1, +1, +0, +0, +0, 8, -7,
        +1, -1, +1, -1, -1, 9, -8),
        make_indices(0, 0, 1, 1, 1, 2, 3));

    check_flatten_stats(generator, 10,
        make_tensor<scalar_t>(make_dims(7), -1, -1, -1, -1, -1, 0, -8),
        make_tensor<scalar_t>(make_dims(7), +1, +1, +1, +1, +1, 9, +1),
        make_tensor<scalar_t>(make_dims(7), -0.2, +0.2, 0, 0, 0, 4.5, -3.5),
        make_tensor<scalar_t>(make_dims(7),
            1.032795558989, 1.032795558989, 0.666666666667, 0.666666666667, 0.666666666667, 3.027650354097, 3.027650354097));

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

UTEST_END_MODULE()
