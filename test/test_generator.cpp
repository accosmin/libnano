#include <utest/utest.h>
#include "fixture/generator.h"
#include <nano/generator/identity.h>
#include <nano/generator/elemwise_scalar.h>
#include <nano/generator/pairwise_scalar.h>

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

UTEST_BEGIN_MODULE(test_dataset_generator)

UTEST_CASE(unsupervised)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<identity_generator_t>();
    generator.fit(arange(0, 10), execution::par);

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

    auto generator = dataset_generator_t{dataset};
    generator.add<identity_generator_t>();
    generator.fit(arange(0, 10), execution::par);

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
    const auto dataset = make_dataset(10, 0U);

    auto generator = dataset_generator_t{dataset};
    generator.add<identity_generator_t>();

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
    const auto dataset = make_dataset(10, 2U);

    auto generator = dataset_generator_t{dataset};
    generator.add<identity_generator_t>();
    generator.fit(arange(0, 10), execution::par);

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
    const auto dataset = make_dataset(10, 3U);

    auto generator = dataset_generator_t{dataset};
    generator.add<identity_generator_t>();
    generator.fit(arange(0, 10), execution::par);

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
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<pairwise_generator_t<product_t>>(struct2scalar::off);
    generator.fit(arange(0, 10), execution::par);

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
}

UTEST_CASE(unsupervised_quadratic_mixed)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<pairwise_generator_t<product_t>>(struct2scalar::on);
    generator.fit(arange(0, 10), execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 21);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"product(f32[0],f32[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"product(f32[0],u8s[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"product(f32[0],u8s[1])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"product(f32[0],u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"product(f32[0],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"product(f32[0],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(6), feature_t{"product(u8s[0],u8s[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(7), feature_t{"product(u8s[0],u8s[1])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(8), feature_t{"product(u8s[0],u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(9), feature_t{"product(u8s[0],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(10), feature_t{"product(u8s[0],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(11), feature_t{"product(u8s[1],u8s[1])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(12), feature_t{"product(u8s[1],u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(13), feature_t{"product(u8s[1],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(14), feature_t{"product(u8s[1],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(15), feature_t{"product(u8s[2],u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(16), feature_t{"product(u8s[2],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(17), feature_t{"product(u8s[2],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(18), feature_t{"product(u8s[3],u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(19), feature_t{"product(u8s[3],f64[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(20), feature_t{"product(f64[0],f64[0])"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(10), 0, 1, 4, 9, 16, 25, 36, 49, 64, 81));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(10), 0, NaN, 6, NaN, 20, NaN, 42, NaN, 72, NaN));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(10), 0, 0, -2, -6, -12, -20, -30, -42, -56, -72));
    check_select(generator, 6, make_tensor<scalar_t>(make_dims(10), 1, NaN, 9, NaN, 25, NaN, 49, NaN, 81, NaN));
    check_select(generator, 7, make_tensor<scalar_t>(make_dims(10), 0, NaN, 6, NaN, 20, NaN, 42, NaN, 72, NaN));
    check_select(generator, 8, make_tensor<scalar_t>(make_dims(10), 0, NaN, 6, NaN, 20, NaN, 42, NaN, 72, NaN));
    check_select(generator, 9, make_tensor<scalar_t>(make_dims(10), 0, NaN, 6, NaN, 20, NaN, 42, NaN, 72, NaN));
    check_select(generator, 10, make_tensor<scalar_t>(make_dims(10), 1, NaN, -3, NaN, -15, NaN, -35, NaN, -63, NaN));
    check_select(generator, 11, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 12, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 13, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 14, make_tensor<scalar_t>(make_dims(10), 0, NaN, -2, NaN, -12, NaN, -30, NaN, -56, NaN));
    check_select(generator, 15, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 16, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 17, make_tensor<scalar_t>(make_dims(10), 0, NaN, -2, NaN, -12, NaN, -30, NaN, -56, NaN));
    check_select(generator, 18, make_tensor<scalar_t>(make_dims(10), 0, NaN, 4, NaN, 16, NaN, 36, NaN, 64, NaN));
    check_select(generator, 19, make_tensor<scalar_t>(make_dims(10), 0, NaN, -2, NaN, -12, NaN, -30, NaN, -56, NaN));
    check_select(generator, 20, make_tensor<scalar_t>(make_dims(10), 1, 0, 1, 4, 9, 16, 25, 36, 49, 64));
    check_select_stats(generator, indices_t{}, indices_t{}, arange(0, 21), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 21),
        0,  0,  0,  0,  0,  0,   1,  0,  0,  0,  1,   0,  0,  0,  0,   0,  0,  0,   0,  0,   1,
        1,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,   0,  0,   0,
        4,  6,  4,  4,  4,  -2,  9,  6,  6,  6,  -3,  4,  4,  4,  -2,  4,  4,  -2,  4,  -2,  1,
        9,  0,  0,  0,  0,  -6,  0,  0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,   0,  0,   4,
        16, 20, 16, 16, 16, -12, 25, 20, 20, 20, -15, 16, 16, 16, -12, 16, 16, -12, 16, -12, 9,
        25, 0,  0,  0,  0,  -20, 0,  0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,   0,  0,   16,
        36, 42, 36, 36, 36, -30, 49, 42, 42, 42, -35, 36, 36, 36, -30, 36, 36, -30, 36, -30, 25,
        49, 0,  0,  0,  0,  -42, 0, 0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,   0,  0,   36,
        64, 72, 64, 64, 64, -56, 81, 72, 72, 72, -63, 64, 64, 64, -56, 64, 64, -56, 64, -56, 49,
        81, 0,  0,  0,  0,  -72, 0,  0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,   0,  0,   64),
        arange(0, 21));
}

UTEST_CASE(unsupervised_slog1p)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<slog1p_t>>(struct2scalar::off);
    generator.fit(arange(0, 10), execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 2);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"slog1p(f32[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"slog1p(f64[0])"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(10),
        std::log1p(0.0), std::log1p(1.0), std::log1p(2.0), std::log1p(3.0), std::log1p(4.0),
        std::log1p(5.0), std::log1p(6.0), std::log1p(7.0), std::log1p(8.0), std::log1p(9.0)));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(10),
        +std::log1p(1.0), +std::log1p(0.0), -std::log1p(1.0), -std::log1p(2.0), -std::log1p(3.0),
        -std::log1p(4.0), -std::log1p(5.0), -std::log1p(6.0), -std::log1p(7.0), -std::log1p(8.0)));
    check_select_stats(generator, indices_t{}, indices_t{}, make_indices(0, 1), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 2),
        std::log1p(0.0), +std::log1p(1.0),
        std::log1p(1.0), +std::log1p(0.0),
        std::log1p(2.0), -std::log1p(1.0),
        std::log1p(3.0), -std::log1p(2.0),
        std::log1p(4.0), -std::log1p(3.0),
        std::log1p(5.0), -std::log1p(4.0),
        std::log1p(6.0), -std::log1p(5.0),
        std::log1p(7.0), -std::log1p(6.0),
        std::log1p(8.0), -std::log1p(7.0),
        std::log1p(9.0), -std::log1p(8.0)),
        make_indices(0, 1));
}

UTEST_CASE(unsupervised_sign)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<sign_t>>(struct2scalar::on);
    generator.fit(arange(0, 10), execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 6);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sign(f32[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sign(u8s[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"sign(u8s[1])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"sign(u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"sign(u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"sign(f64[0])"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(10), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(10), 1, NaN, 1, NaN, 1, NaN, 1, NaN, 1, NaN));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(10), 1, NaN, 1, NaN, 1, NaN, 1, NaN, 1, NaN));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(10), 1, NaN, 1, NaN, 1, NaN, 1, NaN, 1, NaN));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(10), 1, NaN, 1, NaN, 1, NaN, 1, NaN, 1, NaN));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(10), 1, 1, -1, -1, -1, -1, -1, -1, -1, -1));
    check_select_stats(generator, indices_t{}, indices_t{}, make_indices(0, 1, 2, 3, 4, 5), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 6),
        1, 1, 1, 1, 1, +1,
        1, 0, 0, 0, 0, +1,
        1, 1, 1, 1, 1, -1,
        1, 0, 0, 0, 0, -1,
        1, 1, 1, 1, 1, -1,
        1, 0, 0, 0, 0, -1,
        1, 1, 1, 1, 1, -1,
        1, 0, 0, 0, 0, -1,
        1, 1, 1, 1, 1, -1,
        1, 0, 0, 0, 0, -1),
        make_indices(0, 1, 2, 3, 4, 5));
}

UTEST_CASE(unsupervised_sign_class)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<sign_class_t>>(struct2scalar::on);
    generator.fit(arange(0, 10), execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 6);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sign_class(f32[0])"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sign_class(u8s[0])"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"sign_class(u8s[1])"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"sign_class(u8s[2])"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"sign_class(u8s[3])"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"sign_class(f64[0])"}.sclass(strings_t{"neg", "pos"}));

    check_select(generator, 0, make_tensor<int32_t>(make_dims(10), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
    check_select(generator, 1, make_tensor<int32_t>(make_dims(10), 1, -1, 1, -1, 1, -1, 1, -1, 1, -1));
    check_select(generator, 2, make_tensor<int32_t>(make_dims(10), 1, -1, 1, -1, 1, -1, 1, -1, 1, -1));
    check_select(generator, 3, make_tensor<int32_t>(make_dims(10), 1, -1, 1, -1, 1, -1, 1, -1, 1, -1));
    check_select(generator, 4, make_tensor<int32_t>(make_dims(10), 1, -1, 1, -1, 1, -1, 1, -1, 1, -1));
    check_select(generator, 5, make_tensor<int32_t>(make_dims(10), 1, 1, 0, 0, 0, 0, 0, 0, 0, 0));
    check_select_stats(generator, make_indices(0, 1, 2, 3, 4, 5), indices_t{}, indices_t{}, indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 12),
        -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1,
        -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, -1, +1,
        -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1,
        -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +1, -1,
        -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1,
        -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +1, -1,
        -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1,
        -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +1, -1,
        -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1,
        -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +1, -1),
        make_indices(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5));
}

UTEST_END_MODULE()
