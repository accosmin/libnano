#include <utest/utest.h>
#include "fixture/generator.h"
#include <nano/generator/elemwise_scalar2scalar.h>
#include <nano/generator/elemwise_scalar2sclass.h>

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

UTEST_BEGIN_MODULE(test_generator_elemwise_scalar)

UTEST_CASE(empty)
{
    const auto dataset = make_dataset(10, string_t::npos);
    const auto generator = dataset_generator_t{dataset};

    UTEST_CHECK_EQUAL(generator.columns(), 0);
    UTEST_CHECK_EQUAL(generator.features(), 0);
}

UTEST_CASE(unsupervised_slog1p)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<elemwise_scalar2scalar_t<slog1p_t>>>(struct2scalar::off);
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
    generator.add<elemwise_generator_t<elemwise_scalar2scalar_t<sign_t>>>(struct2scalar::on);
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
    generator.add<elemwise_generator_t<elemwise_scalar2sclass_t<sign_class_t>>>(struct2scalar::on);
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
