#include <utest/utest.h>
#include "fixture/generator.h"
#include <nano/generator/pairwise_scalar2scalar.h>
#include <nano/generator/pairwise_scalar2sclass.h>

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

UTEST_BEGIN_MODULE(test_generator_pairwise_scalar)

UTEST_CASE(empty)
{
    const auto dataset = make_dataset(10, string_t::npos);
    const auto generator = dataset_generator_t{dataset};

    UTEST_CHECK_EQUAL(generator.columns(), 0);
    UTEST_CHECK_EQUAL(generator.features(), 0);
}

UTEST_CASE(unsupervised_product_scalar)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    UTEST_CHECK_NOTHROW(generator.add<pairwise_generator_t<pairwise_scalar2scalar_t<product_t>>>(struct2scalar::off));
    UTEST_CHECK_NOTHROW(generator.fit(arange(0, 10), execution::par));

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

UTEST_CASE(unsupervised_product_sclass)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    UTEST_CHECK_NOTHROW(generator.add<pairwise_generator_t<pairwise_scalar2sclass_t<product_sign_class_t>>>(struct2scalar::off));
    UTEST_CHECK_NOTHROW(generator.fit(arange(0, 10), execution::par));

    UTEST_REQUIRE_EQUAL(generator.features(), 3);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"product_sign_class(f32[0],f32[0])"}.sclass({"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"product_sign_class(f32[0],f64[0])"}.sclass({"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"product_sign_class(f64[0],f64[0])"}.sclass({"neg", "pos"}));

    check_select(generator, 0, make_tensor<int32_t>(make_dims(10), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
    check_select(generator, 1, make_tensor<int32_t>(make_dims(10), 1, 1, 0, 0, 0, 0, 0, 0, 0, 0));
    check_select(generator, 2, make_tensor<int32_t>(make_dims(10), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
    check_select_stats(generator, make_indices(0, 1, 2), indices_t{}, indices_t{}, indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 6),
        -1, +1, -1, +1, -1, +1,
        -1, +1, -1, +1, -1, +1,
        -1, +1, +1, -1, -1, +1,
        -1, +1, +1, -1, -1, +1,
        -1, +1, +1, -1, -1, +1,
        -1, +1, +1, -1, -1, +1,
        -1, +1, +1, -1, -1, +1,
        -1, +1, +1, -1, -1, +1,
        -1, +1, +1, -1, -1, +1,
        -1, +1, +1, -1, -1, +1),
        make_indices(0, 0, 1, 1, 2, 2));
}

UTEST_CASE(unsupervised_product_mixed)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    UTEST_CHECK_NOTHROW(generator.add<pairwise_generator_t<pairwise_scalar2scalar_t<product_t>>>(struct2scalar::on));
    UTEST_CHECK_NOTHROW(generator.fit(arange(0, 10), execution::par));

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

UTEST_END_MODULE()
