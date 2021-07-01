#include "fixture/generator.h"
#include <nano/generator/elemwise_histogram.h>

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

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 2, sample % 10);
        }

        tensor_mem_t<tensor_size_t, 3> values(2, 1, 2);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            values(0) = sample % 10;
            values(1) = 9 - sample % 10;
            values(2) = (sample + 3) % 10;
            values(3) = (sample - 1) % 10;
            set(sample, 3, values);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 4, 9 - sample % 10);
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

UTEST_BEGIN_MODULE(test_generator_elemwise_histogram)

UTEST_CASE(unsupervised_ratios)
{
    const auto bins = 10;
    const auto dataset = make_dataset(30, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<ratio_histogram_medians_t>>(struct2scalar::off, bins);
    generator.fit(arange(0, 30), execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 2);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"ratio_hist[10](f32[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"ratio_hist[10](f64[0])"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(30),
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(30),
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
    check_select_stats(generator, indices_t{}, indices_t{}, make_indices(0, 1), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(30, 2),
        0, 9, 1, 8, 2, 7, 3, 6, 4, 5, 5, 4, 6, 3, 7, 2, 8, 1, 9, 0,
        0, 9, 1, 8, 2, 7, 3, 6, 4, 5, 5, 4, 6, 3, 7, 2, 8, 1, 9, 0,
        0, 9, 1, 8, 2, 7, 3, 6, 4, 5, 5, 4, 6, 3, 7, 2, 8, 1, 9, 0),
        make_indices(0, 1));
}

UTEST_CASE(unsupervised_percentiles)
{
    const auto bins = 5;
    const auto dataset = make_dataset(30, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<percentile_histogram_medians_t>>(struct2scalar::on, bins);
    generator.fit(arange(0, 30), execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 6);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"perc_hist[5](f32[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"perc_hist[5](u8s[0])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"perc_hist[5](u8s[1])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"perc_hist[5](u8s[2])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"perc_hist[5](u8s[3])"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"perc_hist[5](f64[0])"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(30),
        0.5, 0.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5, 8.5, 8.5,
        0.5, 0.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5, 8.5, 8.5,
        0.5, 0.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5, 8.5, 8.5));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(30),
        0.0, NaN, 2.0, NaN, 4.0, NaN, 6.0, NaN, 8.0, NaN,
        0.0, NaN, 2.0, NaN, 4.0, NaN, 6.0, NaN, 8.0, NaN,
        0.0, NaN, 2.0, NaN, 4.0, NaN, 6.0, NaN, 8.0, NaN));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(30),
        9.0, NaN, 7.0, NaN, 5.0, NaN, 3.0, NaN, 1.0, NaN,
        9.0, NaN, 7.0, NaN, 5.0, NaN, 3.0, NaN, 1.0, NaN,
        9.0, NaN, 7.0, NaN, 5.0, NaN, 3.0, NaN, 1.0, NaN));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(30),
        3.0, NaN, 5.0, NaN, 7.0, NaN, 9.0, NaN, 1.0, NaN,
        3.0, NaN, 5.0, NaN, 7.0, NaN, 9.0, NaN, 1.0, NaN,
        3.0, NaN, 5.0, NaN, 7.0, NaN, 9.0, NaN, 1.0, NaN));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(30),
        9.0, NaN, 1.0, NaN, 3.0, NaN, 5.0, NaN, 7.0, NaN,
        9.0, NaN, 1.0, NaN, 3.0, NaN, 5.0, NaN, 7.0, NaN,
        9.0, NaN, 1.0, NaN, 3.0, NaN, 5.0, NaN, 7.0, NaN));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(30),
        8.5, 8.5, 6.5, 6.5, 4.5, 4.5, 2.5, 2.5, 0.5, 0.5,
        8.5, 8.5, 6.5, 6.5, 4.5, 4.5, 2.5, 2.5, 0.5, 0.5,
        8.5, 8.5, 6.5, 6.5, 4.5, 4.5, 2.5, 2.5, 0.5, 0.5));
    check_select_stats(generator, indices_t{}, indices_t{}, make_indices(0, 1, 2, 3, 4, 5), indices_t{});

    check_flatten(generator, make_tensor<scalar_t>(make_dims(30, 6),
        0.5, 0.0, 9.0, 3.0, 9.0, 8.5, 0.5, 0.0, 0.0, 0.0, 0.0, 8.5,
        2.5, 2.0, 7.0, 5.0, 1.0, 6.5, 2.5, 0.0, 0.0, 0.0, 0.0, 6.5,
        4.5, 4.0, 5.0, 7.0, 3.0, 4.5, 4.5, 0.0, 0.0, 0.0, 0.0, 4.5,
        6.5, 6.0, 3.0, 9.0, 5.0, 2.5, 6.5, 0.0, 0.0, 0.0, 0.0, 2.5,
        8.5, 8.0, 1.0, 1.0, 7.0, 0.5, 8.5, 0.0, 0.0, 0.0, 0.0, 0.5,
        0.5, 0.0, 9.0, 3.0, 9.0, 8.5, 0.5, 0.0, 0.0, 0.0, 0.0, 8.5,
        2.5, 2.0, 7.0, 5.0, 1.0, 6.5, 2.5, 0.0, 0.0, 0.0, 0.0, 6.5,
        4.5, 4.0, 5.0, 7.0, 3.0, 4.5, 4.5, 0.0, 0.0, 0.0, 0.0, 4.5,
        6.5, 6.0, 3.0, 9.0, 5.0, 2.5, 6.5, 0.0, 0.0, 0.0, 0.0, 2.5,
        8.5, 8.0, 1.0, 1.0, 7.0, 0.5, 8.5, 0.0, 0.0, 0.0, 0.0, 0.5,
        0.5, 0.0, 9.0, 3.0, 9.0, 8.5, 0.5, 0.0, 0.0, 0.0, 0.0, 8.5,
        2.5, 2.0, 7.0, 5.0, 1.0, 6.5, 2.5, 0.0, 0.0, 0.0, 0.0, 6.5,
        4.5, 4.0, 5.0, 7.0, 3.0, 4.5, 4.5, 0.0, 0.0, 0.0, 0.0, 4.5,
        6.5, 6.0, 3.0, 9.0, 5.0, 2.5, 6.5, 0.0, 0.0, 0.0, 0.0, 2.5,
        8.5, 8.0, 1.0, 1.0, 7.0, 0.5, 8.5, 0.0, 0.0, 0.0, 0.0, 0.5),
        make_indices(0, 1, 2, 3, 4, 5));
}

UTEST_END_MODULE()
