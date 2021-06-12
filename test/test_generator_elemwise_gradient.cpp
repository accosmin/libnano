#include <utest/utest.h>
#include "fixture/generator.h"
#include <nano/generator/elemwise_gradient.h>

using namespace std;
using namespace nano;

static auto make_features()
{
    return features_t
    {
        feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}),
        feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(4, 4, 2)),
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

        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            auto values = make_tensor<uint8_t>(make_dims(4, 4, 2),
                1, 0, 2, 1, 3, 1, 4, 1,
                2, 0, 3, 0, 4, 1, 5, 1,
                3, 0, 4, 0, 5, 1, 6, 1,
                4, 1, 4, 0, 4, 0, 5, 0
            );
            values.array() *= (sample + 1);
            set(sample, 3, values);
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

UTEST_BEGIN_MODULE(test_generator_elemwise_gradient)

UTEST_CASE(kernel)
{
    {
        const auto kernel = make_kernel3x3<double>(kernel3x3_type::sobel);
        UTEST_CHECK_CLOSE(kernel[0], 1.0/4.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[1], 2.0/4.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[2], 1.0/4.0, 1e-15);
    }
    {
        const auto kernel = make_kernel3x3<double>(kernel3x3_type::scharr);
        UTEST_CHECK_CLOSE(kernel[0], 3.0/16.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[1], 10.0/16.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[2], 3.0/16.0, 1e-15);
    }
    {
        const auto kernel = make_kernel3x3<double>(kernel3x3_type::prewitt);
        UTEST_CHECK_CLOSE(kernel[0], 1.0/3.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[1], 1.0/3.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[2], 1.0/3.0, 1e-15);
    }
    {
        const auto kernel = make_kernel3x3<double>(static_cast<kernel3x3_type>(-1));
        UTEST_CHECK(!std::isfinite(kernel[0]));
        UTEST_CHECK(!std::isfinite(kernel[1]));
        UTEST_CHECK(!std::isfinite(kernel[2]));
    }
}

UTEST_CASE(gradient)
{
    const auto input = make_tensor<int>
    (
        make_dims(4, 4, 2),
        1, 0, 2, 1, 3, 1, 4, 1,
        2, 0, 3, 0, 4, 1, 5, 1,
        3, 0, 4, 0, 5, 1, 6, 1,
        4, 1, 4, 0, 4, 0, 5, 0
    );

    const std::array<scalar_t, 3> kernel = {+0.25, +0.50, +0.25};

    auto output = tensor_mem_t<scalar_t, 2>(2, 2);
    {
        gradient3x3(gradient3x3_mode::gradx, input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), 2.00, 2.00, 1.50, 1.75);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::gradx, input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), 1.00, 0.75, 0.50, 0.75);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::grady, input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), 2.00, 2.00, 1.00, 0.25);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::grady, input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), -0.50, -0.25, 0.00, -0.75);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::magnitude, input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2),
            sqrt(8.0), sqrt(8.0), sqrt(3.25), sqrt(3.125));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::magnitude, input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2),
            sqrt(1.25), sqrt(0.625), sqrt(0.25), sqrt(1.125));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::angle, input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2),
            atan2(2.0, 2.0), atan2(2.0, 2.0), atan2(1.0, 1.5), atan2(0.25, 1.75));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::angle, input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2),
            atan2(-0.5, 1.0), atan2(-0.25, 0.75), atan2(0.0, 0.5), atan2(-0.75, 0.75));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
}

UTEST_CASE(unsupervised_gradient)
{
    const auto dataset = make_dataset(4, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<elemwise_gradient_t>>();
    generator.fit(arange(0, 4), execution::par);

    UTEST_REQUIRE_EQUAL(generator.features(), 8);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sobel::gx(u8s[channel::0])"}.scalar(feature_type::float64, make_dims(2, 2, 1)));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sobel::gy(u8s[channel::0])"}.scalar(feature_type::float64, make_dims(2, 2, 1)));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"sobel::gg(u8s[channel::0])"}.scalar(feature_type::float64, make_dims(2, 2, 1)));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"sobel::theta(u8s[channel::0])"}.scalar(feature_type::float64, make_dims(2, 2, 1)));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"sobel::gx(u8s[channel::1])"}.scalar(feature_type::float64, make_dims(2, 2, 1)));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"sobel::gy(u8s[channel::1])"}.scalar(feature_type::float64, make_dims(2, 2, 1)));
    UTEST_CHECK_EQUAL(generator.feature(6), feature_t{"sobel::gg(u8s[channel::1])"}.scalar(feature_type::float64, make_dims(2, 2, 1)));
    UTEST_CHECK_EQUAL(generator.feature(7), feature_t{"sobel::theta(u8s[channel::1])"}.scalar(feature_type::float64, make_dims(2, 2, 1)));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        2.0, 2.0, 1.5, 1.75, NaN, NaN, NaN, NaN,
        6.0, 6.0, 4.5, 5.25, NaN, NaN, NaN, NaN));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        2.0, 2.0, 1.0, 0.25, NaN, NaN, NaN, NaN,
        6.0, 6.0, 3.0, 0.75, NaN, NaN, NaN, NaN));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        1*sqrt(8.0), 1*sqrt(8.0), 1*sqrt(3.25), 1*sqrt(3.125), NaN, NaN, NaN, NaN,
        3*sqrt(8.0), 3*sqrt(8.0), 3*sqrt(3.25), 3*sqrt(3.125), NaN, NaN, NaN, NaN));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        atan2(2.0, 2.0), atan2(2.0, 2.0), atan2(1.0, 1.5), atan2(0.25, 1.75), NaN, NaN, NaN, NaN,
        atan2(2.0, 2.0), atan2(2.0, 2.0), atan2(1.0, 1.5), atan2(0.25, 1.75), NaN, NaN, NaN, NaN));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        1.00, 0.75, 0.50, 0.75, NaN, NaN, NaN, NaN,
        3.00, 2.25, 1.50, 2.25, NaN, NaN, NaN, NaN));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        -0.50, -0.25, 0.00, -0.75, NaN, NaN, NaN, NaN,
        -1.50, -0.75, 0.00, -2.25, NaN, NaN, NaN, NaN));
    check_select(generator, 6, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        1*sqrt(1.25), 1*sqrt(0.625), 1*sqrt(0.25), 1*sqrt(1.125), NaN, NaN, NaN, NaN,
        3*sqrt(1.25), 3*sqrt(0.625), 3*sqrt(0.25), 3*sqrt(1.125), NaN, NaN, NaN, NaN));
    check_select(generator, 7, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        atan2(-0.5, 1.0), atan2(-0.25, 0.75), atan2(0.0, 0.5), atan2(-0.75, 0.75), NaN, NaN, NaN, NaN,
        atan2(-0.5, 1.0), atan2(-0.25, 0.75), atan2(0.0, 0.5), atan2(-0.75, 0.75), NaN, NaN, NaN, NaN));
    check_select_stats(generator, indices_t{}, indices_t{}, indices_t{}, make_indices(0, 1, 2, 3, 4, 5, 6, 7));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(4, 32),
        2.0, 2.0, 1.5, 1.75,
        2.0, 2.0, 1.0, 0.25,
        1*sqrt(8.0), 1*sqrt(8.0), 1*sqrt(3.25), 1*sqrt(3.125),
        atan2(2.0, 2.0), atan2(2.0, 2.0), atan2(1.0, 1.5), atan2(0.25, 1.75),

        1.00, 0.75, 0.50, 0.75,
        -0.50, -0.25, 0.00, -0.75,
        1*sqrt(1.25), 1*sqrt(0.625), 1*sqrt(0.25), 1*sqrt(1.125),
        atan2(-0.5, 1.0), atan2(-0.25, 0.75), atan2(0.0, 0.5), atan2(-0.75, 0.75),

        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        6.0, 6.0, 4.5, 5.25,
        6.0, 6.0, 3.0, 0.75,
        3*sqrt(8.0), 3*sqrt(8.0), 3*sqrt(3.25), 3*sqrt(3.125),
        atan2(2.0, 2.0), atan2(2.0, 2.0), atan2(1.0, 1.5), atan2(0.25, 1.75),

        3.00, 2.25, 1.50, 2.25,
        -1.50, -0.75, 0.00, -2.25,
        3*sqrt(1.25), 3*sqrt(0.625), 3*sqrt(0.25), 3*sqrt(1.125),
        atan2(-0.5, 1.0), atan2(-0.25, 0.75), atan2(0.0, 0.5), atan2(-0.75, 0.75),

        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        make_indices(
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7));
}

UTEST_END_MODULE()
