#include <utest/utest.h>
#include "fixture/generator.h"
#include <nano/generator/elemwise_gradient.h>

using namespace std;
using namespace nano;

#define INPUT_DATA \
    make_dims(4, 4, 2), \
    1, 0, 2, 1, 3, 1, 4, 1, \
    2, 0, 3, 0, 4, 1, 5, 1, \
    3, 0, 4, 0, 5, 1, 6, 1, \
    4, 1, 4, 0, 4, 0, 5, 0

#define GX0(scale) (scale) * 2.00, (scale) * 2.00, (scale) * 1.50, (scale) * 1.75
#define GX1(scale) (scale) * 1.00, (scale) * 0.75, (scale) * 0.50, (scale) * 0.75

#define GY0(scale) (scale) * +2.00, (scale) * +2.00, (scale) * 1.00, (scale) * +0.25
#define GY1(scale) (scale) * -0.50, (scale) * -0.25, (scale) * 0.00, (scale) * -0.75

#define GG0(scale) (scale) * sqrt(8.00), (scale) * sqrt(8.000), (scale) * sqrt(3.25), (scale) * sqrt(3.125)
#define GG1(scale) (scale) * sqrt(1.25), (scale) * sqrt(0.625), (scale) * sqrt(0.25), (scale) * sqrt(1.125)

#define THETA0 atan2(+2.0, 2.0), atan2(+2.00, 2.00), atan2(1.0, 1.5), atan2(+0.25, 1.75)
#define THETA1 atan2(-0.5, 1.0), atan2(-0.25, 0.75), atan2(0.0, 0.5), atan2(-0.75, 0.75)

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
            auto values = make_tensor<uint8_t>(
                INPUT_DATA
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
        INPUT_DATA
    );

    const std::array<scalar_t, 3> kernel = {+0.25, +0.50, +0.25};

    auto output = tensor_mem_t<scalar_t, 2>(2, 2);
    {
        gradient3x3(gradient3x3_mode::gradx, input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GX0(1));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::gradx, input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GX1(1));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::grady, input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GY0(1));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::grady, input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GY1(1));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::magnitude, input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GG0(1));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::magnitude, input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GG1(1));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::angle, input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), THETA0);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::angle, input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), THETA1);
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
        GX0(1), NaN, NaN, NaN, NaN,
        GX0(3), NaN, NaN, NaN, NaN));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        GY0(1), NaN, NaN, NaN, NaN,
        GY0(3), NaN, NaN, NaN, NaN));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        GG0(1), NaN, NaN, NaN, NaN,
        GG0(3), NaN, NaN, NaN, NaN));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        THETA0, NaN, NaN, NaN, NaN,
        THETA0, NaN, NaN, NaN, NaN));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        GX1(1), NaN, NaN, NaN, NaN,
        GX1(3), NaN, NaN, NaN, NaN));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        GY1(1), NaN, NaN, NaN, NaN,
        GY1(3), NaN, NaN, NaN, NaN));
    check_select(generator, 6, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        GG1(1), NaN, NaN, NaN, NaN,
        GG1(3), NaN, NaN, NaN, NaN));
    check_select(generator, 7, make_tensor<scalar_t>(make_dims(4, 2, 2, 1),
        THETA1, NaN, NaN, NaN, NaN,
        THETA1, NaN, NaN, NaN, NaN));
    check_select_stats(generator, indices_t{}, indices_t{}, indices_t{}, make_indices(0, 1, 2, 3, 4, 5, 6, 7));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(4, 32),
        GX0(1), GY0(1), GG0(1), THETA0, GX1(1), GY1(1), GG1(1), THETA1,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        GX0(3), GY0(3), GG0(3), THETA0, GX1(3), GY1(3), GG1(3), THETA1,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        make_indices(
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7));
}

UTEST_END_MODULE()
