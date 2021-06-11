#include <utest/utest.h>
#include "fixture/generator.h"
#include <nano/generator/elemwise_gradient.h>

using namespace std;
using namespace nano;

UTEST_BEGIN_MODULE(test_generator_elemwise_gradient)

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

    const scalar_t kernel[] = {+0.25, +0.50, +0.25};

    auto output = tensor_mem_t<scalar_t, 2>(2, 2);
    {
        gradient3x3<gradient_mode::gradx>(input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), 2.00, 2.00, 1.50, 1.75);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-12);
    }
    {
        gradient3x3<gradient_mode::gradx>(input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), 1.00, 0.75, 0.50, 0.75);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-12);
    }
    {
        gradient3x3<gradient_mode::grady>(input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), 2.00, 2.00, 1.00, 0.25);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-12);
    }
    {
        gradient3x3<gradient_mode::grady>(input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), -0.50, -0.25, 0.00, -0.75);
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-12);
    }
    {
        gradient3x3<gradient_mode::magnitude>(input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2),
            sqrt(8.0), sqrt(8.0), sqrt(3.25), sqrt(3.125));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-12);
    }
    {
        gradient3x3<gradient_mode::magnitude>(input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2),
            sqrt(1.25), sqrt(0.625), sqrt(0.25), sqrt(1.125));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-12);
    }
    {
        gradient3x3<gradient_mode::angle>(input.tensor(), 0, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2),
            atan2(2.0, 2.0), atan2(2.0, 2.0), atan2(1.0, 1.5), atan2(0.25, 1.75));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-12);
    }
    {
        gradient3x3<gradient_mode::angle>(input.tensor(), 1, kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2),
            atan2(-0.5, 1.0), atan2(-0.25, 0.75), atan2(0.0, 0.5), atan2(-0.75, 0.75));
        UTEST_CHECK_TENSOR_CLOSE(output, expected_output, 1e-12);
    }
}

UTEST_END_MODULE()
