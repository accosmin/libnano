#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
    enum class gradient_mode
    {
        gradx, grady, magnitude, angle
    };

    ///
    /// \brief compute the gradients, the edge magnitude and the edge orientation
    ///     in a 2D image using a symetric 3x3 kernel.
    ///
    template <gradient_mode mode, typename tscalar_input, typename tscalar_output>
    void gradient3x3(
        tensor_cmap_t<tscalar_input, 3> input, tensor_size_t channel, const tscalar_output kernel[3],
        tensor_map_t<tscalar_output, 2> output)
    {
        const auto rows = output.template size<0>();
        const auto cols = output.template size<1>();

        assert(input.template size<0>() == rows + 2);
        assert(input.template size<1>() == cols + 2);
        assert(input.template size<2>() > channel && channel >= 0);

        for (tensor_size_t row = 0; row < rows; ++ row)
        {
            for (tensor_size_t col = 0; col < cols; ++ col)
            {
                if constexpr(mode == gradient_mode::gradx)
                {
                    const auto gx =
                        input(row + 0, col + 2, channel) * kernel[0] - input(row + 0, col, channel) * kernel[0] +
                        input(row + 1, col + 2, channel) * kernel[1] - input(row + 1, col, channel) * kernel[1] +
                        input(row + 2, col + 2, channel) * kernel[2] - input(row + 2, col, channel) * kernel[2];

                    output(row, col) = gx;
                }
                else if constexpr (mode == gradient_mode::grady)
                {
                    const auto gy =
                        input(row + 2, col + 0, channel) * kernel[0] - input(row, col + 0, channel) * kernel[0] +
                        input(row + 2, col + 1, channel) * kernel[1] - input(row, col + 1, channel) * kernel[1] +
                        input(row + 2, col + 2, channel) * kernel[2] - input(row, col + 2, channel) * kernel[2];

                    output(row, col) = gy;
                }
                else
                {
                    const auto gx =
                        input(row + 0, col + 2, channel) * kernel[0] - input(row + 0, col, channel) * kernel[0] +
                        input(row + 1, col + 2, channel) * kernel[1] - input(row + 1, col, channel) * kernel[1] +
                        input(row + 2, col + 2, channel) * kernel[2] - input(row + 2, col, channel) * kernel[2];

                    const auto gy =
                        input(row + 2, col + 0, channel) * kernel[0] - input(row, col + 0, channel) * kernel[0] +
                        input(row + 2, col + 1, channel) * kernel[1] - input(row, col + 1, channel) * kernel[1] +
                        input(row + 2, col + 2, channel) * kernel[2] - input(row, col + 2, channel) * kernel[2];

                    if constexpr (mode == gradient_mode::magnitude)
                    {
                        output(row, col) = std::sqrt(gx * gx + gy * gy);
                    }
                    else
                    {
                        output(row, col) = std::atan2(gy, gx);
                    }
                }
            }
        }
    }
}
