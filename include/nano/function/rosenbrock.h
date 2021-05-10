#pragma once

#include <nano/core/numeric.h>
#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Rosenbrock function: see https://en.wikipedia.org/wiki/Test_functions_for_optimization.
    ///
    class function_rosenbrock_t final : public function_t
    {
    public:

        explicit function_rosenbrock_t(const tensor_size_t dims) :
            function_t("Rosenbrock", dims, convexity::no)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            const auto ct = scalar_t(100);

            scalar_t fx = 0;
            for (tensor_size_t i = 0; i + 1 < size(); ++ i)
            {
                fx += ct * nano::square(x(i + 1) - x(i) * x(i)) + nano::square(x(i) - 1);
            }

            if (gx != nullptr)
            {
                (*gx).setZero();
                for (tensor_size_t i = 0; i + 1 < size(); ++ i)
                {
                    (*gx)(i) += 2 * (x(i) - 1);
                    (*gx)(i) += ct * 2 * (x(i + 1) - x(i) * x(i)) * (- 2 * x(i));
                    (*gx)(i + 1) += ct * 2 * (x(i + 1) - x(i) * x(i));
                }
            }

            return fx;
        }
    };
}
