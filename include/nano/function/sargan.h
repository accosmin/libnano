#pragma once

#include <nano/core/numeric.h>
#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Sargan function: see http://infinity77.net/global_optimization/test_functions_nd_S.html.
    ///
    class function_sargan_t final : public function_t
    {
    public:

        explicit function_sargan_t(const tensor_size_t dims) :
            function_t("Sargan", dims, convexity::yes)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            const auto x2sum = x.dot(x);

            if (gx != nullptr)
            {
                *gx = (scalar_t(1.2) + scalar_t(1.6) * x2sum) * x;
            }

            return scalar_t(0.6) * x2sum + scalar_t(0.4) * nano::square(x2sum);
        }
    };
}
