#pragma once

#include <nano/core/numeric.h>
#include <nano/function.h>

namespace nano
{
    ///
    /// \brief rotated hyper-ellipsoid function: see https://www.sfu.ca/~ssurjano/rothyp.html.
    ///
    class function_rotated_ellipsoid_t final : public function_t
    {
    public:

        explicit function_rotated_ellipsoid_t(const tensor_size_t dims) :
            function_t("Rotated Hyper-Ellipsoid", dims, convexity::yes)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            scalar_t fx = 0, fi = 0;
            for (tensor_size_t i = 0; i < size(); i ++)
            {
                fi += x(i);
                fx += nano::square(fi);
                if (gx != nullptr)
                {
                    (*gx)(i) = 2 * fi;
                }
            }

            if (gx != nullptr)
            {
                for (auto i = size() - 2; i >= 0; i --)
                {
                    (*gx)(i) += (*gx)(i + 1);
                }
            }

            return fx;
        }
    };
}
