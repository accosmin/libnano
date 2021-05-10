#pragma once

#include <nano/core/numeric.h>
#include <nano/function.h>

namespace nano
{
    ///
    /// \brief Dixon-Price function: see https://www.sfu.ca/~ssurjano/dixonpr.html.
    ///
    class function_dixon_price_t final : public function_t
    {
    public:

        explicit function_dixon_price_t(const tensor_size_t dims) :
            function_t("Dixon-Price", dims, convexity::no),
            m_bias(vector_t::LinSpaced(dims, scalar_t(1), scalar_t(dims)))
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
            const auto xsegm0 = x.segment(0, size() - 1);
            const auto xsegm1 = x.segment(1, size() - 1);

            if (gx != nullptr)
            {
                const auto weight = m_bias.segment(1, size() - 1).array() *
                    2 * (2 * xsegm1.array().square() - xsegm0.array());

                (*gx).setZero();
                (*gx)(0) = 2 * (x(0) - 1);
                (*gx).segment(1, size() - 1).array() += weight * 4 * xsegm1.array();
                (*gx).segment(0, size() - 1).array() -= weight;
            }

            return  nano::square(x(0) - 1) +
                (m_bias.segment(1, size() - 1).array() *
                (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
        }

    private:

        // attributes
        vector_t    m_bias; ///<
    };
}
