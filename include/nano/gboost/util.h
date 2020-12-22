#pragma once

#include <algorithm>
#include <nano/tensor.h>

namespace nano { namespace gboost
{
    ///
    /// \brief min-reduce the given set of per-thread caches using the `min_score` attribute.
    ///
    template <typename tcache>
    const auto& min_reduce(const std::vector<tcache>& caches)
    {
        const auto op = [] (const tcache& one, const tcache& other) { return one.m_score < other.m_score; };
        const auto it = std::min_element(caches.begin(), caches.end(), op);
        return *it;
    }

    ///
    /// \brief map-reduce the given set of per-thread caches into the first cache.
    ///
    template <typename tcache>
    const auto& sum_reduce(std::vector<tcache>& caches, const tensor_size_t samples)
    {
        auto& cache0 = caches[0];
        for (size_t i = 1; i < caches.size(); ++ i)
        {
            cache0 += caches[i];
        }
        return (cache0 /= samples);
    }

    ///
    /// \brief accumulates residuals & feature values of different moment orders
    ///     useful for fitting simple weak learners.
    ///
    class accumulator_t
    {
    public:

        explicit accumulator_t(const tensor3d_dims_t& tdims = make_dims(0, 0, 0)) :
            m_x0(1),
            m_x1(1),
            m_x2(1),
            m_r1(cat_dims(1, tdims)),
            m_rx(cat_dims(1, tdims)),
            m_r2(cat_dims(1, tdims))
        {
            clear();
        }

        auto fvalues() const { return m_r1.size<0>(); }
        auto tdims() const { return make_dims(m_r1.size<1>(), m_r1.size<2>(), m_r1.size<3>()); }

        auto& x0(tensor_size_t fv = 0) { return m_x0(fv); }
        auto& x1(tensor_size_t fv = 0) { return m_x1(fv); }
        auto& x2(tensor_size_t fv = 0) { return m_x2(fv); }
        auto r1(tensor_size_t fv = 0) { return m_r1.array(fv); }
        auto rx(tensor_size_t fv = 0) { return m_rx.array(fv); }
        auto r2(tensor_size_t fv = 0) { return m_r2.array(fv); }

        auto x0(tensor_size_t fv = 0) const { return m_x0(fv); }
        auto x1(tensor_size_t fv = 0) const { return m_x1(fv); }
        auto x2(tensor_size_t fv = 0) const { return m_x2(fv); }
        auto r1(tensor_size_t fv = 0) const { return m_r1.array(fv); }
        auto rx(tensor_size_t fv = 0) const { return m_rx.array(fv); }
        auto r2(tensor_size_t fv = 0) const { return m_r2.array(fv); }

        void clear()
        {
            m_x0.zero();
            m_x1.zero();
            m_x2.zero();
            m_r1.zero();
            m_rx.zero();
            m_r2.zero();
        }

        void clear(tensor_size_t fvalues)
        {
            m_x0.resize(fvalues);
            m_x1.resize(fvalues);
            m_x2.resize(fvalues);
            m_r1.resize(cat_dims(fvalues, tdims()));
            m_rx.resize(cat_dims(fvalues, tdims()));
            m_r2.resize(cat_dims(fvalues, tdims()));

            clear();
        }

        template <typename tarray>
        void update(tarray&& vgrad, tensor_size_t fv = 0)
        {
            x0(fv) += 1;
            r1(fv) -= vgrad;
            r2(fv) += vgrad * vgrad;
        }

        template <typename tarray>
        void update(scalar_t value, tarray&& vgrad, tensor_size_t fv = 0)
        {
            update(vgrad, fv);
            x1(fv) += value;
            x2(fv) += value * value;
            rx(fv) -= vgrad * value;
        }

    private:

        // attributes
        tensor1d_t  m_x0, m_x1, m_x2;       ///<
        tensor4d_t  m_r1, m_rx, m_r2;       ///<
    };
}}
