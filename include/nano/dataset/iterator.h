#pragma once

#include <nano/dataset/mask.h>

namespace nano
{
    ///
    /// \brief utility to iterate over the masked feature values of a given set of samples.
    ///
    template <typename tscalar, size_t trank>
    class dataset_iterator_t
    {
    public:

        using data_cmap_t = tensor_cmap_t<tscalar, trank>;

        dataset_iterator_t() = default;

        dataset_iterator_t(data_cmap_t data, mask_cmap_t mask, indices_cmap_t samples, tensor_size_t index = 0) :
            m_index(index),
            m_data(data),
            m_mask(mask),
            m_samples(samples)
        {
            assert(index >= 0 && index <= m_samples.size());
        }

        tensor_size_t index() const { return m_index; }
        tensor_size_t size() const { return m_samples.size(); }

        dataset_iterator_t& operator++()
        {
            assert(m_index < m_samples.size());

            ++ m_index;
            return *this;
        }

        dataset_iterator_t operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        operator bool() const
        {
            return index() < size();
        }

        auto operator*() const
        {
            assert(m_index < m_samples.size());

            const auto sample = m_samples(m_index);
            const auto given = getbit(m_mask, sample);

            if constexpr(trank == 1)
            {
                return std::make_tuple(m_index, given, m_data(sample));
            }
            else
            {
                return std::make_tuple(m_index, given, m_data.tensor(sample));
            }
        }

    private:

        // attributes
        tensor_size_t       m_index{0}; ///<
        data_cmap_t         m_data;     ///<
        mask_cmap_t         m_mask;     ///<
        indices_cmap_t      m_samples;  ///<
    };

    ///
    /// \brief return true if the two iterators are equivalent.
    ///
    template <typename tscalar, size_t trank>
    bool operator!=(const dataset_iterator_t<tscalar, trank>& lhs, const dataset_iterator_t<tscalar, trank>& rhs)
    {
        assert(lhs.size() == rhs.size());
        return lhs.index() != rhs.index();
    }

    ///
    /// \brief construct an iterator from the given inputs.
    ///
    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    auto make_iterator(const tensor_t<tstorage, tscalar, trank>& data, mask_cmap_t mask, indices_cmap_t samples)
    {
        return dataset_iterator_t<tscalar, trank>{data, mask, samples, 0};
    }

    ///
    /// \brief construct an invalid (end) iterator from the given inputs.
    ///
    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    auto make_end_iterator(const tensor_t<tstorage, tscalar, trank>& data, mask_cmap_t mask, indices_cmap_t samples)
    {
        return dataset_iterator_t<tscalar, trank>{data, mask, samples, samples.size()};
    }

    ///
    /// \brief call the appropriate operator for the given data,
    ///     distinguishing between single-label, multi-label and scalar/structured cases.
    ///
    template
    <
        template <typename, size_t> class tstorage, typename tscalar, size_t trank,
        typename toperator_sclass,
        typename toperator_mclass,
        typename toperator_scalar
    >
    auto loop_samples(
        const tensor_t<tstorage, tscalar, trank>& data, const mask_cmap_t& mask, const indices_cmap_t& samples,
        const toperator_sclass& op_sclass,
        const toperator_mclass& op_mclass,
        const toperator_scalar& op_scalar)
    {
        if constexpr (trank == 1)
        {
            return op_sclass(make_iterator(data, mask, samples));
        }
        else if constexpr (trank == 2)
        {
            return op_mclass(make_iterator(data, mask, samples));
        }
        else
        {
            return op_scalar(make_iterator(data, mask, samples));
        }
    }

    template
    <
        size_t trank_expected,
        template <typename, size_t> class tstorage, typename tscalar, size_t trank,
        typename toperator_expected,
        typename toperator_otherwise
    >
    auto loop_samples(
        const tensor_t<tstorage, tscalar, trank>& data, const mask_cmap_t& mask, const indices_cmap_t& samples,
        const toperator_expected& op_expected,
        const toperator_otherwise& op_otherwise)
    {
        if constexpr (trank == trank_expected)
        {
            return op_expected(make_iterator(data, mask, samples));
        }
        else
        {
            return op_otherwise();
        }
    }
}
