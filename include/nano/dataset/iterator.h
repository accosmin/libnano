#pragma once

#include <nano/dataset/mask.h>

namespace nano
{
    ///
    /// \brief
    ///
    class base_dataset_iterator_t
    {
    public:

        base_dataset_iterator_t() = default;

        base_dataset_iterator_t(indices_cmap_t samples, tensor_size_t index = 0) :
            m_index(index),
            m_samples(samples)
        {
            assert(index >= 0 && index <= m_samples.size());
        }

        tensor_size_t sample() const
        {
            assert(m_index >= 0 && m_index < m_samples.size());

            return m_samples(m_index);
        }

        tensor_size_t index() const { return m_index; }
        tensor_size_t size() const { return m_samples.size(); }

        base_dataset_iterator_t& operator++()
        {
            assert(m_index < m_samples.size());

            ++ m_index;
            return *this;
        }

        base_dataset_iterator_t operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        operator bool() const
        {
            return index() < size();
        }

    private:

        // attributes
        tensor_size_t       m_index{0}; ///<
        indices_cmap_t      m_samples;  ///<
    };

    ///
    /// \brief utility to iterate over the masked feature values of a given set of samples.
    ///
    template <typename tscalar, size_t trank>
    class dataset_iterator_t : public base_dataset_iterator_t
    {
    public:

        using data_cmap_t = tensor_cmap_t<tscalar, trank>;

        dataset_iterator_t() = default;

        dataset_iterator_t(data_cmap_t data, mask_cmap_t mask, indices_cmap_t samples, tensor_size_t index = 0) :
            base_dataset_iterator_t(samples, index),
            m_data(data),
            m_mask(mask)
        {
        }

        auto operator*() const
        {
            const auto sample = this->sample();
            const auto given = getbit(m_mask, sample);

            if constexpr(trank == 1)
            {
                return std::make_tuple(index(), given, m_data(sample));
            }
            else
            {
                return std::make_tuple(index(), given, m_data.tensor(sample));
            }
        }

    private:

        // attributes
        data_cmap_t         m_data;     ///<
        mask_cmap_t         m_mask;     ///<
    };

    ///
    /// \brief utility to iterate over the masked feature values of a given set of samples.
    ///
    template <typename tscalar1, size_t trank1, typename tscalar2, size_t trank2>
    class dataset_pairwise_iterator_t : public base_dataset_iterator_t
    {
    public:

        using data1_cmap_t = tensor_cmap_t<tscalar1, trank1>;
        using data2_cmap_t = tensor_cmap_t<tscalar2, trank2>;

        dataset_pairwise_iterator_t() = default;

        dataset_pairwise_iterator_t(
            data1_cmap_t data1, mask_cmap_t mask1,
            data2_cmap_t data2, mask_cmap_t mask2,
            indices_cmap_t samples, tensor_size_t index = 0) :
            base_dataset_iterator_t(samples, index),
            m_data1(data1),
            m_mask1(mask1),
            m_data2(data2),
            m_mask2(mask2)
        {
        }

        auto operator*() const
        {
            const auto sample = this->sample();
            const auto given1 = getbit(m_mask1, sample);
            const auto given2 = getbit(m_mask2, sample);

            if constexpr(trank1 == 1)
            {
                if constexpr(trank2 == 1)
                {
                    return std::make_tuple(index(), given1, m_data1(sample), given2, m_data2(sample));
                }
                else
                {
                    return std::make_tuple(index(), given1, m_data1(sample), given2, m_data2.tensor(sample));
                }
            }
            else
            {
                if constexpr(trank2 == 1)
                {
                    return std::make_tuple(index(), given1, m_data1.tensor(sample), given2, m_data2(sample));
                }
                else
                {
                    return std::make_tuple(index(), given1, m_data1.tensor(sample), given2, m_data2.tensor(sample));
                }
            }
        }

    private:

        // attributes
        data1_cmap_t        m_data1;    ///<
        mask_cmap_t         m_mask1;    ///<
        data2_cmap_t        m_data2;    ///<
        mask_cmap_t         m_mask2;    ///<
    };

    ///
    /// \brief return true if the two iterators are equivalent.
    ///
    inline bool operator!=(const base_dataset_iterator_t& lhs, const base_dataset_iterator_t& rhs)
    {
        assert(lhs.size() == rhs.size());
        return lhs.index() != rhs.index();
    }

    ///
    /// \brief construct an iterator from the given inputs.
    ///
    template
    <
        template <typename, size_t> class tstorage, typename tscalar, size_t trank
    >
    auto make_iterator(
        const tensor_t<tstorage, tscalar, trank>& data, mask_cmap_t mask,
        indices_cmap_t samples)
    {
        return dataset_iterator_t<tscalar, trank>{data, mask, samples, 0};
    }

    template
    <
        template <typename, size_t> class tstorage1, typename tscalar1, size_t trank1,
        template <typename, size_t> class tstorage2, typename tscalar2, size_t trank2
    >
    auto make_iterator(
        const tensor_t<tstorage1, tscalar1, trank1>& data1, mask_cmap_t mask1,
        const tensor_t<tstorage2, tscalar2, trank2>& data2, mask_cmap_t mask2,
        indices_cmap_t samples)
    {
        return dataset_pairwise_iterator_t<tscalar1, trank1, tscalar2, trank2>{data1, mask1, data2, mask2, samples, 0};
    }

    ///
    /// \brief construct an invalid (end) iterator from the given inputs.
    ///
    inline auto make_end_iterator(indices_cmap_t samples)
    {
        return base_dataset_iterator_t{samples, samples.size()};
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

    template
    <
        size_t trank_expected,
        template <typename, size_t> class tstorage1, typename tscalar1, size_t trank1,
        template <typename, size_t> class tstorage2, typename tscalar2, size_t trank2,
        typename toperator_expected,
        typename toperator_otherwise
    >
    auto loop_samples(
        const tensor_t<tstorage1, tscalar1, trank1>& data1, const mask_cmap_t& mask1,
        const tensor_t<tstorage2, tscalar2, trank2>& data2, const mask_cmap_t& mask2,
        const indices_cmap_t& samples,
        const toperator_expected& op_expected,
        const toperator_otherwise& op_otherwise)
    {
        if constexpr (trank1 == trank_expected && trank2 == trank_expected)
        {
            return op_expected(make_iterator(data1, mask1, data2, mask2, samples));
        }
        else
        {
            return op_otherwise();
        }
    }
}
