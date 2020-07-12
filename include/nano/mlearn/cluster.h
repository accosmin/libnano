#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief efficient assignment of samples to a fixed number of groups:
    ///     - not assigned, if -1
    ///     - group index, otherwise.
    ///
    class NANO_PUBLIC cluster_t
    {
    public:

        cluster_t() = default;

        explicit cluster_t(tensor_size_t samples, const indices_t& indices);
        explicit cluster_t(tensor_size_t samples, tensor_size_t groups = 1);

        ///
        /// \brief assign a sample to a group.
        ///
        void assign(const tensor_size_t sample, const tensor_size_t group)
        {
            assert(sample >= 0 && sample < samples());

            m_indices(sample) = group;
        }

        ///
        /// \brief call the given operator for all samples associated to the given group.
        ///
        template <typename toperator>
        void loop(const tensor_size_t group, const toperator& op)
        {
            assert(group >= 0 && group < groups());

            for (tensor_size_t i = 0, size = samples(); i < size; ++ i)
            {
                if (m_indices(i) == group)
                {
                    op(i);
                }
            }
        }

        ///
        /// \brief returns the samples associated to the given group.
        ///
        [[nodiscard]] indices_t indices(tensor_size_t group) const;

        ///
        /// \brief returns the number of samples associated to the given group.
        ///
        [[nodiscard]] tensor_size_t count(tensor_size_t group) const;

        ///
        /// \brief return the group index associated to the given sample.
        ///
        [[nodiscard]] tensor_size_t group(const tensor_size_t sample) const
        {
            assert(sample >= 0 && sample < samples());
            return m_indices(sample);
        }

        ///
        /// \brief returns the number of groups.
        ///
        [[nodiscard]] tensor_size_t groups() const
        {
            return m_groups;
        }

        ///
        /// \brief returns the number of samples.
        ///
        [[nodiscard]] tensor_size_t samples() const
        {
            return m_indices.size();
        }

    private:

        // attributes
        indices_t       m_indices;      ///< group indices / sample
        tensor_size_t   m_groups{0};    ///< #number of groups
    };
}
