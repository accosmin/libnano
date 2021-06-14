#pragma once

#include <nano/generator/select.h>
#include <nano/generator/pairwise.h>

namespace nano
{
    ///
    /// \brief state-less transformation of pairwise scalar (or structured) features into scalar features
    ///     (e.g. via a non-linear transformation).
    ///
    template <typename toperator>
    class pairwise_scalar2scalar_t : public base_pairwise_generator_t
    {
    public:

        static constexpr auto input_rank1 = 4U;
        static constexpr auto input_rank2 = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        pairwise_scalar2scalar_t(
            const memory_dataset_t& dataset,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& original_feature_indices = indices_t{}) :
            base_pairwise_generator_t(dataset),
            m_s2s(s2s),
            m_original_feature_indices(original_feature_indices)
        {
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return make_pairwise(select_scalar(dataset(), m_s2s, m_original_feature_indices));
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            return make_scalar_feature(ifeature, toperator::name());
        }

        template
        <
            typename tscalar1, typename tscalar2,
            std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true
        >
        void do_select(
            dataset_pairwise_iterator_t<tscalar1, input_rank1, tscalar2, input_rank2> it,
            tensor_size_t ifeature, scalar_map_t storage) const
        {
            const auto component1 = mapped_component1(ifeature);
            const auto component2 = mapped_component2(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                {
                    storage(index) = toperator::scalar(values1(component1), values2(component2));
                }
                else
                {
                    storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                }
            }
        }

        template
        <
            typename tscalar1, typename tscalar2,
            std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true
        >
        void do_flatten(
            dataset_pairwise_iterator_t<tscalar1, input_rank1, tscalar2, input_rank2> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto should_drop = this->should_drop(ifeature);
            const auto component1 = mapped_component1(ifeature);
            const auto component2 = mapped_component2(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                {
                    if (should_drop)
                    {
                        storage(index, column) = +0.0;
                    }
                    else
                    {
                        storage(index, column) = toperator::scalar(values1(component1), values2(component2));
                    }
                }
                else
                {
                    storage(index, column) = +0.0;
                }
            }
            ++ column;
        }

    private:

        // attributes
        struct2scalar   m_s2s{struct2scalar::off};  ///<
        indices_t       m_original_feature_indices; ///<
    };

    class product_t
    {
    public:

        static const char* name()
        {
            return "product";
        }

        template
        <
            typename tscalar1,
            typename tscalar2,
            std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true
        >
        static auto scalar(tscalar1 value1, tscalar2 value2)
        {
            return static_cast<scalar_t>(value1) * static_cast<scalar_t>(value2);
        }
    };
}
