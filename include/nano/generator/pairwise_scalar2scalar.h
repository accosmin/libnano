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

        explicit pairwise_scalar2scalar_t(
            const dataset_t& dataset,
            struct2scalar s2s = struct2scalar::off,
            indices_t original_feature_indices = indices_t{}) :
            base_pairwise_generator_t(dataset),
            m_s2s(s2s),
            m_original_feature_indices(std::move(original_feature_indices))
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

        auto process(tensor_size_t ifeature) const
        {
            const auto component1 = mapped_component1(ifeature);
            const auto component2 = mapped_component2(ifeature);

            const auto colsize = tensor_size_t{1};
            const auto process = [=] (const auto& values1, const auto& values2)
            {
                return toperator::scalar(values1(component1), values2(component2));
            };

            return std::make_tuple(process, colsize);
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
