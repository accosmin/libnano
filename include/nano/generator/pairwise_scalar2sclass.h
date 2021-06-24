#pragma once

#include <nano/generator/select.h>
#include <nano/generator/pairwise.h>

namespace nano
{
    ///
    /// \brief state-less transformation of pairwise scalar (or structured) features into single label features
    ///     (e.g. via a non-linear transformation).
    ///
    template <typename toperator>
    class pairwise_scalar2sclass_t : public base_pairwise_generator_t
    {
    public:

        static constexpr auto input_rank1 = 4U;
        static constexpr auto input_rank2 = 4U;
        static constexpr auto generated_type = generator_type::sclass;

        explicit pairwise_scalar2sclass_t(
            const memory_dataset_t& dataset,
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
            return make_sclass_feature(ifeature, toperator::name(), toperator::label_strings());
        }

        auto process(tensor_size_t ifeature) const
        {
            const auto component1 = mapped_component1(ifeature);
            const auto component2 = mapped_component2(ifeature);

            const auto colsize = toperator::labels();
            const auto process = [=] (const auto& values1, const auto& values2)
            {
                return toperator::label(values1(component1), values2(component2));
            };

            return std::make_tuple(process, colsize);
        }

    private:

        // attributes
        struct2scalar   m_s2s{struct2scalar::off};  ///<
        indices_t       m_original_feature_indices; ///<
    };

    class product_sign_class_t
    {
    public:

        static const char* name()
        {
            return "product_sign_class";
        }

        static strings_t label_strings()
        {
            return {"neg", "pos"};
        }

        static constexpr tensor_size_t labels()
        {
            return 2;
        }

        template
        <
            typename tscalar1,
            typename tscalar2,
            std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true
        >
        static auto label(tscalar1 value1, tscalar2 value2)
        {
            return (static_cast<scalar_t>(value1) * static_cast<scalar_t>(value2) < 0.0) ? 0 : 1;
        }
    };
}
