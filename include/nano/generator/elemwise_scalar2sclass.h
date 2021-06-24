#pragma once

#include <nano/generator/select.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief state-less transformation of scalar (or structured) features into single-label features
    ///     (e.g. via a non-linear transformation).
    ///
    template <typename toperator>
    class elemwise_scalar2sclass_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::sclass;

        explicit elemwise_scalar2sclass_t(
            const memory_dataset_t& dataset,
            struct2scalar s2s = struct2scalar::off,
            indices_t original_feature_indices = indices_t{}) :
            base_elemwise_generator_t(dataset),
            m_s2s(s2s),
            m_original_feature_indices(std::move(original_feature_indices))
        {
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_scalar(dataset(), m_s2s, m_original_feature_indices);
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            return make_sclass_feature(ifeature, toperator::name(), toperator::label_strings());
        }

        auto process(tensor_size_t ifeature) const
        {
            const auto component = mapped_component(ifeature);

            const auto colsize = toperator::labels();
            const auto process = [=] (const auto& values)
            {
                return toperator::label(values(component));
            };

            return std::make_tuple(process, colsize);
        }

    private:

        // attributes
        struct2scalar   m_s2s{struct2scalar::off};  ///<
        indices_t       m_original_feature_indices; ///<
    };

    class sign_class_t
    {
    public:

        static const char* name()
        {
            return "sign_class";
        }

        static strings_t label_strings()
        {
            return {"neg", "pos"};
        }

        static constexpr tensor_size_t labels()
        {
            return 2;
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto label(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return svalue < 0.0 ? 0 : 1;
        }
    };
}
