#pragma once

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

        elemwise_scalar2sclass_t(
            const memory_dataset_t& dataset,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& original_feature_indices = indices_t{}) :
            base_elemwise_generator_t(dataset),
            m_s2s(s2s),
            m_original_feature_indices(original_feature_indices)
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

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, sclass_map_t storage) const
        {
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = toperator::label(values(component));
                }
                else
                {
                    storage(index) = -1;
                }
            }
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_flatten(dataset_iterator_t<tscalar, input_rank> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto colsize = toperator::labels();
            const auto should_drop = this->should_drop(ifeature);
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    auto segment = storage.array(index).segment(column, colsize);
                    if (should_drop)
                    {
                        segment.setConstant(0.0);
                    }
                    else
                    {
                        const auto label = toperator::label(values(component));
                        segment.setConstant(-1.0);
                        segment(label) = 1.0;
                    }
                }
                else
                {
                    auto segment = storage.array(index).segment(column, colsize);
                    segment.setConstant(0.0);
                }
            }
            column += colsize;
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
