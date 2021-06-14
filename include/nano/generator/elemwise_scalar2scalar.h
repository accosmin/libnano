#pragma once

#include <nano/generator/select.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief state-less transformation of scalar (or structured) features into scalar features
    ///     (e.g. via a non-linear transformation).
    ///
    template <typename toperator>
    class elemwise_scalar2scalar_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        elemwise_scalar2scalar_t(
            const memory_dataset_t& dataset,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& original_features = indices_t{}) :
            base_elemwise_generator_t(dataset),
            m_s2s(s2s),
            m_original_features(original_features)
        {
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_scalar(dataset(), m_s2s, m_original_features);
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            return make_scalar_feature(ifeature, toperator::name());
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, scalar_map_t storage) const
        {
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = toperator::scalar(values(component));
                }
                else
                {
                    storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                }
            }
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_flatten(dataset_iterator_t<tscalar, input_rank> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto should_drop = this->should_drop(ifeature);
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    if (should_drop)
                    {
                        storage(index, column) = +0.0;
                    }
                    else
                    {
                        storage(index, column) = toperator::scalar(values(component));
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
        indices_t       m_original_features;        ///<
    };

    class slog1p_t
    {
    public:

        static const char* name()
        {
            return "slog1p";
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto scalar(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return (svalue < 0.0 ? -1.0 : +1.0) * std::log1p(std::fabs(svalue));
        }
    };

    class sign_t
    {
    public:

        static const char* name()
        {
            return "sign";
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto scalar(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return svalue < 0.0 ? -1.0 : +1.0;
        }
    };
}
