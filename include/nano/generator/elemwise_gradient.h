#pragma once

#include <nano/generator/elemwise.h>
#include <nano/generator/gradient.h>

namespace nano
{
    ///
    /// \brief image gradient type.
    ///
    enum class gradient_type
    {
        sobel,
        scharr,
        prewitt,
    };

    template <>
    inline enum_map_t<gradient_type> enum_string<gradient_type>()
    {
        return
        {
            { gradient_type::sobel,         "sobel" },
            { gradient_type::scharr,        "scharr" },
            { gradient_type::prewitt,       "prewitt" },
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, gradient_type value)
    {
        return stream << scat(value);
    }

    ///
    /// \brief generate image gradient-like structured features:
    ///     - vertical and horizontal gradients,
    ///     - edge orientation and magnitude.
    ///
    class elemwise_gradient_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::structured;

        elemwise_gradient_t(
            const memory_dataset_t& dataset,
            gradient_type = gradient_type::sobel,
            const indices_t& original_features = indices_t{});

        feature_t feature(tensor_size_t ifeature) const override;
        feature_mapping_t do_fit(indices_cmap_t, execution) override;

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, struct_map_t storage) const
        {
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = values(component); //toperator::scalar(values(component));
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
                        storage(index, column) = values(component); //toperator::scalar(values(component));
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

        tensor_size_t mapped_feature_type(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return mapping()(ifeature, 6);
        }

        // attributes
        gradient_type   m_type{gradient_type::sobel};   ///<
        indices_t       m_original_features;            ///<
    };
}
