#pragma once

#include <nano/generator/elemwise.h>
#include <nano/generator/gradient.h>

namespace nano
{
    ///
    /// \brief generate image gradient-like structured features:
    ///     - vertical and horizontal gradients,
    ///     - edge orientation and magnitude.
    ///
    class NANO_PUBLIC elemwise_gradient_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::structured;

        elemwise_gradient_t(
            const memory_dataset_t& dataset,
            kernel3x3_type = kernel3x3_type::sobel,
            const indices_t& original_features = indices_t{});

        feature_t feature(tensor_size_t ifeature) const override;
        feature_mapping_t do_fit(indices_cmap_t, execution) override;

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, struct_map_t storage) const
        {
            const auto mode = mapped_mode(ifeature);
            const auto channel = mapped_channel(ifeature);
            const auto kernel = make_kernel3x3<scalar_t>(m_type);
            [[maybe_unused]] const auto [rows, cols, _] = mapped_dims(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    gradient3x3(mode, values, channel, kernel, storage.tensor(index).reshape(rows, cols));
                }
                else
                {
                    storage.tensor(index).full(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_flatten(dataset_iterator_t<tscalar, input_rank> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto should_drop = this->should_drop(ifeature);
            const auto mode = mapped_mode(ifeature);
            const auto channel = mapped_channel(ifeature);
            const auto kernel = make_kernel3x3<scalar_t>(m_type);
            [[maybe_unused]] const auto [rows, cols, _] = mapped_dims(ifeature);
            const auto colsize = rows * cols;
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    auto segment = storage.vector(index).segment(column, colsize);
                    if (should_drop)
                    {
                        segment.setConstant(+0.0);
                    }
                    else
                    {
                        gradient3x3(mode, values, channel, kernel, map_tensor(segment.data(), rows, cols));
                    }
                }
                else
                {
                    auto segment = storage.vector(index).segment(column, colsize);
                    segment.setConstant(+0.0);
                }
            }
            column += rows * cols;
        }

    private:

        tensor_size_t mapped_channel(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return mapping()(ifeature, 6);
        }

        gradient3x3_mode mapped_mode(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return static_cast<gradient3x3_mode>(mapping()(ifeature, 7));
        }

        // attributes
        kernel3x3_type  m_type{kernel3x3_type::sobel};  ///<
        indices_t       m_original_features;            ///<
    };
}
