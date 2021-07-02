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

        explicit elemwise_gradient_t(
            const dataset_t& dataset,
            kernel3x3_type = kernel3x3_type::sobel,
            indices_t original_features = indices_t{});

        feature_t feature(tensor_size_t ifeature) const override;
        feature_mapping_t do_fit(indices_cmap_t, execution) override;

        auto process(tensor_size_t ifeature) const
        {
            const auto dims = mapped_dims(ifeature);
            const auto mode = mapped_mode(ifeature);
            const auto channel = mapped_channel(ifeature);
            const auto kernel = make_kernel3x3<scalar_t>(m_type);

            const auto rows = std::get<0>(dims);
            const auto cols = std::get<1>(dims);
            const auto colsize = rows * cols;
            const auto process = [=] (const auto& values, auto&& storage)
            {
                gradient3x3(mode, values, channel, kernel, map_tensor(storage.data(), rows, cols));
            };

            return std::make_tuple(process, colsize);
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
