#pragma once

#include <nano/dataset/iterator.h>

namespace nano
{
    class memory_dataset_t;

    ///
    /// \brief
    ///
    class NANO_PUBLIC memory_feature_dataset_iterator_t final : public feature_dataset_iterator_t
    {
    public:

        memory_feature_dataset_iterator_t(const memory_dataset_t&, indices_t samples);

        const indices_t& samples() const override;

        feature_t target() const override;
        tensor3d_dims_t target_dims() const override;
        tensor4d_cmap_t targets(tensor4d_t& buffer) const override;

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t feature) const override;
        feature_t original_feature(tensor_size_t feature) const override;

        indices_cmap_t input(tensor_size_t feature, indices_t& buffer) const override;
        tensor1d_cmap_t input(tensor_size_t feature, tensor1d_t& buffer) const override;
        tensor4d_cmap_t input(tensor_size_t feature, tensor4d_t& buffer) const override;

    private:

        using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        const memory_dataset_t&    m_dataset;  ///<
        indices_t           m_samples;  ///<
        feature_mapping_t   m_mapping;  ///<
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC memory_flatten_dataset_iterator_t final : public flatten_dataset_iterator_t
    {
    public:

        memory_flatten_dataset_iterator_t(const memory_dataset_t&, indices_t samples);

        const indices_t& samples() const override;
        tensor2d_t normalize(normalization) const override;
        feature_t original_feature(tensor_size_t input) const override;

        feature_t target() const override;
        tensor3d_dims_t target_dims() const override;
        tensor4d_cmap_t targets(tensor_range_t samples, tensor4d_t& buffer) const override;

        tensor1d_dims_t inputs_dims() const override;
        tensor2d_cmap_t inputs(tensor_range_t samples, tensor2d_t& buffer) const override;

    private:

        using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        const memory_dataset_t&    m_dataset;  ///<
        indices_t           m_samples;  ///<
        feature_mapping_t   m_mapping;  ///<
    };
}
