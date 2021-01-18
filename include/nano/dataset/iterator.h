#pragma once

#include <nano/tensor.h>

namespace nano
{
    // single-label indices
    using sindices_t = tensor_mem_t<tensor_size_t, 1>;
    using sindices_map_t = tensor_map_t<tensor_size_t, 1>;
    using sindices_cmap_t = tensor_cmap_t<tensor_size_t, 1>;

    // multi-label indices
    using mindices_t = tensor_mem_t<tensor_size_t, 2>;
    using mindices_map_t = tensor_map_t<tensor_size_t, 2>;
    using mindices_cmap_t = tensor_cmap_t<tensor_size_t, 2>;

    ///
    /// \brief utility to iterate through a collection of samples of a dataset (e.g. the training samples)
    ///     useful for training and evaluating machine learning models.
    ///
    /// NB: the inputs and the targets are generated on the fly by default, but they can be cached if possible.
    ///
    class dataset_iterator_t
    {
    public:

        dataset_iterator_t(indices_t samples);

        virtual ~dataset_iterator_t() = default;

        virtual bool cache_inputs(int64_t bytes, execution) = 0;
        virtual bool cache_targets(int64_t bytes, execution) = 0;

        virtual feature_t target() const = 0;
        virtual tensor3d_dims_t tdims() const = 0;
        virtual tensor4d_cmap_t targets(tensor_range_t sample_range, tensor4d_t& buffer) const = 0;

        const indices_t& samples() const;

    private:

        // attributes
        indices_t   m_samples;      ///< samples
    };

    ///
    /// \brief dataset iterator useful for machine learning models that perform feature selection
    ///     (e.g. gradient boosting)
    ///
    /// NB: optional inputs are supported.
    /// NB: mixing continuous and discrete features is supported.
    ///
    class feature_dataset_iterator_t : public dataset_iterator_t
    {
    public:

        feature_dataset_iterator_t(indices_t samples);

        virtual tensor_size_t features() const = 0;
        virtual feature_t feature(tensor_size_t feature_index) const = 0;

        virtual sindices_cmap_t input(tensor_size_t feature_index, sindices_t& buffer) const = 0;
        virtual mindices_cmap_t input(tensor_size_t feature_index, mindices_t& buffer) const = 0;
        virtual tensor1d_cmap_t input(tensor_size_t feature_index, tensor1d_t& buffer) const = 0;
    };

    ///
    /// \brief dataset iterator useful for machine learning models
    ///     that map densely continuous inputs to targets (e.g. linear models, MLPs).
    ///
    /// NB: optional inputs are supported by pre-processing the missing values accordingly.
    /// NB: mixing continuous and discrete features is supported.
    ///
    class flatten_dataset_iterator_t : public dataset_iterator_t
    {
    public:

        flatten_dataset_iterator_t(indices_t samples);

        virtual tensor2d_cmap_t inputs(tensor_range_t sample_range, tensor2d_t& buffer) const = 0;
    };

    ///
    /// \brief dataset iterator useful for machine learning models
    ///     that map densely continuous structured inputs to targets (e.g. convolution neural networks).
    ///
    /// NB: optional inputs are not supported.
    /// NB: discrete inputs are not supported.
    /// NB: the structure of the inputs (e.g. images) is preserved if possible.
    ///
    class structured_dataset_iterator_t : public dataset_iterator_t
    {
    public:

        structured_dataset_iterator_t(indices_t samples);

        virtual tensor4d_cmap_t inputs(tensor_range_t sample_range, tensor4d_t& buffer) const = 0;
    };
}
