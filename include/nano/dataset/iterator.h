#pragma once

#include <nano/tensor.h>

namespace nano
{
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

        virtual bool cache_inputs(int64_t bytes) = 0;
        virtual bool cache_targets(int64_t bytes) = 0;

        virtual tensor3d_dims_t tdims() const = 0;
        virtual tensor4d_cmap_t targets(tensor_range_t sample_range, tensor4d_t& buffer) const = 0;

        const indices_t& samples() const;

    private:

        // attributes
        indices_t   m_samples;      ///< samples
    };

    ///
    /// \brief feature-wise sample iterator useful for machine learning models
    ///     that perform feature selection (e.g. gradient boosting).
    ///
    /// NB: optional inputs are supported.
    ///
    class feature_dataset_iterator_t : public dataset_iterator_t
    {
    public:

        feature_dataset_iterator_t(indices_t samples);

        virtual tensor_size_t features() const = 0;
        virtual const feature_t& feature(tensor_size_t feature_index) const = 0;

        virtual indices_cmap_t input(tensor_size_t feature_index, indices_t& buffer) const = 0;
        virtual tensor1d_cmap_t input(tensor_size_t feature_index, tensor1d_t& buffer) const = 0;
    };

    ///
    /// \brief dense sample iterator useful for machine learning models
    ///     that densely map the inputs to targets (e.g. linear models, convolution neural networks).
    ///
    /// NB: optional inputs are not supported.
    /// NB: the structure of the inputs (e.g. images) is preserved if possible.
    ///
    class dense_dataset_iterator_t : public dataset_iterator_t
    {
    public:

        dense_dataset_iterator_t(indices_t samples);

        virtual tensor4d_cmap_t inputs(tensor_range_t sample_range, tensor4d_t& buffer) const = 0;
    };
}
