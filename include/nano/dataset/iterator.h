#pragma once

#include <nano/tensor.h>
#include <nano/mlearn/feature.h>

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

    class feature_dataset_iterator_t;
    class flatten_dataset_iterator_t;

    using rfeature_dataset_iterator_t = std::unique_ptr<feature_dataset_iterator_t>;
    using rflatten_dataset_iterator_t = std::unique_ptr<flatten_dataset_iterator_t>;

    ///
    /// \brief iterate through a collection of samples of a dataset (e.g. the training samples)
    ///     to train and evaluate machine learning models that perform feature selection (e.g. gradient boosting).
    ///
    /// NB: optional inputs are supported.
    /// NB: the targets cannot be optional if defined.
    /// NB: the inputs can be continuous (scalar), structured (3D tensors) or categorical.
    /// NB: the inputs and the targets are generated on the fly by default, but they can be cached if possible.
    ///
    class feature_dataset_iterator_t
    {
    public:

        virtual ~feature_dataset_iterator_t() = default;

        virtual const indices_t& samples() const = 0;

        virtual tensor_size_t features() const = 0;
        virtual indices_t scalar_features() const = 0;
        virtual indices_t struct_features() const = 0;
        virtual indices_t sclass_features() const = 0;
        virtual indices_t mclass_features() const = 0;
        virtual feature_t feature(tensor_size_t feature) const = 0;

        virtual sindices_cmap_t input(tensor_size_t feature, sindices_t& buffer) const = 0;
        virtual mindices_cmap_t input(tensor_size_t feature, mindices_t& buffer) const = 0;
        virtual tensor1d_cmap_t input(tensor_size_t feature, tensor1d_t& buffer) const = 0;
        virtual tensor4d_cmap_t input(tensor_size_t feature, tensor4d_t& buffer) const = 0;

        virtual feature_t target() const = 0;
        virtual tensor3d_dims_t target_dims() const = 0;
        virtual tensor4d_cmap_t targets(tensor_range_t samples, tensor4d_t& buffer) const = 0;

        virtual bool cache_inputs(int64_t bytes, execution) = 0;
        virtual bool cache_targets(int64_t bytes, execution) = 0;
        virtual bool cache_inputs(int64_t bytes, indices_cmap_t features, execution) = 0;
    };

    ///
    /// \brief iterate through a collection of samples of a dataset (e.g. the training samples)
    ///     to map densely continuous inputs to targets (e.g. linear models, MLPs).
    ///
    /// NB: optional inputs are supported.
    /// NB: the targets cannot be optional if defined.
    /// NB: the inputs can be continuous (scalar), structured (3D tensors) or categorical.
    /// NB: the inputs and the targets are generated on the fly by default, but they can be cached if possible.
    ///
    class flatten_dataset_iterator_t
    {
    public:

        virtual ~flatten_dataset_iterator_t() = default;

        virtual const indices_t& samples() const = 0;

        virtual void normalize(normalization) = 0;
        virtual tensor1d_dims_t inputs_dims() const = 0;
        virtual tensor2d_cmap_t inputs(tensor_range_t samples, tensor2d_t& buffer) const = 0;

        virtual feature_t target() const = 0;
        virtual tensor3d_dims_t target_dims() const = 0;
        virtual tensor4d_cmap_t targets(tensor_range_t samples, tensor4d_t& buffer) const = 0;

        virtual bool cache_inputs(int64_t bytes, execution) = 0;
        virtual bool cache_targets(int64_t bytes, execution) = 0;
    };
}
