#pragma once

#include <nano/dataset/dataset.h>

namespace nano
{
    ///
    /// \brief base class to iterate through a collection of samples of a dataset (e.g. the training samples).
    ///
    /// NB: optional inputs are supported.
    /// NB: the targets cannot be optional if defined.
    /// NB: the inputs can be continuous (scalar), structured (3D tensors) or categorical.
    /// NB: the inputs and the targets are generated on the fly by default, but they can be cached if possible.
    ///
    class dataset_iterator_t
    {
    public:

        dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples);

        feature_t target() const;
        tensor3d_dims_t target_dims() const;
        tensor4d_cmap_t targets(tensor4d_t& buffer) const;
        tensor4d_cmap_t targets(tensor_range_t samples_range, tensor4d_t& buffer) const;

        ///
        /// \brief access functions.
        ///
        const auto& samples() const { return m_samples; }
        const auto& dataset() const { return m_dataset; }
        const auto& mapping() const { return m_mapping; }

    protected:

        // map to original feature: feature index, component index (if applicable)
        using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        void map1(tensor_size_t& f, tensor_size_t original, tensor_size_t component)
        {
            m_mapping(f, 0) = original;
            m_mapping(f ++, 1) = component;
        }

        void mapN(tensor_size_t& f, tensor_size_t original, tensor_size_t components)
        {
            for (tensor_size_t component = 0; component < components; ++ component)
            {
                map1(f, original, component);
            }
        }

        // attributes
        const memory_dataset_t& m_dataset;  ///<
        indices_t           m_samples;  ///<
        feature_mapping_t   m_mapping;  ///<
    };

    ///
    /// \brief iterate through a collection of samples of a dataset (e.g. the training samples)
    ///     to train and evaluate machine learning models that perform feature selection (e.g. gradient boosting).
    ///
    class feature_dataset_iterator_t : public dataset_iterator_t
    {
    public:

        using sclass_values_t = tensor_mem_t<tensor_size_t, 1>;
        using mclass_values_t = tensor_mem_t<tensor_size_t, 2>;

        feature_dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples);

        indices_t sclass_features() const;
        indices_t mclass_features() const;
        indices_t scalar_features() const;
        indices_t struct_features() const;

        indices_t original_features(const indices_cmap_t& features) const;

        void inputs(tensor_size_t feature,

        indices_cmap_t input(tensor_size_t feature, indices_t& buffer) const;
        tensor1d_cmap_t input(tensor_size_t feature, tensor1d_t& buffer) const;
        tensor4d_cmap_t input(tensor_size_t feature, tensor4d_t& buffer) const;
    };

    ///
    /// \brief iterate through a collection of samples of a dataset (e.g. the training samples)
    ///     to map densely continuous inputs to targets (e.g. linear models, MLPs).
    ///
    class flatten_dataset_iterator_t : public dataset_iterator_t
    {
    public:

        flatten_dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples);

        template <typename toperator>
        void iterate(execution, tensor_size_t batch, const toperator& op)
        {
            ....
                op(tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
                {
                }
        };


        tensor2d_t normalize(normalization) const;
        feature_t original_feature(tensor_size_t feature) const;

        tensor1d_dims_t inputs_dims() const;
        tensor2d_cmap_t inputs(tensor_range_t samples_range, tensor2d_t& buffer) const;
    };
}
