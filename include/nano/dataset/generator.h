#pragma once

#include <nano/dataset/dataset.h>
#include <nano/mlearn/cluster.h>

namespace nano
{
    // TODO: factory of feature generators!!!
    class generator_t;
    using rgenerator_t = std::unique_ptr<generator_t>;
    using rgenerators_t = std::vector<rgenerator_t>;

    // multi-label categorical feature values: (sample index, label/class index) = {0, 1} if present
    using mclass_mem_t = tensor_mem_t<int8_t, 2>;
    using mclass_cmap_t = tensor_cmap_t<int8_t, 2>;

    // single-label categorical feature values: (sample index) = label/class index
    using sclass_mem_t = tensor_mem_t<int32_t, 1>;
    using sclass_cmap_t = tensor_cmap_t<int32_t, 1>;

    // scalar continuous feature values: (sample index) = scalar feature value
    using scalar_mem_t = tensor_mem_t<scalar_t, 1>;
    using scalar_cmap_t = tensor_cmap_t<scalar_t, 1>;

    // structured continuous feature values: (sample index, dim1, dim2, dim3)
    using struct_mem_t = tensor_mem_t<scalar_t, 4>;
    using struct_cmap_t = tensor_cmap_t<scalar_t, 4>;

    ///
    /// \brief generate features from a given collection of samples of a dataset (e.g. the training samples).
    ///
    /// NB: optional inputs are supported.
    /// NB: the targets cannot be optional if defined.
    /// NB: the inputs can be continuous (scalar), structured (3D tensors) or categorical.
    /// NB: the inputs and the targets are generated on the fly by default, but they can be cached if possible.
    ///
    class generator_t
    {
    public:

        generator_t(const memory_dataset_t& dataset) :
            m_dataset(dataset)
        {
        }

        virtual ~generator_t() = default;

        ///
        /// \brief returns the total number of generated features.
        ///
        virtual tensor_size_t features() const = 0;

        ///
        /// \brief returns the description of the given feature index.
        ///
        virtual feature_t feature(tensor_size_t feature) const = 0;

        ///
        /// \brief map the given feature index to the original list of features.
        ///
        virtual void original(tensor_size_t feature, cluster_t& original_features) const = 0;

        ///
        /// \brief computes the values of the given feature and samples,
        ///     useful for training and evaluating ML models that perform feature selection
        ///     (e.g. gradient boosting).
        ///
        virtual mclass_cmap_t select(tensor_size_t feature, indices_cmap_t samples, mclass_mem_t&) const = 0;
        virtual sclass_cmap_t select(tensor_size_t feature, indices_cmap_t samples, sclass_mem_t&) const = 0;
        virtual scalar_cmap_t select(tensor_size_t feature, indices_cmap_t samples, scalar_mem_t&) const = 0;
        virtual struct_cmap_t select(tensor_size_t feature, indices_cmap_t samples, struct_mem_t&) const = 0;

        ///
        /// \brief computes the values of all features for the given samples,
        ///     useful for training and evaluating ML model that map densely continuous inputs to targets
        ///     (e.g. linear models, MLPs).
        ///
        virtual tensor_size_t flatten_size() const = 0;
        virtual void flatten(indices_cmap_t samples, tensor2d_cmap_t) const = 0;

    protected:

        // attributes
        const memory_dataset_t& m_dataset;  ///<
    };

    ///
    /// \brief forward the original features as-is.
    ///
    /// NB: missing feature values are filled:
    ///     - with NaN/-1 depending if continuous/categorical respectively,
    ///         if accessing one feature at a time (e.g. feature selection models)
    ///
    ///     - with 0,
    ///         if accessing all features at once (e.g. linear models).
    ///
    class NANO_PUBLIC identity_generator_t final : public generator_t
    {
    public:

        identity_generator_t(const memory_dataset_t& dataset);

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;
        void original(tensor_size_t, cluster_t&) const override;

        mclass_cmap_t select(tensor_size_t, indices_cmap_t, mclass_mem_t&) const override;
        sclass_cmap_t select(tensor_size_t, indices_cmap_t, sclass_mem_t&) const override;
        scalar_cmap_t select(tensor_size_t, indices_cmap_t, scalar_mem_t&) const override;
        struct_cmap_t select(tensor_size_t, indices_cmap_t, struct_mem_t&) const override;

        tensor_size_t flatten_size() const override;
        void flatten(indices_cmap_t, tensor2d_cmap_t) const override;

    private:

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
        feature_mapping_t   m_mapping;  ///<
    };

    ///
    /// \brief
    ///
    class dataset_generator_t
    {
    public:

        dataset_generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
            m_dataset(dataset),
            m_samples(samples)
        {
            add_generator<identity_generator_t>();
        }

        template <typename tgenerator, typename... tgenerator_args>
        dataset_generator_t& add_generator(tgenerator_args... args)
        {
            static_assert(std::is_base_of<generator_t, tgenerator>::value);

            m_generators.push_back(std::make_unique<tgenerator>(m_dataset, args...));
            update();
            return *this;
        }

        tensor_size_t features() const;
        feature_t feature(tensor_size_t feature) const;
        indices_t original_features(const indices_t& features) const;

        mclass_cmap_t select(tensor_size_t feature, mclass_mem_t&) const;
        sclass_cmap_t select(tensor_size_t feature, sclass_mem_t&) const;
        scalar_cmap_t select(tensor_size_t feature, scalar_mem_t&) const;
        struct_cmap_t select(tensor_size_t feature, struct_mem_t&) const;

        tensor_size_t flatten_size() const;
        tensor2d_dims_t flatten_dims() const;
        tensor2d_cmap_t flatten(tensor_range_t sample_range, tensor2d_t&) const;
        tensor4d_cmap_t targets(tensor_range_t sample_range, tensor4d_t&) const;

    private:

        void update();

        // attributes
        const memory_dataset_t& m_dataset;      ///<
        const indices_t&        m_samples;      ///<
        rgenerators_t           m_generators;   ///<
    };
}
