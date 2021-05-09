#pragma once

#include <nano/dataset/stats.h>
#include <nano/dataset/dataset.h>

namespace nano
{
    class generator_t;
    using rgenerator_t = std::unique_ptr<generator_t>;
    using rgenerators_t = std::vector<rgenerator_t>;

    // single-label categorical feature values: (sample index) = label/class index
    using sclass_mem_t = tensor_mem_t<int32_t, 1>;
    using sclass_map_t = tensor_map_t<int32_t, 1>;
    using sclass_cmap_t = tensor_cmap_t<int32_t, 1>;

    // multi-label categorical feature values: (sample index, label/class index) = 0 or 1
    using mclass_mem_t = tensor_mem_t<int8_t, 2>;
    using mclass_map_t = tensor_map_t<int8_t, 2>;
    using mclass_cmap_t = tensor_cmap_t<int8_t, 2>;

    // scalar continuous feature values: (sample index) = scalar feature value
    using scalar_mem_t = tensor_mem_t<scalar_t, 1>;
    using scalar_map_t = tensor_map_t<scalar_t, 1>;
    using scalar_cmap_t = tensor_cmap_t<scalar_t, 1>;

    // structured continuous feature values: (sample index, dim1, dim2, dim3)
    using struct_mem_t = tensor_mem_t<scalar_t, 4>;
    using struct_map_t = tensor_map_t<scalar_t, 4>;
    using struct_cmap_t = tensor_cmap_t<scalar_t, 4>;

    ///
    /// \brief toggle the generation of binary classification features
    ///     from single-label and multi-label multi-class features.
    ///
    /// NB: one binary classification feature is generated for each class.
    ///
    enum class sclass2binary : int32_t
    {
        off = 0, on
    };

    enum class mclass2binary : int32_t
    {
        off = 0, on
    };

    ///
    /// \brief toggle the generation of scalar (univariate) continuous features
    ///     from structured (multivariate) continuous features.
    ///
    /// NB: one scalar feature is generated for each component of the structured feature.
    ///
    enum class struct2scalar : int32_t
    {
        off = 0,
        on
    };

    ///
    /// \brief generate features from a given collection of samples of a dataset (e.g. the training samples).
    ///
    /// NB: optional inputs are supported.
    /// NB: the targets cannot be optional if defined.
    /// NB: the inputs can be continuous (scalar), structured (3D tensors) or categorical.
    /// NB: the inputs and the targets are generated on the fly by default, but they can be cached if possible.
    ///
    /// NB: missing feature values are filled:
    ///     - with NaN/-1 depending if continuous/categorical respectively,
    ///         if accessing one feature at a time (e.g. feature selection models)
    ///
    ///     - with 0,
    ///         if accessing all features at once (e.g. linear models).
    ///
    class NANO_PUBLIC generator_t
    {
    public:

        ///
        /// \brief constructor.
        ///
        generator_t(const memory_dataset_t& dataset);

        ///
        /// \brief default destructor.
        ///
        virtual ~generator_t() = default;

        ///
        /// \brief compute dataset-specific parameters for the given set of samples (if needed)
        ///     and process the whole dataset to generate features fast when needed (if needed).
        ///
        virtual void fit(indices_cmap_t samples, execution);

        ///
        /// \brief returns the total number of generated features.
        ///
        virtual tensor_size_t features() const = 0;

        ///
        /// \brief returns the description of the given feature index.
        ///
        virtual feature_t feature(tensor_size_t feature) const = 0;

        ///
        /// \brief toggle dropping of features, useful for feature importance analysis.
        ///
        void undrop();
        void drop(tensor_size_t feature);

        ///
        /// \brief toggle sample permutation of features, useful for feature importance analysis.
        ///
        void unshuffle();
        void shuffle(tensor_size_t feature);
        indices_t shuffled(indices_cmap_t samples, tensor_size_t feature) const;

        ///
        /// \brief computes the values of the given feature and samples,
        ///     useful for training and evaluating ML models that perform feature selection
        ///     (e.g. gradient boosting).
        ///
        virtual void select(indices_cmap_t samples, tensor_size_t feature, sclass_map_t) const;
        virtual void select(indices_cmap_t samples, tensor_size_t feature, mclass_map_t) const;
        virtual void select(indices_cmap_t samples, tensor_size_t feature, scalar_map_t) const;
        virtual void select(indices_cmap_t samples, tensor_size_t feature, struct_map_t) const;

        ///
        /// \brief computes the values of all features for the given samples,
        ///     useful for training and evaluating ML model that map densely continuous inputs to targets
        ///     (e.g. linear models, MLPs).
        ///
        virtual void flatten(indices_cmap_t samples, tensor2d_map_t, tensor_size_t column) const = 0;

    protected:

        void allocate(tensor_size_t features);

        const auto& dataset() const { return m_dataset; }
        auto should_drop(tensor_size_t feature) const { return m_feature_infos(feature) == 0x01; }
        auto should_shuffle(tensor_size_t feature) const { return m_feature_infos(feature) == 0x02; }

        template <typename toperator>
        void iterate1(
            indices_cmap_t samples, tensor_size_t ifeature, tensor_size_t ioriginal,
            const toperator& op) const
        {
            m_dataset.visit_inputs(ioriginal, [&] (const auto& feature, const auto& data, const auto& mask)
            {
                if (should_shuffle(ifeature))
                {
                    op(feature, data, mask, shuffled(samples, ifeature));
                }
                else
                {
                    op(feature, data, mask, samples);
                }
            });
        }

        template <typename toperator>
        void iterate2(
            indices_cmap_t samples, tensor_size_t ifeature, tensor_size_t ioriginal1, tensor_size_t ioriginal2,
            const toperator& op) const
        {
            m_dataset.visit_inputs(ioriginal1, [&] (const auto& feature1, const auto& data1, const auto& mask1)
            {
                m_dataset.visit_inputs(ioriginal2, [&] (const auto& feature2, const auto& data2, const auto& mask2)
                {
                    if (should_shuffle(ifeature))
                    {
                        op(feature1, data1, mask1, feature2, data2, mask2, shuffled(samples, ifeature));
                    }
                    else
                    {
                        op(feature1, data1, mask1, feature2, data2, mask2, samples);
                    }
                });
            });
        }

    private:

        // per feature:
        //  - 0: flags - 0 - default, 1 - to drop, 2 - to shuffle
        using feature_infos_t = tensor_mem_t<uint8_t, 1>;

        // per feature:
        //  - random number generator to use to shuffle the given samples
        using feature_rands_t = std::vector<rng_t>;

        // attributes
        const memory_dataset_t&    m_dataset;   ///<
        feature_infos_t     m_feature_infos;    ///<
        feature_rands_t     m_feature_rands;    ///<
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC dataset_generator_t
    {
    public:

        dataset_generator_t(const memory_dataset_t& dataset);

        template <typename tgenerator, typename... tgenerator_args>
        dataset_generator_t& add(tgenerator_args... args)
        {
            static_assert(std::is_base_of_v<generator_t, tgenerator>);

            auto generator = std::make_unique<tgenerator>(m_dataset, args...);
            m_generators.push_back(std::move(generator));
            update();
            return *this;
        }

        void fit(indices_cmap_t samples, execution) const;

        tensor_size_t features() const;
        feature_t feature(tensor_size_t feature) const;

        select_stats_t select_stats(execution) const;
        sclass_cmap_t select(indices_cmap_t samples, tensor_size_t feature, sclass_mem_t&) const;
        mclass_cmap_t select(indices_cmap_t samples, tensor_size_t feature, mclass_mem_t&) const;
        scalar_cmap_t select(indices_cmap_t samples, tensor_size_t feature, scalar_mem_t&) const;
        struct_cmap_t select(indices_cmap_t samples, tensor_size_t feature, struct_mem_t&) const;

        tensor_size_t columns() const;
        tensor_size_t column2feature(tensor_size_t column) const;
        tensor2d_cmap_t flatten(indices_cmap_t samples, tensor2d_t&) const;
        flatten_stats_t flatten_stats(indices_cmap_t samples, execution, tensor_size_t batch) const;

        feature_t target() const;
        tensor3d_dims_t target_dims() const;
        tensor4d_cmap_t targets(indices_cmap_t samples, tensor4d_t&) const;
        targets_stats_t targets_stats(indices_cmap_t samples, execution, tensor_size_t batch) const;

        tensor1d_t sample_weights(indices_cmap_t samples, const targets_stats_t&) const;

        void undrop() const;
        void drop(tensor_size_t feature) const;

        void unshuffle() const;
        void shuffle(tensor_size_t feature) const;
        indices_t shuffled(indices_cmap_t samples, tensor_size_t feature) const;

        // TODO: support for caching - all or selection of features

        const auto& dataset() const { return m_dataset; }

    private:

        void update();
        void check(tensor_size_t feature) const;
        void check(indices_cmap_t samples) const;
        const rgenerator_t& byfeature(tensor_size_t feature) const;

        // per column:
        //  - 0: generator index,
        //  - 1: column index within generator,
        //  - 2: offset n_features (up to the current generator)
        using column_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // per feature:
        //  - 0: generator index,
        //  - 1: feature index within generator,
        //  - 2-4: feature dimensions (dim1, dim2, dim3)
        using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // per generator:
        //  - 0: number of features
        using generator_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        const memory_dataset_t& m_dataset;              ///<
        rgenerators_t           m_generators;           ///<
        column_mapping_t        m_column_mapping;       ///<
        feature_mapping_t       m_feature_mapping;      ///<
        generator_mapping_t     m_generator_mapping;    ///<
    };
}
