#pragma once

#include <unordered_map>
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
        generator_t(const memory_dataset_t& dataset, const indices_t& samples);

        ///
        /// \brief default destructor.
        ///
        virtual ~generator_t() = default;

        ///
        /// \brief prepare the given samples to generate features fast when needed.
        ///
        virtual void preprocess(execution);

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
        indices_t shuffle(tensor_size_t feature);

        ///
        /// \brief computes the values of the given feature and samples,
        ///     useful for training and evaluating ML models that perform feature selection
        ///     (e.g. gradient boosting).
        ///
        virtual void select(tensor_size_t feature, tensor_range_t sample_range, sclass_map_t) const;
        virtual void select(tensor_size_t feature, tensor_range_t sample_range, mclass_map_t) const;
        virtual void select(tensor_size_t feature, tensor_range_t sample_range, scalar_map_t) const;
        virtual void select(tensor_size_t feature, tensor_range_t sample_range, struct_map_t) const;

        ///
        /// \brief computes the values of all features for the given samples,
        ///     useful for training and evaluating ML model that map densely continuous inputs to targets
        ///     (e.g. linear models, MLPs).
        ///
        virtual void flatten(tensor_range_t sample_range, tensor2d_map_t, tensor_size_t column) const = 0;

        ///
        /// \brief access functions
        ///
        const auto& dataset() const { return m_dataset; }
        const auto& samples() const { return m_samples; }

    protected:

        void allocate(tensor_size_t features);

        auto should_drop(tensor_size_t feature) const { return m_feature_infos(feature) == 1; }
        auto should_shuffle(tensor_size_t feature) const { return m_feature_infos(feature) == 2; }

        auto samples(tensor_size_t feature, tensor_range_t sample_range) const
        {
            const auto& all_samples = should_shuffle(feature) ? m_shuffle_indices.find(feature)->second : samples();
            return all_samples.slice(sample_range);
        }

    private:

        // per-feature information:
        //  - flags: 0 - default, 1 - to drop, 2 - to shuffle
        using feature_infos_t = tensor_mem_t<tensor_size_t, 1>;

        // fixed sample indices for the features to shuffle
        using shuffle_indices_t = std::unordered_map<tensor_size_t, indices_t>;

        // attributes
        const memory_dataset_t& m_dataset;      ///<
        const indices_t&    m_samples;          ///<
        feature_infos_t     m_feature_infos;    ///<
        shuffle_indices_t   m_shuffle_indices;  ///<
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC dataset_generator_t
    {
    public:

        dataset_generator_t(const memory_dataset_t& dataset, indices_t samples);

        template <typename tgenerator, typename... tgenerator_args>
        dataset_generator_t& add(execution ex, tgenerator_args... args)
        {
            static_assert(std::is_base_of_v<generator_t, tgenerator>);

            auto generator = std::make_unique<tgenerator>(m_dataset, m_samples, args...);
            generator->preprocess(ex);
            m_generators.push_back(std::move(generator));
            update();
            return *this;
        }

        tensor_size_t features() const;
        feature_t feature(tensor_size_t feature) const;

        select_stats_t select_stats(execution) const;
        sclass_cmap_t select(tensor_size_t feature, sclass_mem_t&) const;
        mclass_cmap_t select(tensor_size_t feature, mclass_mem_t&) const;
        scalar_cmap_t select(tensor_size_t feature, scalar_mem_t&) const;
        struct_cmap_t select(tensor_size_t feature, struct_mem_t&) const;
        sclass_cmap_t select(tensor_size_t feature, tensor_range_t sample_range, sclass_mem_t&) const;
        mclass_cmap_t select(tensor_size_t feature, tensor_range_t sample_range, mclass_mem_t&) const;
        scalar_cmap_t select(tensor_size_t feature, tensor_range_t sample_range, scalar_mem_t&) const;
        struct_cmap_t select(tensor_size_t feature, tensor_range_t sample_range, struct_mem_t&) const;

        tensor_size_t columns() const;
        tensor_size_t column2feature(tensor_size_t column) const;
        flatten_stats_t flatten_stats(execution, tensor_size_t batch) const;
        tensor2d_cmap_t flatten(tensor_range_t sample_range, tensor2d_t&) const;

        feature_t target() const;
        tensor3d_dims_t target_dims() const;
        targets_stats_t targets_stats(execution, tensor_size_t batch) const;
        tensor4d_cmap_t targets(tensor_range_t sample_range, tensor4d_t&) const;

        void undrop() const;
        void unshuffle() const;
        void drop(tensor_size_t feature) const;
        indices_t shuffle(tensor_size_t feature) const;

        // TODO: support for caching - all or selection of features

        tensor1d_t sample_weights(const targets_stats_t&) const;

        const auto& dataset() const { return m_dataset; }
        const auto& samples() const { return m_samples; }

    private:

        void update();

        const auto& bycolumn(tensor_size_t column) const
        {
            assert(column >= 0 && column < columns());
            return m_generators[static_cast<size_t>(m_column_mapping(column, 0))];
        }

        const auto& byfeature(tensor_size_t feature) const
        {
            assert(feature >= 0 && feature < features());
            return m_generators[static_cast<size_t>(m_feature_mapping(feature, 0))];
        }

        using column_mapping_t = tensor_mem_t<tensor_size_t, 2>;
        using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;
        using generator_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        const memory_dataset_t& m_dataset;              ///<
        indices_t               m_samples;              ///<
        rgenerators_t           m_generators;           ///<
        column_mapping_t        m_column_mapping;       ///<
        feature_mapping_t       m_feature_mapping;      ///<
        generator_mapping_t     m_generator_mapping;    ///<
    };
}
