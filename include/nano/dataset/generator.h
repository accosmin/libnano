#pragma once

#include <variant>
#include <nano/factory.h>
#include <nano/dataset/dataset.h>

namespace nano
{
    class generator_t;
    using generator_factory_t = factory_t<generator_t>;
    using rgenerator_t = generator_factory_t::trobject;
    using rgenerators_t = std::vector<rgenerator_t>;

    // single-label categorical feature values: (sample index) = label/class index
    using sclass_mem_t = tensor_mem_t<int32_t, 1>;
    using sclass_cmap_t = tensor_cmap_t<int32_t, 1>;

    // multi-label categorical feature values: (sample index, label/class index) = 0 or 1
    using mclass_mem_t = tensor_mem_t<int8_t, 2>;
    using mclass_cmap_t = tensor_cmap_t<int8_t, 2>;

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
        /// \brief returns the available implementations.
        ///
        static generator_factory_t& all();

        ///
        /// \brief constructor.
        ///
        generator_t(const memory_dataset_t& dataset);

        ///
        /// \brief default destructor.
        ///
        virtual ~generator_t() = default;

        ///
        /// \brief prepare the given samples to generate features fast when needed.
        ///
        virtual void preprocess(execution, indices_cmap_t samples);

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
        virtual void undrop() = 0;
        virtual void drop(tensor_size_t feature) = 0;

        ///
        /// \brief toggle sample permutation of features, useful for feature importance analysis.
        ///
        virtual void unshuffle() = 0;
        virtual void shuffle(tensor_size_t feature) = 0;

        ///
        /// \brief computes the values of the given feature and samples,
        ///     useful for training and evaluating ML models that perform feature selection
        ///     (e.g. gradient boosting).
        ///
        virtual sclass_cmap_t select(tensor_size_t feature, indices_cmap_t samples, sclass_mem_t&) const;
        virtual mclass_cmap_t select(tensor_size_t feature, indices_cmap_t samples, mclass_mem_t&) const;
        virtual scalar_cmap_t select(tensor_size_t feature, indices_cmap_t samples, scalar_mem_t&) const;
        virtual struct_cmap_t select(tensor_size_t feature, indices_cmap_t samples, struct_mem_t&) const;

        ///
        /// \brief computes the values of all features for the given samples,
        ///     useful for training and evaluating ML model that map densely continuous inputs to targets
        ///     (e.g. linear models, MLPs).
        ///
        virtual tensor_size_t columns() const = 0;
        virtual void flatten(indices_cmap_t samples, tensor2d_map_t, tensor_size_t column_offset) const = 0;

        ///
        /// \brief map the given column to its feature index.
        ///
        virtual tensor_size_t column2feature(tensor_size_t column) const = 0;

        ///
        /// \brief access functions
        ///
        const auto& dataset() const { return m_dataset; }

    private:

        // attributes
        const memory_dataset_t& m_dataset;  ///<
    };

    struct select_stats_t
    {
        indices_t       m_sclass_features;  ///< indices of the single-label features
        indices_t       m_mclass_features;  ///< indices of the multi-label features
        indices_t       m_scalar_features;  ///< indices of the scalar features
        indices_t       m_struct_features;  ///< indices of structured features
    };

    struct scalar_stats_t
    {
        scalar_stats_t() = default;

        scalar_stats_t(tensor_size_t dims) :
            m_min(dims),
            m_max(dims),
            m_mean(dims),
            m_stdev(dims)
        {
            m_mean.zero();
            m_stdev.zero();
            m_min.constant(std::numeric_limits<scalar_t>::max());
            m_max.constant(std::numeric_limits<scalar_t>::lowest());
        }

        template <typename tarray>
        auto& operator+=(const tarray& array)
        {
            m_count ++;
            m_mean.array() += array;
            m_stdev.array() += array.square();
            m_min.array() = m_min.array().min(array);
            m_max.array() = m_max.array().max(array);
            return *this;
        }

        auto& operator+=(const scalar_stats_t& other)
        {
            m_count += other.m_count;
            m_mean.array() += other.m_mean.array();
            m_stdev.array() += other.m_stdev.array();
            m_min.array() = m_min.array().min(other.m_min.array());
            m_max.array() = m_max.array().max(other.m_max.array());
            return *this;
        }

        auto& done()
        {
            if (m_count > 1)
            {
                const auto N = m_count;
                m_stdev.array() = ((m_stdev.array() - m_mean.array().square() / N) / (N - 1)).sqrt();
                m_mean.array() /= static_cast<scalar_t>(N);
            }
            else
            {
                m_stdev.zero();
            }
            return *this;
        }

        tensor_size_t   m_count{0};         ///<
        tensor1d_t      m_min, m_max;       ///<
        tensor1d_t      m_mean, m_stdev;    ///<
    };

    struct sclass_stats_t
    {
        sclass_stats_t() = default;

        sclass_stats_t(tensor_size_t classes) :
            m_class_counts(classes)
        {
            m_class_counts.zero();
        }

        template <typename tscalar>
        auto& operator+=(tscalar label)
        {
            m_class_counts(static_cast<tensor_size_t>(label)) ++;
            return *this;
        }

        template <template <typename, size_t> class tstorage, typename tscalar>
        auto& operator+=(const tensor_t<tstorage, tscalar, 1>& class_hits)
        {
            m_class_counts.array() += class_hits.array().template cast<tensor_size_t>();
            return *this;
        }

        indices_t       m_class_counts;     ///<
    };

    using flatten_stats_t = scalar_stats_t;
    using targets_stats_t = std::variant<scalar_stats_t, sclass_stats_t>;

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

            auto generator = std::make_unique<tgenerator>(m_dataset, args...);
            generator->preprocess(ex, m_samples);
            m_generators.push_back(std::move(generator));
            update();
            return *this;
        }

        tensor_size_t features() const;
        feature_t feature(tensor_size_t feature) const;

        select_stats_t select_stats(execution) const;
        sclass_cmap_t select(tensor_size_t feature, indices_cmap_t samples, sclass_mem_t&) const;
        mclass_cmap_t select(tensor_size_t feature, indices_cmap_t samples, mclass_mem_t&) const;
        scalar_cmap_t select(tensor_size_t feature, indices_cmap_t samples, scalar_mem_t&) const;
        struct_cmap_t select(tensor_size_t feature, indices_cmap_t samples, struct_mem_t&) const;

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
        void shuffle(tensor_size_t feature) const;

        // TODO: support for drop column and sample permutation!
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

        // attributes
        const memory_dataset_t& m_dataset;          ///<
        indices_t               m_samples;          ///<
        rgenerators_t           m_generators;       ///<
        column_mapping_t        m_column_mapping;   ///<
        feature_mapping_t       m_feature_mapping;  ///<
    };
}
