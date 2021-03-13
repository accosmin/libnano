#pragma once

#include <nano/arch.h>
#include <nano/factory.h>
#include <nano/dataset/storage.h>

namespace nano
{
    class memory_dataset_t;
    using dataset_factory_t = factory_t<memory_dataset_t>;
    using rmemory_dataset_t = dataset_factory_t::trobject;

    ///
    /// \brief machine learning dataset consisting of a collection of iid samples.
    ///
    /// NB: each sample consists of:
    ///     - a fixed number of (input) feature values and
    ///     - optionally a target if a supervised ML task.
    ///
    /// NB: the input features and the target feature can be optional.
    /// NB: the categorical features can be single-label or multi-label.
    /// NB: the continuous features can be structured (multi-dimensional) if feature_t::dims() != (1, 1, 1).
    ///
    class NANO_PUBLIC memory_dataset_t
    {
    public:

        ///
        /// \brief returns the available implementations.
        ///
        static dataset_factory_t& all();

        ///
        /// \brief default constructor
        ///
        memory_dataset_t() = default;

        ///
        /// \brief disable copying
        ///
        memory_dataset_t(const memory_dataset_t&) = default;
        memory_dataset_t& operator=(const memory_dataset_t&) = default;

        ///
        /// \brief enable moving
        ///
        memory_dataset_t(memory_dataset_t&&) noexcept = default;
        memory_dataset_t& operator=(memory_dataset_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        virtual ~memory_dataset_t() = default;

        ///
        /// \brief load dataset in memory.
        ///
        /// NB: any error is considered critical and an exception will be triggered.
        ///
        virtual void load() = 0;

        ///
        /// \brief returns the appropriate mathine learning task (by inspecting the target feature).
        ///
        task_type type() const;

        ///
        /// \brief returns the total number of samples.
        ///
        tensor_size_t samples() const
        {
            return m_samples;
        }

        ///
        /// \brief returns the samples that can be used for training.
        ///
        indices_t train_samples() const
        {
            return make_train_samples();
        }

        ///
        /// \brief returns the samples that should only be used for testing.
        ///
        /// NB: assumes a fixed set of test samples.
        ///
        indices_t test_samples() const
        {
            return make_test_samples();
        }

        ///
        /// \brief set all the samples for training.
        ///
        void no_testing()
        {
            m_testing.resize(samples());
            m_testing.zero();
        }

        ///
        /// \brief set the given range of samples for testing.
        ///
        /// NB: this accumulates the previous range of samples set for testing.
        ///
        void testing(tensor_range_t range)
        {
            if (m_testing.size() != samples())
            {
                m_testing.resize(samples());
                m_testing.zero();
            }

            assert(range.begin() >= 0 && range.end() <= m_testing.size());
            m_testing.vector().segment(range.begin(), range.size()).setConstant(1);
        }

        ///
        /// \brief returns the total number of features.
        ///
        tensor_size_t features() const
        {
            const auto total = m_storage_range.size<0>();
            return (m_target < total) ? (total - 1) : total;
        }

        ///
        /// \brief returns the feature at the given index.
        ///
        const feature_t& feature(tensor_size_t feature) const
        {
            assert(feature >= 0 && feature < features());
            return m_features[static_cast<size_t>(feature >= m_target ? feature + 1 : feature)];
        }

        ///
        /// \brief op(feature, tensor, mask)
        ///
        template <typename toperator>
        auto visit_target(const toperator& op) const
        {
            assert(has_target());
            return visit(m_target, op);
        }

        ///
        /// \brief
        ///
        template <typename toperator>
        auto visit_inputs(tensor_size_t feature, const toperator& op) const
        {
            assert(feature >= 0 && feature < features());
            return visit(feature >= m_target ? feature + 1 : feature, op);
        }

    protected:

        ///
        /// \brief allocate the dataset to storage the given number of samples and samples.
        ///
        /// NB: no target feature is given
        ///     and as such the dataset represents an unsupervised ML task.
        ///
        void resize(tensor_size_t samples, const features_t& features);

        ///
        /// \brief allocate the dataset to storage the given number of samples and samples.
        ///
        /// NB: the target feature is given as an index in the list of features
        ///     and as such the dataset represents a supervised ML task.
        ///
        void resize(tensor_size_t samples, const features_t& features, size_t target);

        ///
        /// \brief safely write a feature value for the given sample.
        ///
        template <typename tvalue>
        void set(tensor_size_t sample, tensor_size_t index, const tvalue& value)
        {
            assert(sample >= 0 && sample < m_samples);
            assert(index >= 0 && index < m_storage_range.size<0>());

            this->visit(index, [&] (const feature_t& feature, const auto& tensor, const auto& mask)
            {
                const auto setter = feature_storage_t{feature};
                setter.set(tensor, sample, value);
                setbit(mask, sample);
            });
        }

    private:

        bool has_target() const
        {
            return m_target < m_storage_range.size<0>();
        }

        mask_cmap_t tmask() const
        {
            assert(has_target());
            return this->mask(m_target);
        }

        mask_cmap_t imask(tensor_size_t feature) const
        {
            assert(feature >= 0 && feature < features());
            return this->mask(feature >= m_target ? feature + 1 : feature);
        }

        indices_t make_train_samples() const
        {
            const auto samples = this->samples();
            const auto has_testing = m_testing.size() == samples;
            return has_testing ? filter(samples - m_testing.vector().sum(), samples, 0) : arange(0, samples);
        }

        indices_t make_test_samples() const
        {
            const auto samples = this->samples();
            const auto has_testing = m_testing.size() == samples;
            return has_testing ? filter(m_testing.vector().sum(), samples, 1) : indices_t{};
        }

        indices_t filter(tensor_size_t count, tensor_size_t samples, tensor_size_t condition) const
        {
            indices_t indices(count);
            for (tensor_size_t sample = 0, index = 0; sample < samples; ++ sample)
            {
                if (m_testing(sample) == condition)
                {
                    assert(index < indices.size());
                    indices(index ++) = sample;
                }
            }
            return indices;
        }

        mask_map_t mask(tensor_size_t index)
        {
            return m_storage_mask.tensor(index);
        }

        mask_cmap_t mask(tensor_size_t index) const
        {
            return m_storage_mask.tensor(index);
        }

        template <typename toperator>
        auto visit(tensor_size_t index, const toperator& op)
        {
            const auto& feature = m_features[static_cast<size_t>(index)];

            const auto mask = this->mask(index);
            const auto [d0, d1, d2] = feature.dims();
            const auto range = make_range(m_storage_range(index, 0), m_storage_range(index, 1));

            static const auto maxu08 = tensor_size_t(1) << 8;
            static const auto maxu16 = tensor_size_t(1) << 16;
            static const auto maxu32 = tensor_size_t(1) << 32;

            switch (feature.type())
            {
            case feature_type::sclass:  return                                  (feature.classes() <= maxu08) ?
                    op(feature, m_storage_u08.slice(range).reshape(-1), mask) : (feature.classes() <= maxu16) ?
                    op(feature, m_storage_u16.slice(range).reshape(-1), mask) : (feature.classes() <= maxu32) ?
                    op(feature, m_storage_u32.slice(range).reshape(-1), mask) :
                    op(feature, m_storage_u64.slice(range).reshape(-1), mask);
            case feature_type::mclass:  return op(feature, m_storage_u08.slice(range).reshape(m_samples, -1), mask);
            case feature_type::float32: return op(feature, m_storage_f32.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::float64: return op(feature, m_storage_f64.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::int8:    return op(feature, m_storage_i08.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::int16:   return op(feature, m_storage_i16.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::int32:   return op(feature, m_storage_i32.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::int64:   return op(feature, m_storage_i64.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::uint8:   return op(feature, m_storage_u08.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::uint16:  return op(feature, m_storage_u16.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::uint32:  return op(feature, m_storage_u32.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::uint64:  return op(feature, m_storage_u64.slice(range).reshape(m_samples, d0, d1, d2), mask);
            default:                    critical0("in-memory dataset: unhandled feature type (", feature.type(), ")!");
            }
            return op(feature, m_storage_u08.slice(range).reshape(-1), mask);
        }

        template <typename toperator>
        auto visit(tensor_size_t index, const toperator& op) const
        {
            const auto& feature = m_features[static_cast<size_t>(index)];

            const auto mask = this->mask(index);
            const auto [d0, d1, d2] = feature.dims();
            const auto range = make_range(m_storage_range(index, 0), m_storage_range(index, 1));

            static const auto maxu08 = tensor_size_t(1) << 8;
            static const auto maxu16 = tensor_size_t(1) << 16;
            static const auto maxu32 = tensor_size_t(1) << 32;

            switch (feature.type())
            {
            case feature_type::sclass:  return                                  (feature.classes() <= maxu08) ?
                    op(feature, m_storage_u08.slice(range).reshape(-1), mask) : (feature.classes() <= maxu16) ?
                    op(feature, m_storage_u16.slice(range).reshape(-1), mask) : (feature.classes() <= maxu32) ?
                    op(feature, m_storage_u32.slice(range).reshape(-1), mask) :
                    op(feature, m_storage_u64.slice(range).reshape(-1), mask);
            case feature_type::mclass:  return op(feature, m_storage_u08.slice(range).reshape(m_samples, -1), mask);
            case feature_type::float32: return op(feature, m_storage_f32.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::float64: return op(feature, m_storage_f64.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::int8:    return op(feature, m_storage_i08.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::int16:   return op(feature, m_storage_i16.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::int32:   return op(feature, m_storage_i32.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::int64:   return op(feature, m_storage_i64.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::uint8:   return op(feature, m_storage_u08.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::uint16:  return op(feature, m_storage_u16.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::uint32:  return op(feature, m_storage_u32.slice(range).reshape(m_samples, d0, d1, d2), mask);
            case feature_type::uint64:  return op(feature, m_storage_u64.slice(range).reshape(m_samples, d0, d1, d2), mask);
            default:                    critical0("in-memory dataset: unhandled feature type (", feature.type(), ")!");
            }
            return op(feature, m_storage_u08.slice(range).reshape(-1), mask);
        }

        template <typename tscalar>
        using storage_t = tensor_mem_t<tscalar, 2>;
        using storage_mask_t = tensor_mem_t<uint8_t, 2>;
        using storage_type_t = std::vector<feature_type>;
        using storage_range_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        indices_t               m_testing;      ///< (#samples,) - mark sample for testing, if != 0
        features_t              m_features;     ///<
        tensor_size_t           m_target{0};    ///<
        tensor_size_t           m_samples{0};   ///<
        storage_t<float>        m_storage_f32;  ///<
        storage_t<double>       m_storage_f64;  ///<
        storage_t<int8_t>       m_storage_i08;  ///<
        storage_t<int16_t>      m_storage_i16;  ///<
        storage_t<int32_t>      m_storage_i32;  ///<
        storage_t<int64_t>      m_storage_i64;  ///<
        storage_t<uint8_t>      m_storage_u08;  ///<
        storage_t<uint16_t>     m_storage_u16;  ///<
        storage_t<uint32_t>     m_storage_u32;  ///<
        storage_t<uint64_t>     m_storage_u64;  ///<
        storage_mask_t          m_storage_mask; ///< feature value given if the bit (feature, sample) is 1
        storage_type_t          m_storage_type; ///<
        storage_range_t         m_storage_range;///<
    };
}
