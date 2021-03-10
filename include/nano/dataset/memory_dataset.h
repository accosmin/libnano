#pragma once

#include <nano/dataset/storage.h>
#include <nano/dataset/dataset.h>

namespace nano
{
    ///
    /// \brief in-memory dataset that efficiently stores a mixture of different input features.
    ///
    /// NB: the input features and the target feature can be optional.
    /// NB: the categorical features can be single-label or multi-label.
    /// NB: the continuous features can be structured (multi-dimensional) if feature_t::dims() != (1, 1, 1).
    ///
    class NANO_PUBLIC memory_dataset_t : public dataset_t
    {
    public:

        memory_dataset_t();

        task_type type() const override
        {
            return  has_target() ?
                    static_cast<task_type>(m_features[static_cast<size_t>(m_target)]) :
                    task_type::unsupervised;
        }

        tensor_size_t samples() const override
        {
            return m_samples;
        }

        tensor_size_t features() const
        {
            const auto total = m_storage_range.size<0>();
            return (m_target < total) ? (total - 1) : total;
        }

        const feature_t& feature(tensor_size_t feature) const
        {
            assert(feature >= 0 && feature < features());
            return m_features[static_cast<size_t>(feature >= m_target ? feature + 1 : feature)];
        }

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

        template <typename toperator>
        auto visit_target(const toperator& op) const
        {
            assert(has_target());
            return visit(m_target, op);
        }

        template <typename toperator>
        auto visit_inputs(tensor_size_t feature, const toperator& op) const
        {
            assert(feature >= 0 && feature < features());
            return visit(feature >= m_target ? feature + 1 : feature, op);
        }

        void resize(tensor_size_t samples, const features_t& features);
        void resize(tensor_size_t samples, const features_t& features, size_t target);

        feature_t target() const;
        tensor3d_dims_t target_dims() const;
        tensor4d_cmap_t targets(const indices_cmap_t& samples, tensor4d_t& buffer) const;

        rfeature_dataset_iterator_t feature_iterator(indices_t samples) const override;
        rflatten_dataset_iterator_t flatten_iterator(indices_t samples) const override;

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

        mask_cmap_t mask(tensor_size_t index)
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

            switch (feature.type())
            {
            case feature_type::sclass:  return
                (feature.classes() <= (tensor_size_t(1) << 8))  ? op(feature, m_storage_u08.slice(range), mask) :
                (feature.classes() <= (tensor_size_t(1) << 16)) ? op(feature, m_storage_u16.slice(range), mask) :
                (feature.classes() <= (tensor_size_t(1) << 32)) ? op(feature, m_storage_u32.slice(range), mask) :
                                                                  op(feature, m_storage_u64.slice(range), mask);
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
            return op(feature, m_storage_u08.slice(range), mask);
        }

        template <typename toperator>
        auto visit(tensor_size_t index, const toperator& op) const
        {
            const auto& feature = m_features[static_cast<size_t>(index)];

            const auto mask = this->mask(index);
            const auto [d0, d1, d2] = feature.dims();
            const auto range = make_range(m_storage_range(index, 0), m_storage_range(index, 1));

            switch (feature.type())
            {
            case feature_type::sclass:  return
                (feature.classes() <= (tensor_size_t(1) << 8))  ? op(feature, m_storage_u08.slice(range), mask) :
                (feature.classes() <= (tensor_size_t(1) << 16)) ? op(feature, m_storage_u16.slice(range), mask) :
                (feature.classes() <= (tensor_size_t(1) << 32)) ? op(feature, m_storage_u32.slice(range), mask) :
                                                                  op(feature, m_storage_u64.slice(range), mask);
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
            return op(feature, m_storage_u08.slice(range), mask);
        }

        template <typename tscalar>
        using storage_t = tensor_mem_t<tscalar, 2>;
        using storage_mask_t = tensor_mem_t<uint8_t, 2>;
        using storage_type_t = std::vector<feature_type>;
        using storage_range_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
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
