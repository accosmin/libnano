#pragma once

#include <variant>
#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief
    ///
    class feature_storage_t
    {
    public:

        using storage_t = std::variant
        <
            // continuous
            tensor_mem_t<float, 4>,
            tensor_mem_t<double, 4>,
            tensor_mem_t<uint8_t, 4>,
            tensor_mem_t<uint16_t, 4>,
            tensor_mem_t<uint32_t, 4>,
            tensor_mem_t<uint64_t, 4>,
            tensor_mem_t<int8_t, 4>,
            tensor_mem_t<int16_t, 4>,
            tensor_mem_t<int32_t, 4>,
            tensor_mem_t<int64_t, 4>,

            // discrete
            tensor_mem_t<uint8_t, 1>,
            tensor_mem_t<uint16_t, 1>,
        >;

        feature_storage_t() = default;

        feature_storage_t(feature_t, tensor_size_t samples);

        tensor_size_t samples() const;
        const auto& feature() const { return m_feature; }
        const auto& storage() const { return m_storage; }

    private:

        // attributes
        feature_t       m_feature;  ///<
        storage_t       m_storage;  ///<
    };

    ///
    /// \brief
    ///
    class memory_dataset_t : public dataset_t
    {
    public:

        memory_dataset_t();

        feature_t target() const override;
        tensor_size_t samples() const override;
        tensor3d_dims_t tdims() const override;

        const auto& istorage() const { return m_inputs; }
        const auto& tstorage() const { return m_target; }

        void resize(tensor_size_t samples, const features_t& features, size_t target);

    private:

        using target_t = feature_storage_t;
        using inputs_t = std::vector<feature_storage_t>;

        // attributes
        target_t        m_target;   ///<
        inputs_t        m_inputs;   ///<
    };

    ///
    /// \brief tabular datasets mixes features of different types:
    ///     - discrete features stored using either 8-bit or 16-bit integers
    ///     - continuous features stored as single or double precision floating points
    ///
    /// NB: the input features can be optional.
    /// NB: the discrete input features cannot be multi-class.
    /// NB: the continuous input features cannot be multi-dimensional (::dims() == (1, 1, 1)).
    ///
    class tabular_dataset_t : private dataset_t
    {
    public:

        using tabular_u8_t = tensor_mem_t<uint8_t, 2>;
        using tabular_u16_t = tensor_mem_t<uint16_t, 2>;

        using tabular_f32_t = tensor_mem_t<float, 2>;
        using tabular_f64_t = tensor_mem_t<double, 2>;

        using dataset_t::tdim;
        using dataset_t::target;
        using dataset_t::samples;
        using dataset_t::targets;

        tabular_dataset_t() = default;

        void resize(tensor_size_t samples, const features_t& features, size_t target)
        {
            if (target == string_t::npos)
            {
                dataset_t::resize(samples, feature_t{});
            }
            else
            {
                assert(target < features.size());

                const auto& feature = features[target];
                assert(!feature.optional());

                dataset_t::resize(resize, feature);
            }

            features_t u1_features, u2_features, f4_features;
            for (const auto& feature : features)
            {
                if (feature.discrete())
                {
                    if (tabular_u1(features.labels().size()))
                    {
                        u1_features.append(feature);
                    }
                    else if (tabular_u2(features.labels().size()))
                    {
                        u2_features.append(feature);
                    }
                    else
                    {
                        assert(false);
                    }
                }
                else
                {
                    f4_features.append(feature);
                }
            }

            m_features = u1_features;
            m_features.insert(m_features.end(), u2_features.begin(), u2_features.end());
            m_features.insert(m_features.end(), f4_features.begin(), f4_features.end());

            m_inputs_u1.resize(samples, static_cast<tensor_size_t>(u1_features));
            m_inputs_u2.resize(samples, static_cast<tensor_size_t>(u2_features));
            m_inputs_f4.resize(samples, static_cast<tensor_size_t>(f4_features));
        }

        // TODO: map tabular features to the associated storage chunk
        // TODO: accessors to the individual storage chunks (e.g. for features computation)

        const auto& targets() const { return m_targets; }

    private:

        static bool tabular_u1(size_t labels) { return labels < 0xFF; }
        static bool tabular_u2(size_t labels) { return labels < 0xFFFF; }

        static bool has_tabular(uint8_t value) { return value < 0xFF; }
        static bool has_tabular(uint16_t value) { return value < 0xFFFF; }
        static bool has_tabular(float value) { return std::isfinite(value); }

        // attributes
        features_t          m_features;     ///< input features in the order (u1, u2, f4)
        tabular_u1_t        m_inputs_u1;    ///< tabular discrete inputs with [2, 2^8-1) labels (#samples, #features)
        tabular_u2_t        m_inputs_u2;    ///< tabular discrete inputs with [2^8, 2^16-1) labels (#samples, #features)
        tabular_f4_t        m_inputs_f4;    ///< tabular continuous inputs (#samples, #features)
    };

    ///
    /// \brief fixed size, non-optional, continuous
    ///
    template <typename tscalar, size_t tfeature_rank>
    class tensor_dataset_t<tscalar, tfeature_rank>
    {
    public:

        using inputs_t = tensor_mem_t<tscalar, tfeature_rank + 1>;

        tensor_dataset_t() = default;

        void resize(tensor_size_t samples, feature_t input, feature_t target)
        {

        }


    private:

        // attributes
        inputs_t            m_inputs;       ///< inputs (#samples, ...)
    };
}
