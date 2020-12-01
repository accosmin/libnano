#pragma once

#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief
    ///
    class dataset_storage_t
    {
    public:

        using targets_t = tensor_mem_t<scalar_t, 4>;

        using tabular_f4_t = tensor_mem_t<float, 2>;
        using tabular_f8_t = tensor_mem_t<double, 2>;
        using tabular_u1_t = tensor_mem_t<uint8_t, 2>;
        using tabular_u2_t = tensor_mem_t<uint16_t, 2>;

        using structured_u1_t = tensor_mem_t<uint8_t, 4>;

        struct allocator_t
        {
            allocator_t() = default;

            allocator_t& samples(tensor_size_t samples)
            {
                m_samples = samples;
                return *this;
            }

            allocator_t& targets(tensor3d_dim_t tdim)
            {
                m_tdim = tdim;
                return *this;
            }

            allocator_t& targets(const feature_t& feature)
            {
                assert(!feature.optional());
                m_tdim = make_dims(feature.discrete() ? static_cast<tensor_size_t>(feature.labels.size()) : 1, 1, 1);
                return *this;
            }

            allocator_t& structured(tensor3d_dim_t idim)
            {
                m_idim = idim;
                return *this;
            }

            allocator_t& tabular(const features_t& features)
            {
                m_features = features;
                return *this;
            }

            tensor_size_t   m_samples{0};       ///<
            tensor3d_dim_t  m_idim{0, 0, 0};    ///<
            tensor3d_dim_t  m_tdim{0, 0, 0};    ///<
            features_t      m_features;         ///<
        };

        ///
        /// \brief default constructor
        ///
        dataset_storage_t() = default;

        // TODO: map tabular features to the associated storage chunk
        // TODO: accessors to the individual storage chunks (e.g. for features computation)

        const auto& targets() const { return m_targets; }
        const auto& structured_u1() const { return m_structured_u1; }
        const auto& tabular_f4() const { return m_tabular_f4; }

    private:

        static bool tabular_u1(size_t labels) { return labels < 0xFF; }
        static bool tabular_u2(size_t labels) { return labels < 0xFFFF; }

        static bool has_tabular(uint8_t value) { return value < 0xFF; }
        static bool has_tabular(uint16_t value) { return value < 0xFFFF; }
        static bool has_tabular(float value) { return std::isfinite(value); }
        static bool has_tabular(double value) { return std::isfinite(value); }

        // attributes
        targets_t           m_targets;          ///< optional targets (#samples, ...)
        structured_u1_t     m_structured_u1;    ///< 3D structured inputs (#samples, #channels, #rows, #columns)
        tabular_f4_t        m_tabular_f4;       ///< tabular continuous inputs (#samples, #features)
        tabular_f8_t        m_tabular_f8;       ///< tabular continuous inputs (#samples, #features)
        tabular_u1_t        m_tabular_u1;       ///< tabular discrete inputs with [2, 2^8-1) labels (#samples, #features)
        tabular_u2_t        m_tabular_u2;       ///< tabular discrete inputs with [2^8, 2^16-1) labels (#samples, #features)
    };
}
