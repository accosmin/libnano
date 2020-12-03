#pragma once

#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief base class for storing the sample of a machine learning dataset.
    ///
    class dataset_storage_t
    {
    public:

        using targets_t = tensor_mem_t<scalar_t, 4>;

        dataset_storage_t() = default;

        void resize(tensor_size_t samples, tensor3d_dim_t tdim)
        {
            m_targets.resize(make_dims(samples, tdim));
        }

        auto samples() const
        {
            return m_targets.size<0>();
        }

        auto tdim() const
        {
            return make_dims(m_targets.size<1>(), m_targets.size<2>(), m_targets.size<3>());
        }

        const auto& targets() const
        {
            return m_targets;
        }

    private:

        // attributes
        targets_t           m_targets;          ///< optional targets (#samples, ...)
    };

    ///
    /// \brief efficient storage of tabular datasets:
    ///     - discrete features are stored using either 8-bit or 16-bit integers
    ///     - continuous features are stored as single precision floating points
    ///
    class tabular_dataset_storage_t : private dataset_storage_t
    {
    public:

        using tabular_f4_t = tensor_mem_t<float, 2>;
        using tabular_u1_t = tensor_mem_t<uint8_t, 2>;
        using tabular_u2_t = tensor_mem_t<uint16_t, 2>;

        using dataset_storage_t::tdim;
        using dataset_storage_t::samples;
        using dataset_storage_t::targets;

        tabular_dataset_storage_t() = default;

        void resize(tensor_size_t samples, const features_t& features, size_t target)
        {
            if (target == string_t::npos)
            {
                dataset_storage_t::resize(samples, make_dims(0, 0, 0));
            }
            else
            {
                assert(target < features.size());

                const auto& feature = features[target];
                assert(!feature.optional());

                dataset_storage_t::resize(resize, make_dims(static_cast<tensor_size_t>(feature.labels.size()), 1, 1));
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
}
