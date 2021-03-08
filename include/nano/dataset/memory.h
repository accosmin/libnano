#pragma once

#include <nano/logger.h>
#include <nano/mlearn/feature.h>

namespace nano
{
    // continuous feature: (sample, dim1, dim2, dim3)
    template <typename tscalar>
    using scalar_storage_t = tensor_mem_t<tscalar, 4>;

    // single-label categorical feature: (sample) = label index
    template <typename tscalar>
    using sclass_storage_t = tensor_mem_t<tscalar, 1>;

    // multi-label categorical feature: (sample, label_index) = 1 if the label index is active, otherwise 0
    template <typename tscalar>
    using mclass_storage_t = tensor_mem_t<tscalar, 2>;

    // bitwise mask for a feature: (sample) = 1 if the feature value is available, otherwise 0
    using mask_t = tensor_mem_t<uint8_t, 1>;
    using mask_map_t = tensor_map_t<uint8_t, 1>;
    using mask_cmap_t = tensor_cmap_t<uint8_t, 1>;

    ///
    /// \brief allocate and initialize a tensor bitmask where the last dimension is the number of samples.
    ///
    template <size_t trank>
    inline auto make_mask(const tensor_dims_t<trank>& dims)
    {
        const auto samples = std::get<trank - 1>(dims);

        auto bit_dims = dims;
        bit_dims[trank - 1] = (samples + 7) / 8;

        tensor_mem_t<uint8_t, trank> mask(bit_dims);
        mask.zero();
        return mask;
    }

    ///
    /// \brief mark a feature value as set for a particular sample.
    ///
    inline void setbit(const mask_map_t& mask, tensor_size_t sample)
    {
        assert(sample >= 0 && sample < (8 * mask.size()));
        mask(sample / 8) |= static_cast<uint8_t>(0x01 << (7 - (sample % 8)));
    }

    ///
    /// \brief check if a feature value exists for a particular sample.
    ///
    inline bool getbit(const mask_cmap_t& mask, tensor_size_t sample)
    {
        assert(sample >= 0 && sample < (8 * mask.size()));
        return (mask(sample / 8) & (0x01 << (7 - (sample % 8)))) != 0x00;
    }

    ///
    /// \brief returns true if the feature is optional (aka some samples haven't been set).
    ///
    inline bool optional(const mask_cmap_t& mask, tensor_size_t samples)
    {
        const auto bytes = samples / 8;
        for (tensor_size_t byte = 0; byte < bytes; ++ byte)
        {
            if (mask(byte) != 0xFF)
            {
                return true;
            }
        }
        for (tensor_size_t sample = 8 * bytes; sample < samples; ++ sample)
        {
            if (!getbit(mask, sample))
            {
                return true;
            }
        }
        return false;
    }

    ///
    /// \brief call the given operator for all samples that have feature values associated with.
    ///
    template <typename toperator>
    void loop_masked(const mask_cmap_t& mask, const indices_t& samples, const toperator& op)
    {
        for (tensor_size_t i = 0; i < samples.size(); ++ i)
        {
            const auto sample = samples(i);
            if (getbit(mask, sample))
            {
                op(i, sample);
            }
        }
    }

    ///
    /// \brief per-feature statistics for continuous feature values
    ///     (e.g. useful for normalizing inputs).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct feature_scalar_stats_t
    {
        tensor_size_t   m_count{0};         ///< total number of samples
        tensor3d_t      m_min, m_max;       ///<
        tensor3d_t      m_mean, m_stdev;    ///<
    };

    ///
    /// \brief per-feature statistics for single-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct feature_sclass_stats_t
    {
        indices_t       m_class_counts{0};  ///< number of samples per class (label)
    };

    ///
    /// \brief per-feature statistics for mult-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct feature_mclass_stats_t
    {
        indices_t       m_class_counts{0};  ///< number of samples per class (label)
    };

    ///
    /// \brief utility to safely access feature values.
    ///
    /// a feature value to write can be of a variety of types:
    ///     - a scalar,
    ///     - a label index (if single-label categorical),
    ///     - a label hit vector (if multi-label categorical),
    ///     - a 3D tensor (if structured continuous) or
    ///     - a string.
    ///
    class feature_storage_t
    {
    public:

        feature_storage_t(const feature_t& feature) :
            m_feature(feature)
        {
        }

        auto classes() const { return m_feature.classes(); }
        const auto& dims() const { return m_feature.dims(); }
        const auto& name() const { return m_feature.name(); }
        const feature_t& feature() const { return m_feature; }

        template <typename tscalar, typename tvalue>
        void set(const tensor_map_t<tscalar, 1>& tensor, tensor_size_t sample, const tvalue& value) const
        {
            tensor_size_t label;
            if constexpr (std::is_same<tvalue, string_t>::value)
            {
                label = check_from_string<tensor_size_t>("single-label", value);
            }
            else if constexpr (std::is_arithmetic<tvalue>::value)
            {
                label = static_cast<tensor_size_t>(value);
            }
            else
            {
                critical0("in-memory dataset: cannot set single-label feature <", name(), ">!");
            }

            critical(
                label < 0 || label >= classes(),
                "in-memory dataset: cannot set single-label feature <", name(),
                ">: invalid label ", label, " not in [0, ", classes(), ")!");

            tensor(sample) = static_cast<tscalar>(label);
        }

        template <typename tscalar, typename tvalue>
        void set(const tensor_map_t<tscalar, 2>& tensor, tensor_size_t sample, const tvalue& value) const
        {
            if constexpr (::nano::is_tensor<tvalue>::value)
            {
                if constexpr (tvalue::rank() == 1)
                {
                    critical(
                        value.size() != classes(),
                        "in-memory dataset: cannot set multi-label feature <", name(),
                        ">: invalid number of labels ", value.size(), " vs. ", classes(), "!");

                    tensor.vector(sample) = value.vector().template cast<tscalar>();
                }
                else
                {
                    critical0("in-memory dataset: cannot set multi-label feature <", name(), ">!");
                }
            }
            else
            {
                critical0("in-memory dataset: cannot set multi-label feature <", name(), ">!");
            }
        }

        template <typename tscalar, typename tvalue>
        void set(const tensor_map_t<tscalar, 4>& tensor, tensor_size_t sample, const tvalue& value) const
        {
            if constexpr (std::is_same<tvalue, string_t>::value)
            {
                critical(
                    ::nano::size(dims()) != 1,
                    "in-memory dataset: cannot set scalar feature <", name(),
                    ">: invalid tensor dimensions ", dims(), "!");

                tensor(sample) = check_from_string<tscalar>("scalar", value);
            }
            else if constexpr (std::is_arithmetic<tvalue>::value)
            {
                critical(
                    ::nano::size(dims()) != 1,
                    "in-memory dataset: cannot set scalar feature <", name(),
                    ">: invalid tensor dimensions ", dims(), "!");

                tensor(sample) = static_cast<tscalar>(value);
            }
            else if constexpr (::nano::is_tensor<tvalue>())
            {
                critical(
                    ::nano::size(dims()) != value.size(),
                    "in-memory dataset: cannot set scalar feature <", name(),
                    ">: invalid tensor dimensions ", dims(), " vs. ", value.dims(), "!");

                tensor.vector(sample) = value.vector().template cast<tscalar>();
            }
            else
            {
                critical0("in-memory dataset: cannot set scalar feature <", name(), ">!");
            }
        }

        template <typename tscalar, size_t trank>
        auto scalar_stats(const tensor_cmap_t<tscalar, trank>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask) const
        {
            feature_scalar_stats_t stats;

            if constexpr (trank == 4)
            {
                stats.m_min.resize(dims());
                stats.m_max.resize(dims());
                stats.m_mean.resize(dims());
                stats.m_stdev.resize(dims());

                stats.m_count = 0;
                stats.m_mean.zero();
                stats.m_stdev.zero();
                stats.m_min.constant(std::numeric_limits<scalar_t>::max());
                stats.m_max.constant(std::numeric_limits<scalar_t>::lowest());

                loop_masked(mask, samples, [&] (tensor_size_t, tensor_size_t sample)
                {
                    const auto values = tensor.array(sample).template cast<scalar_t>();

                    stats.m_count ++;
                    stats.m_mean.array() += values;
                    stats.m_stdev.array() += values.square();
                    stats.m_min.array() = stats.m_min.array().min(values);
                    stats.m_max.array() = stats.m_max.array().max(values);
                });

                if (stats.m_count > 1)
                {
                    const auto N = stats.m_count;
                    stats.m_stdev.array() = ((stats.m_stdev.array() - stats.m_mean.array().square() / N) / (N - 1)).sqrt();
                    stats.m_mean.array() /= static_cast<scalar_t>(N);
                }
            }
            else
            {
                critical0("in-memory dataset: cannot access scalar feature <", name(), ">!");
            }
            return stats;
        }

        template <typename tscalar, size_t trank>
        auto sclass_stats(const tensor_cmap_t<tscalar, trank>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask) const
        {
            feature_sclass_stats_t stats;

            if constexpr (trank == 1)
            {
                stats.m_class_counts.resize(static_cast<tensor_size_t>(m_feature.labels().size()));
                stats.m_class_counts.zero();

                loop_masked(mask, samples, [&] (tensor_size_t, tensor_size_t sample)
                {
                    const auto label = static_cast<tensor_size_t>(tensor(sample));

                    stats.m_class_counts(label) ++;
                });
            }
            else
            {
                critical0("in-memory dataset: cannot access single-label feature <", name(), ">!");
            }
            return stats;
        }

        template <typename tscalar, size_t trank>
        auto mclass_stats(const tensor_cmap_t<tscalar, trank>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask) const
        {
            feature_mclass_stats_t stats;

            if constexpr (trank == 2)
            {
                stats.m_class_counts.resize(static_cast<tensor_size_t>(m_feature.labels().size()));
                stats.m_class_counts.zero();

                loop_masked(mask, samples, [&] (tensor_size_t, tensor_size_t sample)
                {
                    stats.m_class_counts.array() += tensor.array(sample).template cast<tensor_size_t>();
                });
            }
            else
            {
                critical0("in-memory dataset: cannot access multi-label feature <", name(), ">!");
            }
            return stats;
        }

    private:

        template <typename tscalar>
        auto check_from_string(const char* type, const string_t& value) const
        {
            tscalar scalar;
            try
            {
                scalar = ::nano::from_string<tscalar>(value);
            }
            catch (std::exception& e)
            {
                critical0(
                    "in-memory dataset: cannot set ", type, " feature <", name(),
                    ">: caught exception <", e.what(), ">!");
            }
            return scalar;
        }

        // attributes
        const feature_t&    m_feature;  ///<
    };
}
