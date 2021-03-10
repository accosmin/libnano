#pragma once

#include <nano/logger.h>
#include <nano/dataset/mask.h>
#include <nano/mlearn/feature.h>

namespace nano
{
    ///
    /// \brief per-feature statistics for continuous feature values
    ///     (e.g. useful for normalizing inputs).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct feature_scalar_stats_t
    {
        feature_scalar_stats_t() = default;

        feature_scalar_stats_t(tensor3d_dims_t dims) :
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
        feature_sclass_stats_t() = default;

        feature_sclass_stats_t(tensor_size_t classes) :
            m_class_counts(classes)
        {
            m_class_counts.zero();
        }

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
        feature_mclass_stats_t() = default;

        feature_mclass_stats_t(tensor_size_t classes) :
            m_class_counts(classes)
        {
            m_class_counts.zero();
        }

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

        ///
        /// \brief constructor.
        ///
        feature_storage_t(const feature_t& feature) :
            m_feature(feature)
        {
        }

        auto classes() const { return m_feature.classes(); }
        const auto& dims() const { return m_feature.dims(); }
        const auto& name() const { return m_feature.name(); }
        const feature_t& feature() const { return m_feature; }

        ///
        /// \brief set the feature value of a sample for a single-label categorical feature.
        ///
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

        ///
        /// \brief set the feature value of a sample for a multi-label categorical feature.
        ///
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

        ///
        /// \brief set the feature value of a sample for a continuous scalar or structured feature.
        ///
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

        ///
        /// \brief compute statistics from single-label categorical feature values.
        ///
        template <typename tscalar>
        auto stats(const tensor_cmap_t<tscalar, 1>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask) const
        {
            feature_sclass_stats_t stats{classes()};
            loop_masked(mask, samples, [&] (tensor_size_t, tensor_size_t sample)
            {
                const auto label = static_cast<tensor_size_t>(tensor(sample));

                stats.m_class_counts(label) ++;
            });
            return stats;
        }

        ///
        /// \brief compute statistics from multi-label categorical feature values.
        ///
        template <typename tscalar>
        auto stats(const tensor_cmap_t<tscalar, 2>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask) const
        {
            feature_mclass_stats_t stats{classes()};
            loop_masked(mask, samples, [&] (tensor_size_t, tensor_size_t sample)
            {
                stats.m_class_counts.array() += tensor.array(sample).template cast<tensor_size_t>();
            });
            return stats;
        }

        ///
        /// \brief compute statistics from continuous categorical feature values.
        ///
        template <typename tscalar>
        auto stats(const tensor_cmap_t<tscalar, 4>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask) const
        {
            feature_scalar_stats_t stats{dims()};
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
            else
            {
                stats.m_stdev.zero();
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
