#pragma once

#include <nano/dataset/feature.h>
#include <nano/dataset/iterator.h>

namespace nano
{
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

        template <typename tscalar>
        static auto make(const feature_t& feature, feature_iterator_t<tscalar, 1> iterator)
        {
            feature_sclass_stats_t stats{feature.classes()};
            for ( ; iterator; ++ iterator)
            {
                if (const auto [index, given, label] = *iterator; given)
                {
                    stats.m_class_counts(static_cast<tensor_size_t>(label)) ++;
                }
            }
            return stats;
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

        template <typename tscalar>
        static auto make(const feature_t& feature, feature_iterator_t<tscalar, 2> iterator)
        {
            feature_mclass_stats_t stats{feature.classes()};
            for ( ; iterator; ++ iterator)
            {
                if (const auto [index, given, values] = *iterator; given)
                {
                    stats.m_class_counts.array() += values.array().template cast<tensor_size_t>();
                }
            }
            return stats;
        }

        indices_t       m_class_counts{0};  ///< number of samples per class (label)
    };

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

        template <typename tscalar>
        static auto make(const feature_t& feature, feature_iterator_t<tscalar, 4> iterator)
        {
            feature_scalar_stats_t stats{feature.dims()};
            for ( ; iterator; ++ iterator)
            {
                if (const auto [index, given, values] = *iterator; given)
                {
                    const auto array = values.array().template cast<scalar_t>();
                    stats.m_count ++;
                    stats.m_mean.array() += array;
                    stats.m_stdev.array() += array.square();
                    stats.m_min.array() = stats.m_min.array().min(array);
                    stats.m_max.array() = stats.m_max.array().max(array);
                }
            }
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

        tensor_size_t   m_count{0};         ///< total number of samples
        tensor3d_t      m_min, m_max;       ///<
        tensor3d_t      m_mean, m_stdev;    ///<
    };
}
