#pragma once

#include <nano/tensor.h>

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
}
