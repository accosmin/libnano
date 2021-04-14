#pragma once

#include <variant>
#include <nano/dataset/feature.h>
#include <nano/dataset/iterator.h>

namespace nano
{
    ///
    /// \brief
    ///
    struct select_stats_t
    {
        indices_t       m_sclass_features;  ///< indices of the single-label features
        indices_t       m_mclass_features;  ///< indices of the multi-label features
        indices_t       m_scalar_features;  ///< indices of the scalar features
        indices_t       m_struct_features;  ///< indices of structured features
    };

    ///
    /// \brief per-feature statistics for continuous feature values
    ///     (e.g. useful for normalizing inputs and targets).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
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
            m_min.full(std::numeric_limits<scalar_t>::max());
            m_max.full(std::numeric_limits<scalar_t>::lowest());
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

        template <typename tscalar, size_t trank>
        static auto make(const feature_t& feature, dataset_iterator_t<tscalar, trank> it)
        {
            scalar_stats_t stats{size(feature.dims())};
            for (; it; ++ it)
            {
                if ([[maybe_unused]] const auto [index, given, values] = *it; given)
                {
                    stats += values.array().template cast<scalar_t>();
                }
            }
            return stats.done();
        }

        tensor_size_t   m_count{0};         ///<
        tensor1d_t      m_min, m_max;       ///<
        tensor1d_t      m_mean, m_stdev;    ///<
    };

    ///
    /// \brief per-feature statistics for single-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct sclass_stats_t
    {
        sclass_stats_t() = default;

        sclass_stats_t(tensor_size_t classes) :
            m_class_counts(classes)
        {
            m_class_counts.zero();
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
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

        template <typename tscalar, size_t trank>
        static auto make(const feature_t& feature, dataset_iterator_t<tscalar, trank> it)
        {
            sclass_stats_t stats{feature.classes()};
            for (; it; ++ it)
            {
                if ([[maybe_unused]] const auto [index, given, values] = *it; given)
                {
                    stats += values;
                }
            }
            return stats;
        }

        indices_t       m_class_counts;     ///<
    };

    ///
    /// \brief per-feature statistics for multi-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct mclass_stats_t
    {
        mclass_stats_t() = default;

        template <template <typename, size_t> class tstorage, typename tscalar>
        static auto hash(const tensor_t<tstorage, tscalar, 1>& class_hits)
        {
            string_t str(static_cast<size_t>(class_hits.size()), '0');
            for (tensor_size_t i = 0, size = class_hits.size(); i < size; ++ i)
            {
                str[static_cast<size_t>(i)] = (class_hits(i) == tscalar(0)) ? '0' : '1';
            }
            return str;
        }

        template <template <typename, size_t> class tstorage, typename tscalar>
        auto& operator+=(const tensor_t<tstorage, tscalar, 1>& class_hits)
        {
            m_class_counts[hash(class_hits)] ++;
            return *this;
        }

        template <typename tscalar, size_t trank>
        static auto make(const feature_t&, dataset_iterator_t<tscalar, trank> it)
        {
            mclass_stats_t stats;
            for (; it; ++ it)
            {
                if ([[maybe_unused]] const auto [index, given, values] = *it; given)
                {
                    stats += values;
                }
            }
            return stats;
        }

        using class_counts_t = std::unordered_map<string_t, tensor_size_t>;

        class_counts_t  m_class_counts;     /// (representation of class hits, count)
    };

    ///
    /// \brief per-column statistics for flatten feature values.
    ///
    using flatten_stats_t = scalar_stats_t;

    ///
    /// \brief
    ///
    using targets_stats_t = std::variant
    <
        std::monostate,
        scalar_stats_t,
        sclass_stats_t,
        mclass_stats_t
    >;
}
