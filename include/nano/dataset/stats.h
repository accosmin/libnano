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
    class scalar_stats_t
    {
    public:

        scalar_stats_t() = default;

        explicit scalar_stats_t(tensor_size_t dims) :
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
            m_samples ++;
            m_mean.array() += array;
            m_stdev.array() += array.square();
            m_min.array() = m_min.array().min(array);
            m_max.array() = m_max.array().max(array);
            return *this;
        }

        auto& operator+=(const scalar_stats_t& other)
        {
            m_samples += other.m_samples;
            m_mean.array() += other.m_mean.array();
            m_stdev.array() += other.m_stdev.array();
            m_min.array() = m_min.array().min(other.m_min.array());
            m_max.array() = m_max.array().max(other.m_max.array());
            return *this;
        }

        auto& done()
        {
            if (m_samples > 1)
            {
                const auto N = m_samples;
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

        auto samples() const { return m_samples; }
        const auto& min() const { return m_min; }
        const auto& max() const { return m_max; }
        const auto& mean() const { return m_mean; }
        const auto& stdev() const { return m_stdev; }

    private:

        // attributes
        tensor_size_t   m_samples{0};       ///<
        tensor1d_t      m_min, m_max;       ///<
        tensor1d_t      m_mean, m_stdev;    ///<
    };

    ///
    /// \brief per-feature statistics for single-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    class sclass_stats_t
    {
    public:

        sclass_stats_t() = default;

        explicit sclass_stats_t(tensor_size_t classes) :
            m_class_counts(classes),
            m_class_weights(classes)
        {
            m_class_counts.zero();
            m_class_weights.zero();
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        auto& operator+=(tscalar label)
        {
            m_samples ++;
            m_class_counts(static_cast<tensor_size_t>(label)) ++;
            return *this;
        }

        auto& done()
        {
            m_class_weights.array() =
                static_cast<scalar_t>(m_samples) /
                static_cast<scalar_t>(m_class_counts.size()) /
                m_class_counts.array().cast<scalar_t>().max(1.0);
            return *this;
        }

        template <typename tscalar, size_t trank>
        static auto make(const feature_t& feature, dataset_iterator_t<tscalar, trank> it)
        {
            sclass_stats_t stats{feature.classes()};
            for (; it; ++ it)
            {
                if ([[maybe_unused]] const auto [index, given, label] = *it; given)
                {
                    stats += label;
                }
            }
            return stats.done();
        }

        template <typename tscalar, size_t trank>
        auto sample_weights(const feature_t& feature, dataset_iterator_t<tscalar, trank> it) const
        {
            tensor1d_t weights(it.size());
            if (feature.classes() != m_class_counts.size())
            {
                weights.zero();
            }
            else
            {
                scalar_t samples = 0;
                for (; it; ++ it)
                {
                    if (const auto [index, given, label] = *it; given)
                    {
                        samples += 1.0;
                        weights(index) = m_class_weights(static_cast<tensor_size_t>(label));
                    }
                    else
                    {
                        weights(index) = 0.0;
                    }
                }
                if (samples > 0)
                {
                    const auto scale = samples / static_cast<scalar_t>(weights.sum());
                    weights.array() *= scale;
                }
            }
            return weights;
        }

        auto samples() const { return m_samples; }
        auto classes() const { return m_class_counts.size(); }
        const auto& class_counts() const { return m_class_counts; }

    private:

        // attributes
        tensor_size_t   m_samples{0};       ///<
        indices_t       m_class_counts;     ///<
        tensor1d_t      m_class_weights;    ///<
    };

    ///
    /// \brief per-feature statistics for multi-label categorical feature values
    ///     (e.g. useful for handling unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    class mclass_stats_t
    {
    public:

        mclass_stats_t() = default;

        explicit mclass_stats_t(tensor_size_t classes) :
            m_class_counts(2 * classes),
            m_class_weights(2 * classes)
        {
            m_class_counts.zero();
            m_class_weights.zero();
        }

        template <template <typename, size_t> class tstorage, typename tscalar>
        auto& operator+=(const tensor_t<tstorage, tscalar, 1>& class_hits)
        {
            m_samples ++;
            m_class_counts(hash(class_hits)) ++;
            return *this;
        }

        auto& done()
        {
            m_class_weights.array() =
                static_cast<scalar_t>(m_samples) /
                static_cast<scalar_t>(m_class_counts.size()) /
                m_class_counts.array().cast<scalar_t>().max(1.0);
            return *this;
        }

        template <typename tscalar, size_t trank>
        static auto make(const feature_t& feature, dataset_iterator_t<tscalar, trank> it)
        {
            mclass_stats_t stats(feature.classes());
            for (; it; ++ it)
            {
                if ([[maybe_unused]] const auto [index, given, class_hits] = *it; given)
                {
                    stats += class_hits;
                }
            }
            return stats.done();
        }

        template <typename tscalar, size_t trank>
        auto sample_weights(const feature_t& feature, dataset_iterator_t<tscalar, trank> it) const
        {
            tensor1d_t weights(it.size());
            if (feature.classes() * 2 != m_class_counts.size())
            {
                weights.zero();
            }
            else
            {
                scalar_t samples = 0.0;
                for (; it; ++ it)
                {
                    if (const auto [index, given, class_hits] = *it; given)
                    {
                        samples += 1.0;
                        weights(index) = m_class_weights(hash(class_hits));
                    }
                    else
                    {
                        weights(index) = 0.0;
                    }
                }
                if (samples > 0)
                {
                    const auto scale = samples / static_cast<scalar_t>(weights.sum());
                    weights.array() *= scale;
                }
            }
            return weights;
        }

        auto samples() const { return m_samples; }
        auto classes() const { return m_class_counts.size() / 2; }
        const auto& class_counts() const { return m_class_counts; }

    private:

        template <template <typename, size_t> class tstorage, typename tscalar>
        static tensor_size_t hash(const tensor_t<tstorage, tscalar, 1>& class_hits)
        {
            const auto hits = class_hits.array().template cast<tensor_size_t>().sum();
            if (hits == 0)
            {
                return 0;
            }
            else if (hits == 1)
            {
                tensor_size_t coeff = 0;
                class_hits.array().maxCoeff(&coeff);
                return 1 + coeff;
            }
            else
            {
                return class_hits.size() + hits - 1;
            }
        }

        // attributes
        tensor_size_t   m_samples{0};       ///<
        indices_t       m_class_counts;     ///<
        tensor1d_t      m_class_weights;    ///<
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
