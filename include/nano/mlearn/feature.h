#pragma once

#include <cmath>
#include <nano/tensor.h>
#include <nano/mlearn/enums.h>

namespace nano
{
    class feature_t;
    using features_t = std::vector<feature_t>;

    ///
    /// \brief input feature (e.g. describes a column in a CSV file)
    ///     that can be either discrete/categorical or scalar/continuous
    ///     and with or without missing values.
    ///
    class feature_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        feature_t() = default;

        ///
        /// \brief constructor
        ///
        explicit feature_t(string_t name) :
            m_name(std::move(name))
        {
        }

        ///
        /// \brief set the feature as continuous.
        ///
        auto& continuous(feature_type type = feature_type::float32, tensor3d_dims_t dims = make_dims(1, 1, 1))
        {
            assert(type == feature_type::float32 || type == feature_type::float64);

            m_dims = dims;
            m_type = type;
            m_labels.clear();
            return *this;
        }

        ///
        /// \brief set the feature as discrete, by passing the labels.
        /// NB: this is useful when the labels are known before loading some dataset.
        ///
        auto& discrete(strings_t labels, feature_type type = feature_type::sclass)
        {
            assert(type == feature_type::sclass || type == feature_type::mclass);

            m_type = type;
            m_labels = std::move(labels);
            return *this;
        }

        ///
        /// \brief set the feature as discrete, but the labels are not known.
        /// NB: this is useful when the labels are discovered while loading some dataset.
        ///
        auto& discrete(size_t count, feature_type type = feature_type::sclass)
        {
            assert(type == feature_type::sclass || type == feature_type::mclass);

            m_type = type;
            m_labels = strings_t(count);
            return *this;
        }

        ///
        /// \brief set the placeholder (the feature becomes optional if the placeholder is not empty).
        ///
        auto& placeholder(string_t placeholder)
        {
            m_placeholder = std::move(placeholder);
            return *this;
        }

        ///
        /// \brief try to add the given label if possible.
        /// NB: this is useful when the labels are discovered while loading some dataset.
        ///
        size_t set_label(const string_t& label)
        {
            if (label.empty())
            {
                return string_t::npos;
            }

            const auto it = std::find(m_labels.begin(), m_labels.end(), label);
            if (it == m_labels.end())
            {
                // new label, replace the first empty label with it
                for (size_t i = 0; i < m_labels.size(); ++ i)
                {
                    if (m_labels[i].empty())
                    {
                        m_labels[i] = label;
                        return i;
                    }
                }

                // new label, but no new place for it
                return string_t::npos;
            }
            else
            {
                // known label, ignore
                return static_cast<size_t>(std::distance(m_labels.begin(), it));
            }
        }

        ///
        /// \brief returns true if the feature is discrete.
        ///
        bool discrete() const
        {
            return !m_labels.empty();
        }

        ///
        /// \brief returns true if the feature is optional.
        ///
        bool optional() const
        {
            return !m_placeholder.empty();
        }

        ///
        /// \brief returns the value to store when the feature value is missing.
        ///
        static auto placeholder_value()
        {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }

        ///
        /// \brief returns true if the given stored value indicates that the feature value is missing.
        ///
        static bool missing(const scalar_t value)
        {
            return !std::isfinite(value);
        }

        ///
        /// \brief returns the label associated to the given feature value (if possible).
        ///
        auto label(const scalar_t value) const
        {
            if (!discrete())
            {
                throw std::invalid_argument("labels are only available for discrete features");
            }
            else
            {
                return missing(value) ? string_t() : m_labels.at(static_cast<size_t>(value));
            }
        }

        ///
        /// \brief returns true if the feature is valid (aka defined).
        ///
        operator bool() const // NOLINT(hicpp-explicit-conversions)
        {
            return !m_name.empty();
        }

        ///
        /// \brief returns the associated machine learning task if this feature is the target.
        ///
        operator task_type() const
        {
            if (!static_cast<bool>(*this))
            {
                return task_type::unsupervised;
            }
            else
            {
                switch (m_type)
                {
                case feature_type::sclass:
                    return task_type::sclassification;

                case feature_type::mclass:
                    return task_type::mclassification;

                default:
                    return task_type::regression;
                }
            }
        }

        ///
        /// \brief access functions
        ///
        auto type() const { return m_type; }
        const auto& dims() const { return m_dims; }
        const auto& name() const { return m_name; }
        const auto& labels() const { return m_labels; }
        const auto& placeholder() const { return m_placeholder; }

    private:

        // attributes
        feature_type    m_type{feature_type::float32};  ///<
        tensor3d_dims_t m_dims{1, 1, 1};        ///< dimensions (if continuous)
        string_t        m_name;                 ///<
        strings_t       m_labels;               ///< possible labels (if the feature is discrete/categorical)
        string_t        m_placeholder;          ///< placeholder string used if its value is missing
    };

    ///
    /// \brief compare two features.
    ///
    inline bool operator==(const feature_t& f1, const feature_t& f2)
    {
        return  f1.type() == f2.type() &&
                f1.name() == f2.name() &&
                f1.labels() == f2.labels() &&
                f1.placeholder() == f2.placeholder();
    }

    inline bool operator!=(const feature_t& f1, const feature_t& f2)
    {
        return  f1.type() != f2.type() ||
                f1.name() != f2.name() ||
                f1.labels() != f2.labels() ||
                f1.placeholder() != f2.placeholder();
    }

    ///
    /// \brief stream the given feature.
    ///
    inline std::ostream& operator<<(std::ostream& stream, const feature_t& feature)
    {
        stream << "name=" << feature.name() << ",type=" << feature.type() << ",labels[";
        for (const auto& label : feature.labels())
        {
            stream << label;
            if (&label != &(*(feature.labels().rbegin())))
            {
                stream << ",";
            }
        }
        return stream << "],placeholder=" << feature.placeholder();
    }

    ///
    /// \brief describe a feature (e.g. as selected by a weak learner) in terms of
    ///     e.g. importance (impact on error rate).
    ///
    class feature_info_t;
    using feature_infos_t = std::vector<feature_info_t>;

    class feature_info_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        feature_info_t() = default;

        ///
        /// \brief constructor
        ///
        feature_info_t(tensor_size_t feature, tensor_size_t count, scalar_t importance) :
            m_feature(feature),
            m_count(count),
            m_importance(importance)
        {
        }

        ///
        /// \brief sort a list of (selected) features by their index.
        ///
        static void sort_by_index(feature_infos_t& features)
        {
            std::stable_sort(features.begin(), features.end(), [] (const auto& lhs, const auto& rhs)
            {
                return lhs.m_feature < rhs.m_feature;
            });
        }

        ///
        /// \brief sort a list of (selected) features descendingly by their importance.
        ///
        static void sort_by_importance(feature_infos_t& features)
        {
            std::stable_sort(features.begin(), features.end(), [] (const auto& lhs, const auto& rhs)
            {
                return lhs.m_importance > rhs.m_importance;
            });
        }

        ///
        /// \brief change the feature's importance.
        ///
        void importance(scalar_t importance)
        {
            m_importance = importance;
        }

        ///
        /// \brief access functions
        ///
        auto count() const { return m_count; }
        auto feature() const { return m_feature; }
        auto importance() const { return m_importance; }

    private:

        // attributes
        tensor_size_t   m_feature{-1};      ///< feature index
        tensor_size_t   m_count{0};         ///< how many times it was selected (e.g. folds)
        scalar_t        m_importance{0.0};  ///< feature importance (e.g. impact on performance)
    };
}
