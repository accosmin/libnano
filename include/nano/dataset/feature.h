#pragma once

#include <cmath>
#include <nano/arch.h>
#include <nano/tensor.h>
#include <nano/mlearn/enums.h>

namespace nano
{
    class feature_t;
    using features_t = std::vector<feature_t>;

    ///
    /// \brief input feature (e.g. describes a column in a csv file)
    ///     that can be either discrete/categorical or scalar/continuous
    ///     and with or without missing values.
    ///
    class NANO_PUBLIC feature_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        feature_t() = default;

        ///
        /// \brief constructor
        ///
        explicit feature_t(string_t name);

        ///
        /// \brief set the feature as continuous.
        ///
        feature_t& scalar(feature_type type = feature_type::float32, tensor3d_dims_t dims = make_dims(1, 1, 1));

        ///
        /// \brief set the feature as discrete, by passing the labels.
        /// NB: this is useful when the labels are known before loading some dataset.
        ///
        feature_t& sclass(strings_t labels);
        feature_t& mclass(strings_t labels);

        ///
        /// \brief set the feature as discrete, but the labels are not known.
        /// NB: this is useful when the labels are discovered while loading some dataset.
        ///
        feature_t& sclass(size_t count);
        feature_t& mclass(size_t count);

        ///
        /// \brief set the feature optional.
        ///
        feature_t& optional(bool optional);

        ///
        /// \brief try to add the given label if possible.
        /// NB: this is useful when the labels are discovered while loading some dataset.
        ///
        size_t set_label(const string_t& label);

        ///
        /// \brief returns true if the feature is discrete.
        ///
        bool discrete() const;

        ///
        /// \brief returns the value to store when the feature value is missing.
        ///
        static scalar_t placeholder_value();

        ///
        /// \brief returns true if the given scalar value is valid,
        ///     otherwise it indicates a missing feature value for some sample.
        ///
        static bool missing(scalar_t value)
        {
            return !std::isfinite(value);
        }

        ///
        /// \brief returns true if the given label index  is valid,
        ///     otherwise it indicates a missing feature value for some sample.
        ///
        static bool missing(tensor_size_t label)
        {
            return label < 0;
        }

        ///
        /// \brief returns the label associated to the given feature value (if possible).
        ///
        string_t label(const scalar_t value) const;

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
        auto optional() const { return m_optional; }
        const auto& dims() const { return m_dims; }
        const auto& name() const { return m_name; }
        const auto& labels() const { return m_labels; }
        auto classes() const { return static_cast<tensor_size_t>(m_labels.size()); }

    private:

        // attributes
        bool            m_optional{false};      ///<
        feature_type    m_type{feature_type::float32};  ///<
        tensor3d_dims_t m_dims{1, 1, 1};        ///< dimensions (if continuous)
        string_t        m_name;                 ///<
        strings_t       m_labels;               ///< possible labels (if the feature is discrete/categorical)
    };

    ///
    /// \brief compare two features.
    ///
    NANO_PUBLIC bool operator==(const feature_t& f1, const feature_t& f2);
    NANO_PUBLIC bool operator!=(const feature_t& f1, const feature_t& f2);

    ///
    /// \brief stream the given feature.
    ///
    NANO_PUBLIC std::ostream& operator<<(std::ostream& stream, const feature_t& feature);

    ///
    /// \brief describe a feature (e.g. as selected by a weak learner) in terms of
    ///     e.g. importance (impact on error rate).
    ///
    class feature_info_t;
    using feature_infos_t = std::vector<feature_info_t>;

    class NANO_PUBLIC feature_info_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        feature_info_t() = default;

        ///
        /// \brief constructor
        ///
        feature_info_t(tensor_size_t feature, tensor_size_t count, scalar_t importance);

        ///
        /// \brief sort a list of (selected) features by their index.
        ///
        static void sort_by_index(feature_infos_t& features);

        ///
        /// \brief sort a list of (selected) features descendingly by their importance.
        ///
        static void sort_by_importance(feature_infos_t& features);

        ///
        /// \brief change the feature's importance.
        ///
        void importance(scalar_t importance);

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
