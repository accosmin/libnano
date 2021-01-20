#pragma once

#include <cmath>
#include <variant>
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
        feature_t();

        ///
        /// \brief constructor
        ///
        explicit feature_t(string_t name);

        ///
        /// \brief set the feature as continuous.
        ///
        feature_t& continuous(feature_type type = feature_type::float32, tensor3d_dims_t dims = make_dims(1, 1, 1));

        ///
        /// \brief set the feature as discrete, by passing the labels.
        /// NB: this is useful when the labels are known before loading some dataset.
        ///
        feature_t& discrete(strings_t labels, feature_type type = feature_type::sclass);

        ///
        /// \brief set the feature as discrete, but the labels are not known.
        /// NB: this is useful when the labels are discovered while loading some dataset.
        ///
        feature_t& discrete(size_t count, feature_type type = feature_type::sclass);

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
        /// \brief returns true if the given stored value indicates that the feature value is missing.
        ///
        static bool missing(const scalar_t value);

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
        feature_info_t();

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

    ///
    /// \brief store the feature values of a collection of samples as compact as possible.
    ///
    class NANO_PUBLIC feature_storage_t
    {
    public:

        using storage_t = std::variant
        <
            // continuous
            tensor_mem_t<float, 4>,
            tensor_mem_t<double, 4>,
            tensor_mem_t<int8_t, 4>,
            tensor_mem_t<int16_t, 4>,
            tensor_mem_t<int32_t, 4>,
            tensor_mem_t<int64_t, 4>,
            tensor_mem_t<uint8_t, 4>,
            tensor_mem_t<uint16_t, 4>,
            tensor_mem_t<uint32_t, 4>,
            tensor_mem_t<uint64_t, 4>,

            // discrete (single label)
            tensor_mem_t<uint8_t, 1>,
            tensor_mem_t<uint16_t, 1>,

            // discrete (multi label)
            tensor_mem_t<uint8_t, 2>,
            tensor_mem_t<uint16_t, 2>
        >;

        feature_storage_t();
        feature_storage_t(feature_t, tensor_size_t samples);

        tensor_size_t samples() const;
        const feature_t& feature() const { return m_feature; }
        const storage_t& storage() const { return m_storage; }

        void set(tensor_size_t sample, float value);
        void set(tensor_size_t sample, double value);
        void set(tensor_size_t sample, int8_t value);
        void set(tensor_size_t sample, int16_t value);
        void set(tensor_size_t sample, int32_t value);
        void set(tensor_size_t sample, int64_t value);
        void set(tensor_size_t sample, uint8_t value);
        void set(tensor_size_t sample, uint16_t value);
        void set(tensor_size_t sample, uint32_t value);
        void set(tensor_size_t sample, uint64_t value);
        void set(tensor_size_t sample, const string_t& value);

        void set(tensor_size_t sample, tensor_cmap_t<float, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<double, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<int8_t, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<int16_t, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<int32_t, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<int64_t, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<uint8_t, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<uint16_t, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<uint32_t, 3> values);
        void set(tensor_size_t sample, tensor_cmap_t<uint64_t, 3> values);

        // TODO: interface to set multi-label features!
        // void set(tensor_size_t sample, const strings_t& labels);

        tensor_cmap_t<float, 4> continuous_float() const;
        tensor_cmap_t<double, 4> continuous_double() const;
        tensor_cmap_t<int8_t, 4> continuous_int8() const;
        tensor_cmap_t<int16_t, 4> continuous_int16() const;
        tensor_cmap_t<int32_t, 4> continuous_int32() const;
        tensor_cmap_t<int64_t, 4> continuous_int64() const;
        tensor_cmap_t<uint8_t, 4> continuous_uint8() const;
        tensor_cmap_t<uint16_t, 4> continuous_uint16() const;
        tensor_cmap_t<uint32_t, 4> continuous_uint32() const;
        tensor_cmap_t<uint64_t, 4> continuous_uint64() const;

        tensor_cmap_t<uint8_t, 1> sclass_uint8() const;
        tensor_cmap_t<uint16_t, 1> sclass_uint16() const;

        tensor_cmap_t<uint8_t, 2> mclass_uint8() const;
        tensor_cmap_t<uint16_t, 2> mclass_uint16() const;

    private:

        // attributes
        feature_t       m_feature;  ///<
        storage_t       m_storage;  ///<
    };
}
