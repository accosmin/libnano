#pragma once

#include <nano/string.h>

namespace nano
{
    ///
    /// \brief execution policy.
    ///
    enum class execution : int32_t
    {
        seq = 0,        ///< sequential: using only the thread
        par             ///< parallel: use all the available threads
    };

    ///
    /// \brief machine learning task type.
    ///
    enum class task_type : int32_t
    {
        regression = 0,     ///< regression
        sclassification,    ///< single-label classification
        mclassification,    ///< multi-label classification
        unsupervised,       ///< unsupervised
    };

    template <>
    inline enum_map_t<task_type> enum_string<task_type>()
    {
        return
        {
            { task_type::regression,        "regression" },
            { task_type::sclassification,   "s-classification" },
            { task_type::mclassification,   "m-classification" },
            { task_type::unsupervised,      "unsupervised" },
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, task_type value)
    {
        return stream << scat(value);
    }

    ///
    /// \brief input or target feature type.
    ///
    enum class feature_type : int32_t
    {
        // continuous features
        int8 = 0,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,

        // discrete features
        sclass,         ///< categorical feature (single-label - one value possible out of a fixed set)
        mclass,         ///< categorical feature (mulit-label - a subset of values possible out of a fixed set)
    };

    template <>
    inline enum_map_t<feature_type> enum_string<feature_type>()
    {
        return
        {
            { feature_type::int8,       "int8" },
            { feature_type::int16,      "int16" },
            { feature_type::int32,      "int32" },
            { feature_type::int64,      "int64" },
            { feature_type::uint8,      "uint8" },
            { feature_type::uint16,     "uint16" },
            { feature_type::uint32,     "uint32" },
            { feature_type::uint64,     "uint64" },
            { feature_type::float32,    "float32" },
            { feature_type::float64,    "float64" },
            { feature_type::sclass,     "sclass" },
            { feature_type::mclass,     "mclass" },
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, feature_type value)
    {
        return stream << scat(value);
    }

    ///
    /// \brief input feature scaling methods.
    ///
    enum class feature_scaling : int32_t
    {
        none = 0,       ///< no normalization, use the feature values as is
        mean,           ///< mean normalization: x = (x - mean(x)) / (max(x) - min(x))
        minmax,         ///< min-max normalization: x = (x - min(x)) / (max(x) - min(x))
        standard        ///< standardization with zero mean and unit variance: x = (x - min(x)) / stdev(x)
    };

    template <>
    inline enum_map_t<feature_scaling> enum_string<feature_scaling>()
    {
        return
        {
            { feature_scaling::none,    "none" },
            { feature_scaling::mean,    "mean" },
            { feature_scaling::minmax,  "minmax" },
            { feature_scaling::standard,"standard" }
        };
    }

    ///
    /// \brief method to scale weak learners.
    ///
    enum class wscale : int32_t
    {
        gboost = 0,     ///< use the same scaling factor for all samples (e.g. vanilla GradientBoosting)
        tboost,         ///< use a potentially different scaling factor for each split (e.g. see TreeBoost variation)
    };

    template <>
    inline enum_map_t<wscale> enum_string<wscale>()
    {
        return
        {
            { wscale::gboost,       "gboost" },
            { wscale::tboost,       "tboost" }
        };
    }

    ///
    /// \brief method to estimate the importance of a feature.
    ///
    enum class importance : int32_t
    {
        shuffle = 0,    ///< impact on the error rate by shuffling the feature values across samples without retraining
        dropcol,        ///< impact on the error rate by dropping the feature (aka column) and retraining without it
    };

    template <>
    inline enum_map_t<importance> enum_string<importance>()
    {
        return
        {
            { importance::shuffle,  "shuffle" },
            { importance::dropcol,  "dropcol" },
        };
    }

    ///
    /// \brief hinge type (see MARS).
    ///
    /// see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    enum class hinge : int32_t
    {
        left = 0,   ///< beta * (threshold - x(feature))+       => zero on the right, linear on the left!
        right,      ///< beta * (x(feature) - threshold)+       => zero on the left, linear on the right!
    };

    template <>
    inline enum_map_t<hinge> enum_string<hinge>()
    {
        return
        {
            { hinge::left,          "left" },
            { hinge::right,         "right" },
        };
    }

    ///
    /// \brief methods to combine the predictions of different models trained on different folds.
    ///
    /// see "Bagging Predictors", by Leo Breiman
    /// see "Stacked Regressions", by Leo Breiman
    /// see "Model search and inference by bootstrap bumping", by R. Tibshirani and K. Knight
    /// see "Combining estimates in regression and classification", by M. LeBlanc and R. Tibshirani
    ///
    enum class ensemble : int32_t
    {
        bumping = 0,///< see bumping
        stacking,   ///< see stacking
        bagging,    ///< see bagging
        median,     ///< see bagging, but output the median per sample of the models' predictions
    };

    template <>
    inline enum_map_t<ensemble> enum_string<ensemble>()
    {
        return
        {
            { ensemble::bumping,    "bumping" },
            { ensemble::stacking,   "stacking" },
            { ensemble::bagging,    "average" },
            { ensemble::median,     "median" },
        };
    }
}
