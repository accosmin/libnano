#pragma once

#include <nano/logger.h>
#include <nano/dataset/mask.h>
#include <nano/dataset/feature.h>

namespace nano
{
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
