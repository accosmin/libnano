#pragma once

#include <variant>
#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief
    ///
    class feature_storage_t
    {
    public:

        using storage_t = std::variant
        <
            // continuous
            tensor_mem_t<float, 4>,
            tensor_mem_t<double, 4>,
            tensor_mem_t<uint8_t, 4>,
            tensor_mem_t<uint16_t, 4>,
            tensor_mem_t<uint32_t, 4>,
            tensor_mem_t<uint64_t, 4>,
            tensor_mem_t<int8_t, 4>,
            tensor_mem_t<int16_t, 4>,
            tensor_mem_t<int32_t, 4>,
            tensor_mem_t<int64_t, 4>,

            // discrete (single label)
            tensor_mem_t<uint8_t, 1>,
            tensor_mem_t<uint16_t, 1>,

            // discrete (multi label)
            tensor_mem_t<uint8_t, 2>,
            tensor_mem_t<uint16_t, 2>,
        >;

        feature_storage_t();
        feature_storage_t(feature_t, tensor_size_t samples);

        tensor_size_t samples() const;
        const auto& feature() const { return m_feature; }
        const auto& storage() const { return m_storage; }

        tensor_cmap_t<float, 4> as_float() const;
        tensor_cmap_t<float, 4> as_double() const;

    private:

        // attributes
        feature_t       m_feature;  ///<
        storage_t       m_storage;  ///<
    };

    ///
    /// \brief in-memory tabular datasets mixes features of different types:
    ///     - discrete features stored using either 8-bit or 16-bit integers
    ///     - continuous features stored as single or double precision floating points
    ///
    /// NB: the input features can be optional.
    /// NB: the discrete input features can be both single-label or multi-label.
    /// NB: the continuous input features cannot be multi-dimensional (::dims() == (1, 1, 1)).
    ///
    class memory_dataset_t : public dataset_t
    {
    public:

        memory_dataset_t();

        feature_t target() const override;
        tensor_size_t samples() const override;
        tensor3d_dims_t tdims() const override;

        auto& istorage() { return m_inputs; }
        auto& tstorage() { return m_target; }

        const auto& istorage() const { return m_inputs; }
        const auto& tstorage() const { return m_target; }

        void resize(tensor_size_t samples, const features_t& features, size_t target);

        template <typename tscalar>
        void write(tensor_size_t sample, tensor_size_t feature, tscalar value);

        template <typename tscalar>
        void write(tensor_size_t sample, tensor_size_t feature, tensor_cmap_t<tscalar, 3> value);

        template <typename tscalar>
        void write(tensor_size_t sample, tensor_size_t feature, tensor_size_t);

        bool missing(tensor_size_t sample, tensor_size_t feature) const
        {
            return (m_missing(sample, feature / 8) & (0x01 << (7 - feature % 8))) != 0x00;
        }

    private:

        using target_t = feature_storage_t;
        using inputs_t = std::vector<feature_storage_t>;
        using missing_t = tensor_mem_t<uint8_t, 2>;

        // attributes
        target_t        m_target;   ///<
        inputs_t        m_inputs;   ///<
        missing_t       m_missing;  ///< bitwise indication of missing features (samples, inputs/8)
    };
}
