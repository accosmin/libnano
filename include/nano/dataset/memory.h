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
            tensor_mem_t<uint16_t, 2>,
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

        void set(tensor_size_t sample, const strings_t& labels);
        void set(tensor_size_t sample, const string_t& value_or_label);

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

        template <typename tvalue>
        void set(tensor_size_t sample, tensor_size_t feature, const tvalue& value);

        template <typename tvalue>
        void set(tensor_size_t sample, const tvalue& value);

        bool missing(tensor_size_t sample, tensor_size_t feature) const
        {
            const auto mask = 0x01 << (7 - (feature % 8));
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
