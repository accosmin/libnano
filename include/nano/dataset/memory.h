#pragma once

#include <nano/dataset.h>

namespace nano
{
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
