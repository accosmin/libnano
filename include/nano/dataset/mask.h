#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
    // bitwise mask for a feature: (sample) = 1 if the feature value is available, otherwise 0
    using mask_t = tensor_mem_t<uint8_t, 1>;
    using mask_map_t = tensor_map_t<uint8_t, 1>;
    using mask_cmap_t = tensor_cmap_t<uint8_t, 1>;

    ///
    /// \brief allocate and initialize a tensor bitmask where the last dimension is the number of samples.
    ///
    template <size_t trank>
    inline auto make_mask(const tensor_dims_t<trank>& dims)
    {
        const auto samples = std::get<trank - 1>(dims);

        auto bit_dims = dims;
        bit_dims[trank - 1] = (samples + 7) / 8;

        tensor_mem_t<uint8_t, trank> mask(bit_dims);
        mask.zero();
        return mask;
    }

    ///
    /// \brief mark a feature value as set for a particular sample.
    ///
    inline void setbit(const mask_map_t& mask, tensor_size_t sample)
    {
        assert(sample >= 0 && sample < (8 * mask.size()));
        mask(sample / 8) |= static_cast<uint8_t>(0x01 << (7 - (sample % 8)));
    }

    ///
    /// \brief check if a feature value exists for a particular sample.
    ///
    inline bool getbit(const mask_cmap_t& mask, tensor_size_t sample)
    {
        assert(sample >= 0 && sample < (8 * mask.size()));
        return (mask(sample / 8) & (0x01 << (7 - (sample % 8)))) != 0x00;
    }

    ///
    /// \brief returns true if the feature is optional (aka some samples haven't been set).
    ///
    inline bool optional(const mask_cmap_t& mask, tensor_size_t samples)
    {
        const auto bytes = samples / 8;
        for (tensor_size_t byte = 0; byte < bytes; ++ byte)
        {
            if (mask(byte) != 0xFF)
            {
                return true;
            }
        }
        for (tensor_size_t sample = 8 * bytes; sample < samples; ++ sample)
        {
            if (!getbit(mask, sample))
            {
                return true;
            }
        }
        return false;
    }
}
