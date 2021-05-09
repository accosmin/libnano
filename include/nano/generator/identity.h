#pragma once

#include <nano/generator.h>

namespace nano
{
    ///
    /// \brief
    ///
    class NANO_PUBLIC identity_generator_t : public generator_t
    {
    public:

        identity_generator_t(const memory_dataset_t& dataset);

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;

        void select(indices_cmap_t, tensor_size_t, sclass_map_t) const override;
        void select(indices_cmap_t, tensor_size_t, mclass_map_t) const override;
        void select(indices_cmap_t, tensor_size_t, scalar_map_t) const override;
        void select(indices_cmap_t, tensor_size_t, struct_map_t) const override;
        void flatten(indices_cmap_t, tensor2d_map_t, tensor_size_t) const override;
    };
}
