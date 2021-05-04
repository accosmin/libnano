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

        identity_generator_t(const memory_dataset_t& dataset, const indices_t& samples);

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;

        void select(tensor_size_t, tensor_range_t, sclass_map_t) const override;
        void select(tensor_size_t, tensor_range_t, mclass_map_t) const override;
        void select(tensor_size_t, tensor_range_t, scalar_map_t) const override;
        void select(tensor_size_t, tensor_range_t, struct_map_t) const override;
        void flatten(tensor_range_t, tensor2d_map_t, tensor_size_t) const override;
    };
}
