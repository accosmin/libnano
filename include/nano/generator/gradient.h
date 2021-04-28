#pragma once

#include <nano/generator.h>

namespace nano
{
    // TODO: generic single and paired generator to handle the mapping and the dropping and shuffling part
    // TODO: feature-wise non-linear transformations of scalar features - sign(x)*log(1+x*x), x/sqrt(1+x*x)
    // TODO: polynomial features
    // TODO: basic image-based features: gradients, magnitude, orientation, HoG
    // TODO: histogram-based scalar features - assign scalar value into its percentile range index
    // TODO: sign -> transform scalar value to its sign class or sign scalar value
    // TODO: clamp_perc -> clamp scalar value outside a given percentile range
    // TODO: clamp -> clamp scalar value to given range

    ///
    /// \brief
    ///
    class NANO_PUBLIC gradient_generator_t : public generator_t
    {
    public:

        gradient_generator_t(const memory_dataset_t& dataset, const indices_t& samples);

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;

        void select(tensor_size_t, tensor_range_t, struct_map_t) const override;
        void flatten(tensor_range_t, tensor2d_map_t, tensor_size_t) const override;

    private:

        // attributes
    };
}
