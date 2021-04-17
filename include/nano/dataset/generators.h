#pragma once

#include <nano/dataset/generator.h>

namespace nano
{
    // TODO: generic single and paired generator to handle the mapping and the dropping and shuffling part
    // TODO: feature-wise non-linear transformations of scalar features - sign(x)*log(1+x*x), x/sqrt(1+x*x)
    // TODO: polynomial features
    // TODO: basic image-based features: gradients, magnitude, orientation, HoG

    ///
    /// \brief
    ///
    class NANO_PUBLIC identity_generator_t : public generator_t
    {
    public:

        identity_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            mclass2binary = mclass2binary::off,
            sclass2binary = sclass2binary::off,
            struct2scalar = struct2scalar::off);

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;
        void flatten(tensor_range_t, tensor2d_map_t, tensor_size_t) const override;

        void select(tensor_size_t, tensor_range_t, sclass_map_t) const override;
        void select(tensor_size_t, tensor_range_t, mclass_map_t) const override;
        void select(tensor_size_t, tensor_range_t, scalar_map_t) const override;
        void select(tensor_size_t, tensor_range_t, struct_map_t) const override;

    private:

        void map1(tensor_size_t& index, tensor_size_t ifeature, tensor_size_t icomponent)
        {
            m_feature_mapping(index, 0) = ifeature;
            m_feature_mapping(index ++, 1) = icomponent;
        }

        void mapN(tensor_size_t& index, tensor_size_t ifeature, tensor_size_t components)
        {
            for (tensor_size_t icomponent = 0; icomponent < components; ++ icomponent)
            {
                map1(index, ifeature, icomponent);
            }
        }

        using featmapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        mclass2binary   m_mclass2binary{mclass2binary::off};    ///<
        sclass2binary   m_sclass2binary{sclass2binary::off};    ///<
        struct2scalar   m_struct2scalar{struct2scalar::off};    ///<
        featmapping_t   m_feature_mapping;                      ///<
    };
}
