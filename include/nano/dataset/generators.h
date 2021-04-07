#pragma once

#include <nano/dataset/generator.h>

namespace nano
{
    // TODO: generic single and paired generator to handle the mapping and the dropping and shuffling part

    ///
    /// \brief
    ///
    class NANO_PUBLIC identity_generator_t : public generator_t
    {
    public:

        using generator_t::samples;

        identity_generator_t(const memory_dataset_t& dataset, const indices_t& samples);

        tensor_size_t columns() const override { return m_column_mapping.size<0>(); }
        tensor_size_t features() const override { return m_feature_mapping.size<0>(); }

        feature_t feature(tensor_size_t) const override;
        tensor_size_t column2feature(tensor_size_t column) const override;
        void flatten(tensor_range_t, tensor2d_map_t, tensor_size_t) const override;

        sclass_cmap_t select(tensor_size_t, tensor_range_t, sclass_mem_t&) const override;
        mclass_cmap_t select(tensor_size_t, tensor_range_t, mclass_mem_t&) const override;
        scalar_cmap_t select(tensor_size_t, tensor_range_t, scalar_mem_t&) const override;
        struct_cmap_t select(tensor_size_t, tensor_range_t, struct_mem_t&) const override;

        void undrop() override;
        void unshuffle() override;
        void drop(tensor_size_t feature) override;
        indices_t shuffle(tensor_size_t feature) override;

    private:

        auto should_drop(tensor_size_t feature) const { return m_feature_mapping(feature, 1) == 1; }
        auto should_shuffle(tensor_size_t feature) const { return m_feature_mapping(feature, 1) == 2; }

        auto samples(tensor_size_t feature, tensor_range_t sample_range) const
        {
            const auto& all_samples = should_shuffle(feature) ? m_shuffle_indices.find(feature)->second : samples();
            return all_samples.slice(sample_range);
        }

        // per-flatten column information:
        //  - feature index
        using column_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // per-feature information:
        //  - original feature index
        //  - flags: 0 - default, 1 - to drop, 2 - to shuffle
        using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // fixed sample indices for the features to shuffle
        using shuffle_indices_t = std::map<tensor_size_t, indices_t>;

        // attributes
        column_mapping_t    m_column_mapping;   ///<
        feature_mapping_t   m_feature_mapping;  ///<
        shuffle_indices_t   m_shuffle_indices;  ///<
    };

    // TODO: feature-wise non-linear transformations of scalar features - sign(x)*log(1+x*x), x/sqrt(1+x*x)
    // TODO: polynomial features
}
