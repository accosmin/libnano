#pragma once

#include <nano/generator/util.h>

namespace nano
{
    ///
    /// \brief interface for pair-wise feature generators.
    ///
    ///     new features are generated as a function of:
    ///         * original feature1,
    ///         * component index of the original feature1,
    ///         * original feature2,
    ///         * component index of the original feature2.
    ///
    class NANO_PUBLIC pairwise_generator_t : public generator_t
    {
    public:

        pairwise_generator_t(const memory_dataset_t&, const feature_mapping_t&);

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;

    protected:

        auto mapped_ifeature1(tensor_size_t ifeature) const { return m_mapping(ifeature, 0); }
        auto mapped_ifeature2(tensor_size_t ifeature) const { return m_mapping(ifeature, 2); }
        auto mapped_component1(tensor_size_t ifeature) const { return m_mapping(ifeature, 1); }
        auto mapped_component2(tensor_size_t ifeature) const { return m_mapping(ifeature, 3); }

        virtual feature_t make_feature(
            const feature_t&, tensor_size_t component1,
            const feature_t&, tensor_size_t component2) const = 0;

    private:

        // attributes
        feature_mapping_t   m_mapping;      ///< (feature1, component1, feature2, component2)
    };
}
