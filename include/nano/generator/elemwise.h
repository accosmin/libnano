#pragma once

#include <nano/generator/util.h>

namespace nano
{
    ///
    /// \brief interface for element-wise feature generators.
    ///
    ///     new features are generated as a function of:
    ///         * original feature,
    ///         * component index of the original feature.
    ///
    class NANO_PUBLIC elemwise_generator_t : public generator_t
    {
    public:

        elemwise_generator_t(const memory_dataset_t&, const indices_t& samples, feature_mapping_t);

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;

    protected:

        auto mapped_ifeature(tensor_size_t ifeature) const { return m_mapping(ifeature, 0); }
        auto mapped_component(tensor_size_t ifeature) const { return m_mapping(ifeature, 1); }

        virtual feature_t make_feature(
            const feature_t&, tensor_size_t component) const = 0;

    private:

        // attributes
        feature_mapping_t   m_mapping;      ///< (feature, component)
    };
}
