#pragma once

#include <nano/generator.h>

namespace nano
{
    ///
    /// \brief interface for element-wise feature generators.
    ///
    ///     new features are generated as a function of:
    ///         * the original feature and
    ///         * the component index of the original feature.
    ///
    class NANO_PUBLIC base_elemwise_generator_t : public generator_t
    {
    public:

        base_elemwise_generator_t(const memory_dataset_t& dataset) :
            generator_t(dataset)
        {
        }

        void fit(indices_cmap_t samples, execution ex) override
        {
            m_feature_mapping = do_fit(samples, ex);
            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor_size_t mapped_component(tensor_size_t ifeature, bool clamp = true) const
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto component = m_feature_mapping(ifeature, 1);
            return clamp ? std::max(component, tensor_size_t{0}) : component;
        }

        tensor_size_t mapped_classes(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 2);
        }

        tensor3d_dims_t mapped_dims(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return make_dims(
                m_feature_mapping(ifeature, 3),
                m_feature_mapping(ifeature, 4),
                m_feature_mapping(ifeature, 5));
        }

        const auto& mapping() const
        {
            return m_feature_mapping;
        }

        virtual feature_mapping_t do_fit(indices_cmap_t samples, execution ex) = 0;

    protected:

        feature_t make_scalar_feature(tensor_size_t ifeature, const char* name) const;
        feature_t make_sclass_feature(tensor_size_t ifeature, const char* name, strings_t labels) const;

    private:

        // attributes
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };
}
