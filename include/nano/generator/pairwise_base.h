#pragma once

#include <nano/generator.h>

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
    class NANO_PUBLIC base_pairwise_generator_t : public generator_t
    {
    public:

        explicit base_pairwise_generator_t(const dataset_t& dataset) :
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

        tensor_size_t mapped_original1(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor_size_t mapped_original2(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 6);
        }

        tensor_size_t mapped_component1(tensor_size_t ifeature, bool clamp = true) const
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto component = m_feature_mapping(ifeature, 1);
            return clamp ? std::max(component, tensor_size_t{0}) : component;
        }

        tensor_size_t mapped_component2(tensor_size_t ifeature, bool clamp = true) const
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto component = m_feature_mapping(ifeature, 7);
            return clamp ? std::max(component, tensor_size_t{0}) : component;
        }

        tensor_size_t mapped_classes1(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 2);
        }

        tensor_size_t mapped_classes2(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 8);
        }

        tensor3d_dims_t mapped_dims1(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return make_dims(
                m_feature_mapping(ifeature, 3),
                m_feature_mapping(ifeature, 4),
                m_feature_mapping(ifeature, 5));
        }

        tensor3d_dims_t mapped_dims2(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return make_dims(
                m_feature_mapping(ifeature, 9),
                m_feature_mapping(ifeature, 10),
                m_feature_mapping(ifeature, 11));
        }

        const auto& mapping() const
        {
            return m_feature_mapping;
        }

        virtual feature_mapping_t do_fit(indices_cmap_t samples, execution ex) = 0;

    protected:

        static feature_mapping_t make_pairwise(const feature_mapping_t& mapping);
        feature_t make_scalar_feature(tensor_size_t ifeature, const char* name) const;
        feature_t make_sclass_feature(tensor_size_t ifeature, const char* name, strings_t labels) const;

    private:

        // attributes
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };
}
