#include <nano/generator/pairwise.h>

using namespace nano;

pairwise_generator_t::pairwise_generator_t(
    const memory_dataset_t& dataset, const indices_t& samples, const feature_mapping_t& mapping) :
    generator_t(dataset, samples)
{
    const auto size = mapping.size<0>();

    m_mapping.resize(size * (size + 1) / 2, 4);
    for (tensor_size_t i = 0, k = 0; i < size; ++ i)
    {
        const auto feature1 = mapping(i * 2 + 0);
        const auto component1 = mapping(i * 2 + 1);

        for (tensor_size_t j = i; j < size; ++ j, ++ k)
        {
            const auto feature2 = mapping(j * 2 + 0);
            const auto component2 = mapping(j * 2 + 1);

            m_mapping(k, 0) = feature1;
            m_mapping(k, 1) = component1;
            m_mapping(k, 2) = feature2;
            m_mapping(k, 3) = component2;
        }
    }

    allocate(this->features());
}

tensor_size_t pairwise_generator_t::features() const
{
    return m_mapping.size<0>();
}

feature_t pairwise_generator_t::feature(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < m_mapping.size<0>());

    const auto component1 = m_mapping(ifeature, 1);
    const auto component2 = m_mapping(ifeature, 3);

    const auto& feature1 = dataset().feature(m_mapping(ifeature, 0));
    const auto& feature2 = dataset().feature(m_mapping(ifeature, 2));

    return make_feature(feature1, component1, feature2, component2);
}
