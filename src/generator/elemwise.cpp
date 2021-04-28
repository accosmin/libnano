#include <nano/generator/elemwise.h>

using namespace nano;

elemwise_generator_t::elemwise_generator_t(
    const memory_dataset_t& dataset, const indices_t& samples, feature_mapping_t mapping) :
    generator_t(dataset, samples),
    m_mapping(std::move(mapping))
{
    allocate(this->features());
}

tensor_size_t elemwise_generator_t::features() const
{
    return m_mapping.size<0>();
}

feature_t elemwise_generator_t::feature(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < m_mapping.size<0>());

    const auto component = m_mapping(ifeature, 1);
    const auto& feature = dataset().feature(m_mapping(ifeature, 0));

    return make_feature(feature, component);
}
