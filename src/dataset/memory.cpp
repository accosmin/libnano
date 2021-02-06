#include <nano/dataset/memory.h>

using namespace nano;

memory_dataset_t::memory_dataset_t() = default;

task_type memory_dataset_t::type() const
{
    return static_cast<task_type>(m_target.feature());
}

tensor_size_t memory_dataset_t::samples() const
{
    if (m_inputs.empty())
    {
        return 0;
    }
    else
    {
        return m_inputs.begin()->samples();
    }
}

void memory_dataset_t::resize(tensor_size_t samples, const features_t& features, size_t target)
{
    m_inputs.clear();
    m_inputs.shrink_to_fit();
    m_target = feature_storage_t{};

    for (size_t i = 0; i < features.size(); ++ i)
    {
        if (i == target)
        {
            m_target = feature_storage_t{features[i], samples};
        }
        else
        {
            m_inputs.emplace_back(features[i], samples);
        }
    }
}

rfeature_dataset_iterator_t memory_dataset_t::feature_iterator(indices_t samples) const
{
    return std::make_unique<memory_feature_dataset_iterator_t>(*this, std::move(samples));
}

rflatten_dataset_iterator_t memory_dataset_t::flatten_iterator(indices_t samples) const
{
    return std::make_unique<memory_flatten_dataset_iterator_t>(*this, std::move(samples));
}

memory_feature_dataset_iterator_t::memory_feature_dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
}

const indices_t& memory_feature_dataset_iterator_t::samples() const
{
    return m_samples;
}

memory_flatten_dataset_iterator_t::memory_flatten_dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
}

const indices_t& memory_flatten_dataset_iterator_t::samples() const
{
    return m_samples;
}
