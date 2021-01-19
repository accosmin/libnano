#include <nano/logger.h>
#include <nano/dataset/memory.h>

using namespace nano;

memory_dataset_t::memory_dataset_t() = default;

feature_t memory_dataset_t::target() const
{
    return m_target.feature();
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

tensor3d_dims_t memory_dataset_t::tdims() const
{
    const auto& feature = m_target.feature();

    if (!static_cast<bool>(feature))
    {
        return make_dims(0, 0, 0);
    }
    else if (feature.discrete())
    {
        return make_dims(static_cast<tensor_size_t>(feature.labels().size()), 1, 1);
    }
    else
    {
        return feature.dims();
    }
}

void memory_dataset_t::resize(tensor_size_t samples, const features_t& features, size_t target)
{
    m_inputs.clear();
    m_inputs.shrink_to_fit();
    m_target = feature_storage_t{};

    m_missing.resize(samples, static_cast<tensor_size_t>(features.size()));
    m_missing.constant(0xFF);

    for (size_t i = 0; i < features.size(); ++ i)
    {
        if (i == target)
        {
            critical(
                features[i].optional(),
                "optional target features are not supported!");

            m_target = feature_storage_t{features[i], samples};
        }
        else
        {
            m_inputs.emplace_back(features[i], samples);
        }
    }
}
