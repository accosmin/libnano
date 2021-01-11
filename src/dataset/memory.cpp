#include <nano/logger.h>
#include <nano/dataset/memory.h>

using namespace nano;

feature_storage_t::feature_storage() = default;

feature_storage_t::feature_storage_t(feature_t feature, tensor_size_t samples) :
    m_feature(std::move(feature))
{
    if (m_feature.discrete())
    {
        const auto labels = m_feature.labels().size();
        if (labels < 0xFF)
        {
            m_storage = tensor_mem_t<uint8_t, 1>{samples};
        }
        else if (labels < 0xFFFF)
        {
            m_storage = tensor_mem_t<uint16_t, 1>{samples};
        }
        else
        {
            const auto& name = m_feature.name();
            critical(scat("discrete feature <", name, "> has too many labels ", labels, " vs. 65535!"));
        }
    }
    else
    {
        const auto& name = m_feature.name();
        const auto dims = make_dims(samples, m_feature.dims());

        switch (m_feature.type())
        {
        feature_type::float32:  m_storage = tensor_mem_t<float, 4>{dims}; break;
        feature_type::float64:  m_storage = tensor_mem_t<double, 4>{dims}; break;
        feature_type::int8:     m_storage = tensor_mem_t<int8_t, 4>{dims}; break;
        feature_type::int16:    m_storage = tensor_mem_t<int16_t, 4>{dims}; break;
        feature_type::int32:    m_storage = tensor_mem_t<int32_t, 4>{dims}; break;
        feature_type::int64:    m_storage = tensor_mem_t<int64_t, 4>{dims}; break;
        feature_type::uint8:    m_storage = tensor_mem_t<uint8_t, 4>{dims}; break;
        feature_type::uint16:   m_storage = tensor_mem_t<uint16_t, 4>{dims}; break;
        feature_type::uint32:   m_storage = tensor_mem_t<uint32_t, 4>{dims}; break;
        feature_type::uint64:   m_storage = tensor_mem_t<uint64_t, 4>{dims}; break;

        default:
            critical(scat("continuous feature <", name, "> has invalid type ", static_cast<int>(m_feature.type()), "!"));
            break;
        }
    }
}

tensor_size_t feature_storage_t::samples() const
{
         if (auto pvalue = std::get_if<tensor_mem_t<float, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<double, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<int8_t, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<int16_t, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<int32_t, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<int64_t, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint8_t, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint16_t, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint32_t, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint64_t, 4>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint8_t, 1>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint16_t, 1>>(&m_storage)) { return pvalue->size<0>(); }
    else { return tensor_size_t{0}; }
}

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

    for (size_t i = 0; i < features.size(); ++ i)
    {
        if (i == target)
        {
            critical(
                features[i].optional(),
                scat("optional target features are not supported!"));

            m_target = feature_storage_t{features[i], samples};
        }
        else
        {
            m_inputs.emplace_back(features[i], samples);
        }
    }
}
