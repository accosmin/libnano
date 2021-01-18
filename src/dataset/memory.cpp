#include <nano/logger.h>
#include <nano/dataset/memory.h>

using namespace nano;

feature_storage_t::feature_storage_t() = default;

feature_storage_t::feature_storage_t(feature_t feature, tensor_size_t samples) :
    m_feature(std::move(feature))
{
    const auto& name = m_feature.name();
    const auto itype = static_cast<int>(m_feature.type());

    if (m_feature.discrete())
    {
        const auto ulabels = m_feature.labels().size();
        const auto ilabels = static_cast<tensor_size_t>(ulabels);

        if (ulabels <= 0xFF)
        {
            switch (m_feature.type())
            {
            case feature_type::sclass:  m_storage = tensor_mem_t<uint8_t, 1>{samples}; break;
            case feature_type::mclass:  m_storage = tensor_mem_t<uint8_t, 2>{samples, ilabels}; break;
            default:                    critical0("feature <", name, "> has inconsistent type ", itype, "!"); break;
            }
        }
        else if (ulabels <= 0xFFFF)
        {
            switch (m_feature.type())
            {
            case feature_type::sclass:  m_storage = tensor_mem_t<uint16_t, 1>{samples}; break;
            case feature_type::mclass:  m_storage = tensor_mem_t<uint16_t, 2>{samples, ilabels}; break;
            default:                    critical0("feature <", name, "> has inconsistent type ", itype, "!"); break;
            }
        }
        else
        {
            critical0("discrete feature <", name, "> has too many labels ", ulabels, " vs. 65535!");
        }
    }
    else
    {
        const auto dims = make_dims(samples, m_feature.dims());

        switch (m_feature.type())
        {
        case feature_type::float32: m_storage = tensor_mem_t<float, 4>{dims}; break;
        case feature_type::float64: m_storage = tensor_mem_t<double, 4>{dims}; break;
        case feature_type::int8:    m_storage = tensor_mem_t<int8_t, 4>{dims}; break;
        case feature_type::int16:   m_storage = tensor_mem_t<int16_t, 4>{dims}; break;
        case feature_type::int32:   m_storage = tensor_mem_t<int32_t, 4>{dims}; break;
        case feature_type::int64:   m_storage = tensor_mem_t<int64_t, 4>{dims}; break;
        case feature_type::uint8:   m_storage = tensor_mem_t<uint8_t, 4>{dims}; break;
        case feature_type::uint16:  m_storage = tensor_mem_t<uint16_t, 4>{dims}; break;
        case feature_type::uint32:  m_storage = tensor_mem_t<uint32_t, 4>{dims}; break;
        case feature_type::uint64:  m_storage = tensor_mem_t<uint64_t, 4>{dims}; break;
        default:                    critical0("feature <", name, "> has inconsistent type ", itype, "!"); break;
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
    else if (auto pvalue = std::get_if<tensor_mem_t<uint8_t, 2>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint16_t, 1>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint16_t, 2>>(&m_storage)) { return pvalue->size<0>(); }
    else { return tensor_size_t{0}; }
}

template <typename tscalar_storage, typename tscalar_value>
static void set_scalar(
    const feature_t& feature, tensor_mem_t<tscalar_storage, 4>& tensor, tensor_size_t sample, tscalar_value value)
{
    critical(
        ::nano::size(feature.dims()) != 1 ||
        sample < 0 || sample >= tensor.size<0>(),
        "cannot set continuous feature <", feature.name(), "> of dimensions ", tensor.dims(), " from scalar!");

    tensor(sample) = static_cast<tscalar_storage>(value);
}

template <typename tscalar_storage, typename tscalar_value>
static void set_scalar(
    const feature_t& feature, tensor_mem_t<tscalar_storage, 1>& tensor, tensor_size_t sample, tscalar_value value)
{
    const auto label = static_cast<tensor_size_t>(value);
    const auto labels = static_cast<tensor_size_t>(feature.labels().size());

    critical(
        label < 0 || label >= labels ||
        sample < 0 || sample >= tensor.size<0>(),
        "cannot set single-label categorical feature <", feature.name(), "> to label ", label, "/", labels, "!");

    tensor(sample

    tensor(0) = static_cast<tscalar_storage>(value);
}

template <typename tscalar>
static void set_scalar(const string_t& name, feature_storage_t::storage_t& storage, tscalar value)
{
         if (auto pvalue = std::get_if<tensor_mem_t<float, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<double, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<int8_t, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<int16_t, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<int32_t, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<int64_t, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint8_t, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint16_t, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint32_t, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint64_t, 4>>(&m_storage)) { set_scalar(name, *pvalue, value); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint8_t, 1>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint8_t, 2>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint16_t, 1>>(&m_storage)) { return pvalue->size<0>(); }
    else if (auto pvalue = std::get_if<tensor_mem_t<uint16_t, 2>>(&m_storage)) { return pvalue->size<0>(); }
    else { return tensor_size_t{0}; }

}

void set(tensor_size_t sample, float value);
void set(tensor_size_t sample, double value);
void set(tensor_size_t sample, int8_t value);
void set(tensor_size_t sample, int16_t value);
void set(tensor_size_t sample, int32_t value);
void set(tensor_size_t sample, int64_t value);
void set(tensor_size_t sample, uint8_t value);
void set(tensor_size_t sample, uint16_t value);
void set(tensor_size_t sample, uint32_t value);
void set(tensor_size_t sample, uint64_t value);

void set(tensor_size_t sample, tensor_cmap_t<float, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<double, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<int8_t, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<int16_t, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<int32_t, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<int64_t, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<uint8_t, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<uint16_t, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<uint32_t, 3> values);
void set(tensor_size_t sample, tensor_cmap_t<uint64_t, 3> values);

void set(tensor_size_t sample, const strings_t& labels);
void set(tensor_size_t sample, const string_t& value_or_label);

tensor_cmap_t<float, 4> continuous_float() const;
tensor_cmap_t<double, 4> continuous_double() const;
tensor_cmap_t<int8_t, 4> continuous_int8() const;
tensor_cmap_t<int16_t, 4> continuous_int16() const;
tensor_cmap_t<int32_t, 4> continuous_int32() const;
tensor_cmap_t<int64_t, 4> continuous_int64() const;
tensor_cmap_t<uint8_t, 4> continuous_uint8() const;
tensor_cmap_t<uint16_t, 4> continuous_uint16() const;
tensor_cmap_t<uint32_t, 4> continuous_uint32() const;
tensor_cmap_t<uint64_t, 4> continuous_uint64() const;

tensor_cmap_t<uint8_t, 1> sclass_uint8() const;
tensor_cmap_t<uint16_t, 1> sclass_uint16() const;

tensor_cmap_t<uint8_t, 2> mclass_uint8() const;
tensor_cmap_t<uint16_t, 2> mclass_uint16() const;

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
