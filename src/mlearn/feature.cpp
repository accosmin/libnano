#include <nano/logger.h>
#include <nano/mlearn/feature.h>

using namespace nano;

feature_t::feature_t() = default;

feature_t::feature_t(string_t name) :
    m_name(std::move(name))
{
}

feature_t& feature_t::continuous(feature_type type, tensor3d_dims_t dims)
{
    assert(type == feature_type::float32 || type == feature_type::float64);

    m_dims = dims;
    m_type = type;
    m_labels.clear();
    return *this;
}

feature_t& feature_t::discrete(strings_t labels, feature_type type)
{
    assert(type == feature_type::sclass || type == feature_type::mclass);

    m_type = type;
    m_labels = std::move(labels);
    return *this;
}

feature_t& feature_t::discrete(size_t count, feature_type type)
{
    assert(type == feature_type::sclass || type == feature_type::mclass);

    m_type = type;
    m_labels = strings_t(count);
    return *this;
}

feature_t& feature_t::optional(bool optional)
{
    m_optional = optional;
    return *this;
}

size_t feature_t::set_label(const string_t& label)
{
    if (label.empty())
    {
        return string_t::npos;
    }

    const auto it = std::find(m_labels.begin(), m_labels.end(), label);
    if (it == m_labels.end())
    {
        // new label, replace the first empty label with it
        for (size_t i = 0; i < m_labels.size(); ++ i)
        {
            if (m_labels[i].empty())
            {
                m_labels[i] = label;
                return i;
            }
        }

        // new label, but no new place for it
        return string_t::npos;
    }
    else
    {
        // known label, ignore
        return static_cast<size_t>(std::distance(m_labels.begin(), it));
    }
}

bool feature_t::discrete() const
{
    return !m_labels.empty();
}

scalar_t feature_t::placeholder_value()
{
    return std::numeric_limits<scalar_t>::quiet_NaN();
}

bool feature_t::missing(scalar_t value)
{
    return !std::isfinite(value);
}

string_t feature_t::label(scalar_t value) const
{
    if (!discrete())
    {
        throw std::invalid_argument("labels are only available for discrete features");
    }
    else
    {
        return missing(value) ? string_t() : m_labels.at(static_cast<size_t>(value));
    }
}

bool ::nano::operator==(const feature_t& f1, const feature_t& f2)
{
    return  f1.type() == f2.type() &&
            f1.name() == f2.name() &&
            f1.labels() == f2.labels() &&
            f1.optional() == f2.optional();
}

bool ::nano::operator!=(const feature_t& f1, const feature_t& f2)
{
    return  f1.type() != f2.type() ||
            f1.name() != f2.name() ||
            f1.labels() != f2.labels() ||
            f1.optional() != f2.optional();
}

std::ostream& ::nano::operator<<(std::ostream& stream, const feature_t& feature)
{
    stream << "name=" << feature.name() << ",type=" << feature.type() << ",labels[";
    for (const auto& label : feature.labels())
    {
        stream << label;
        if (&label != &(*(feature.labels().rbegin())))
        {
            stream << ",";
        }
    }
    return stream << "]," << (feature.optional() ? "optional" : "mandatory");
}

feature_info_t::feature_info_t() = default;

feature_info_t::feature_info_t(tensor_size_t feature, tensor_size_t count, scalar_t importance) :
    m_feature(feature),
    m_count(count),
    m_importance(importance)
{
}

void feature_info_t::sort_by_index(feature_infos_t& features)
{
    std::stable_sort(features.begin(), features.end(), [] (const auto& lhs, const auto& rhs)
    {
        return lhs.m_feature < rhs.m_feature;
    });
}

void feature_info_t::sort_by_importance(feature_infos_t& features)
{
    std::stable_sort(features.begin(), features.end(), [] (const auto& lhs, const auto& rhs)
    {
        return lhs.m_importance > rhs.m_importance;
    });
}

void feature_info_t::importance(scalar_t importance)
{
    m_importance = importance;
}

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
        const auto dims = cat_dims(samples, m_feature.dims());

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
         if (auto tensor = std::get_if<tensor_mem_t<float, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<double, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<int8_t, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<int16_t, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<int32_t, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<int64_t, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint8_t, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint16_t, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint32_t, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint64_t, 4>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint8_t, 1>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint8_t, 2>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint16_t, 1>>(&m_storage)) { return tensor->size<0>(); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint16_t, 2>>(&m_storage)) { return tensor->size<0>(); }
    else { return tensor_size_t{0}; }
}

template
<
    typename tscalar,
    typename tvalue,
    typename = typename std::enable_if<std::is_arithmetic<tvalue>::value>::type
>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 1>& tensor, tensor_size_t sample, tvalue value)
{
    const auto label = static_cast<tensor_size_t>(value);
    const auto labels = static_cast<tensor_size_t>(feature.labels().size());

    critical(
        label < 0 || label >= labels ||
        sample < 0 || sample >= tensor.size(),
        "cannot set single-label discrete feature <", feature.name(), "> to label ", label, "/", labels, "!");

    tensor(sample) = static_cast<tscalar>(value);
}

template
<
    typename tscalar,
    typename tvalue,
    typename = typename std::enable_if<std::is_arithmetic<tvalue>::value>::type
>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 2>&, tensor_size_t, tvalue)
{
    critical0("cannot set multi-label discrete feature <", feature.name(), ">!");
}

template
<
    typename tscalar,
    typename tvalue,
    typename = typename std::enable_if<std::is_arithmetic<tvalue>::value>::type
>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 4>& tensor, tensor_size_t sample, tvalue value)
{
    critical(
        ::nano::size(feature.dims()) != 1 ||
        sample < 0 || sample >= tensor.size(),
        "cannot set continuous feature <", feature.name(), "> of dimensions ", tensor.dims(), " from scalar!");

    tensor(sample) = static_cast<tscalar>(value);
}

template
<
    typename tscalar,
    typename tvalue,
    typename = typename std::enable_if<std::is_arithmetic<tvalue>::value>::type
>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 1>&, tensor_size_t, tensor_cmap_t<tvalue, 3>)
{
    critical0("cannot set single-label discrete feature <", feature.name(), "> from tensor!");
}

template
<
    typename tscalar,
    typename tvalue,
    typename = typename std::enable_if<std::is_arithmetic<tvalue>::value>::type
>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 2>&, tensor_size_t, tensor_cmap_t<tvalue, 3>)
{
    critical0("cannot set single-label discrete feature <", feature.name(), "> from tensor!");
}

template
<
    typename tscalar,
    typename tvalue,
    typename = typename std::enable_if<std::is_arithmetic<tvalue>::value>::type
>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 4>& tensor, tensor_size_t sample, tensor_cmap_t<tvalue, 3> values)
{
    critical(
        feature.dims() != values.dims() ||
        sample < 0 || sample >= tensor.size(),
        "cannot set continuous feature <", feature.name(), "> from tensor of dimensions ", values.dims(), "/", tensor.dims(), "!");

    tensor.vector(sample) = values.vector().template cast<tscalar>();
}

template <typename tscalar>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 1>& tensor, tensor_size_t sample, const string_t& value)
{
    set(feature, tensor, sample, ::nano::from_string<tensor_size_t>(value));
}

template <typename tscalar>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 2>& tensor, tensor_size_t sample, const string_t& value)
{
    set(feature, tensor, sample, ::nano::from_string<tensor_size_t>(value));
}

template <typename tscalar>
static void set(const feature_t& feature, tensor_mem_t<tscalar, 4>& tensor, tensor_size_t sample, const string_t& value)
{
    set(feature, tensor, sample, ::nano::from_string<tscalar>(value));
}

template <typename tvalue>
static void set(
    const feature_t& feature, feature_storage_t::storage_t& storage, tensor_size_t sample, const tvalue& value)
{
         if (auto tensor = std::get_if<tensor_mem_t<float, 4>>(&storage))    { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<double, 4>>(&storage))   { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<int8_t, 4>>(&storage))   { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<int16_t, 4>>(&storage))  { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<int32_t, 4>>(&storage))  { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<int64_t, 4>>(&storage))  { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint8_t, 4>>(&storage))  { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint16_t, 4>>(&storage)) { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint32_t, 4>>(&storage)) { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint64_t, 4>>(&storage)) { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint8_t, 1>>(&storage))  { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint8_t, 2>>(&storage))  { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint16_t, 1>>(&storage)) { ::set(feature, *tensor, sample, value); }
    else if (auto tensor = std::get_if<tensor_mem_t<uint16_t, 2>>(&storage)) { ::set(feature, *tensor, sample, value); }
    else
    {
        critical0("cannot set unitialized feature <", feature.name(), ">!");
    }
}

void feature_storage_t::set(tensor_size_t sample, float value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, double value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, int8_t value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, int16_t value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, int32_t value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, int64_t value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, uint8_t value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, uint16_t value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, uint32_t value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, uint64_t value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, const string_t& value)
{
    ::set(m_feature, m_storage, sample, value);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<float, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<double, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<int8_t, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<int16_t, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<int32_t, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<int64_t, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<uint8_t, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<uint16_t, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<uint32_t, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}

void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<uint64_t, 3> values)
{
    ::set(m_feature, m_storage, sample, values);
}
