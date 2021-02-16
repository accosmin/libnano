#include <nano/logger.h>
#include <nano/mlearn/feature.h>

using namespace nano;

feature_t::feature_t(string_t name) :
    m_name(std::move(name))
{
}

feature_t& feature_t::scalar(feature_type type, tensor3d_dims_t dims)
{
    assert(
        type != feature_type::sclass &&
        type != feature_type::mclass);

    m_dims = dims;
    m_type = type;
    m_labels.clear();
    return *this;
}

feature_t& feature_t::sclass(strings_t labels)
{
    m_type = feature_type::sclass;
    m_labels = std::move(labels);
    return *this;
}

feature_t& feature_t::mclass(strings_t labels)
{
    m_type = feature_type::mclass;
    m_labels = std::move(labels);
    return *this;
}

feature_t& feature_t::sclass(size_t count)
{
    m_type = feature_type::sclass;
    m_labels = strings_t(count);
    return *this;
}

feature_t& feature_t::mclass(size_t count)
{
    m_type = feature_type::mclass;
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
            case feature_type::mclass:  m_storage = tensor_mem_t<uint8_t, 2>{samples, ilabels}; break;
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

    m_mask.resize((samples + 7) / 8);
    m_mask.zero();
}

namespace {

template <typename tscalar, typename tvalue>
void set(const feature_t& feature, tensor_mem_t<tscalar, 1>& tensor, tensor_size_t sample, const tvalue& value)
{
    const auto samples = tensor.template size<0>();

    critical(
        sample < 0 || sample >= samples,
        "cannot set single-label feature <", feature.name(),
        ">: invalid sample ", sample, " not in [0, ", samples, ")!");

    if constexpr (std::is_arithmetic<tvalue>::value)
    {
        const auto label = static_cast<tensor_size_t>(value);
        const auto labels = static_cast<tensor_size_t>(feature.labels().size());

        critical(
            label < 0 || label >= labels,
            "cannot set single-label feature <", feature.name(),
            ">: invalid label ", label, " not in [0, ", labels, ")!");

        tensor(sample) = static_cast<tscalar>(value);
    }

    else
    {
        critical0("cannot set single-label feature <", feature.name(), ">!");
    }
}

template <typename tscalar, typename tvalue>
void set(const feature_t& feature, tensor_mem_t<tscalar, 2>& tensor, tensor_size_t sample, const tvalue& value)
{
    const auto samples = tensor.template size<0>();

    critical(
        sample < 0 || sample >= samples,
        "cannot set multi-label feature <", feature.name(),
        ">: invalid sample ", sample, " not in [0, ", samples, ")!");

    if constexpr (::nano::is_tensor<tvalue>::value)
    {
        if constexpr (tvalue::rank() == 1)
        {
            const auto labels = static_cast<tensor_size_t>(feature.labels().size());

            critical(
                value.size() != labels,
                "cannot set multi-label feature <", feature.name(),
                ">: invalid number of labels ", value.size(), " vs. ", labels, "!");

            tensor.vector(sample) = value.vector().template cast<tscalar>();
        }
        else
        {
            critical0("cannot set multi-label feature <", feature.name(), ">!");
        }
    }
    else
    {
        critical0("cannot set multi-label feature <", feature.name(), ">!");
    }
}

template <typename tscalar, typename tvalue>
void set(const feature_t& feature, tensor_mem_t<tscalar, 4>& tensor, tensor_size_t sample, const tvalue& value)
{
    const auto samples = tensor.template size<0>();

    critical(
        sample < 0 || sample >= samples,
        "cannot set scalar feature <", feature.name(),
        ">: invalid sample ", sample, " not in [0, ", samples, ")!");

    if constexpr (std::is_arithmetic<tvalue>::value)
    {
        critical(
            ::nano::size(feature.dims()) != 1,
            "cannot set scalar feature <", feature.name(),
            ">: invalid tensor dimensions ", feature.dims(), "!");

        tensor(sample) = static_cast<tscalar>(value);
    }
    else if constexpr (tvalue::rank() == 1)
    {
        critical(
            ::nano::size(feature.dims()) != value.size(),
            "cannot set scalar feature <", feature.name(),
            ">: invalid tensor dimensions ", feature.dims(), " vs. ", value.dims(), "!");

        tensor.vector(sample) = value.vector().template cast<tscalar>();
    }
    else if constexpr (tvalue::rank() == 3)
    {
        critical(
            feature.dims() != value.dims(),
            "cannot set scalar feature <", feature.name(),
            ">: invalid tensor dimensions ", feature.dims(), " vs. ", value.dims(), "!");

        tensor.vector(sample) = value.vector().template cast<tscalar>();
    }
    else
    {
        critical0("cannot set scalar feature <", feature.name(), ">!");
    }
}

template <typename tscalar>
void set(const feature_t& feature, tensor_mem_t<tscalar, 1>& tensor, tensor_size_t sample, const string_t& value)
{
    tensor_size_t label;
    try
    {
        label = ::nano::from_string<tensor_size_t>(value);
    }
    catch (std::exception& e)
    {
        critical0("cannot set single-label feature <", feature.name(), ">: caught exception <", e.what(), ">!");
    }
    set(feature, tensor, sample, label);
}

template <typename tscalar>
void set(const feature_t& feature, tensor_mem_t<tscalar, 2>& tensor, tensor_size_t sample, const string_t& value)
{
    tensor_size_t scalar;
    try
    {
        scalar = ::nano::from_string<tensor_size_t>(value);
    }
    catch (std::exception& e)
    {
        critical0("cannot set multi-label feature <", feature.name(), ">: caught exception <", e.what(), ">!");
    }
    set(feature, tensor, sample, scalar);
}

template <typename tscalar>
void set(const feature_t& feature, tensor_mem_t<tscalar, 4>& tensor, tensor_size_t sample, const string_t& value)
{
    tscalar scalar;
    try
    {
        scalar = ::nano::from_string<tscalar>(value);
    }
    catch (std::exception& e)
    {
        critical0("cannot set scalar feature <", feature.name(), ">: caught exception <", e.what(), ">!");
    }
    set(feature, tensor, sample, scalar);
}

}

#define FEATURE_STORAGE_SET_SCALAR(SCALAR) \
void feature_storage_t::set(tensor_size_t sample, SCALAR value) \
{ \
    visit([&] (auto& tensor) { ::set(m_feature, tensor, sample, value); }); \
    ::nano::setbit(m_mask, sample); \
} \
\
void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<SCALAR, 1> values) \
{ \
    visit([&] (auto& tensor) { ::set(m_feature, tensor, sample, values); }); \
    ::nano::setbit(m_mask, sample); \
} \
\
void feature_storage_t::set(tensor_size_t sample, tensor_cmap_t<SCALAR, 3> values) \
{ \
    visit([&] (auto& tensor) { ::set(m_feature, tensor, sample, values); }); \
    ::nano::setbit(m_mask, sample); \
}

FEATURE_STORAGE_SET_SCALAR(float)
FEATURE_STORAGE_SET_SCALAR(double)
FEATURE_STORAGE_SET_SCALAR(int8_t)
FEATURE_STORAGE_SET_SCALAR(int16_t)
FEATURE_STORAGE_SET_SCALAR(int32_t)
FEATURE_STORAGE_SET_SCALAR(int64_t)
FEATURE_STORAGE_SET_SCALAR(uint8_t)
FEATURE_STORAGE_SET_SCALAR(uint16_t)
FEATURE_STORAGE_SET_SCALAR(uint32_t)
FEATURE_STORAGE_SET_SCALAR(uint64_t)

#undef FEATURE_STORAGE_SET_SCALAR

void feature_storage_t::set(tensor_size_t sample, const string_t& value)
{
    visit([&] (auto& tensor) { ::set(m_feature, tensor, sample, value); });
    ::nano::setbit(m_mask, sample);
}

namespace
{

template <typename tscalar, typename tvalue, size_t trank>
void get(
    const feature_t& feature,
    const scalar_storage_t<tscalar>& tensor,
    const feature_mask_t& mask,
    const indices_cmap_t& samples,
    tensor_mem_t<tvalue, trank>& values)
{
    if constexpr (trank == 4)
    {
        values.resize(cat_dims(samples.size(), feature.dims()));
        values.constant(std::numeric_limits<scalar_t>::quiet_NaN());

        for (tensor_size_t i = 0; i < samples.size(); ++ i)
        {
            if (::nano::getbit(mask, samples(i)))
            {
                values.vector(i) = tensor.vector(samples(i)).template cast<scalar_t>();
            }
        }
    }
    else
    {
        critical0("cannot access scalar feature <", feature.name(), ">!");
    }
}

template <typename tscalar, typename tvalue, size_t trank>
void get(
    const feature_t& feature,
    const sclass_storage_t<tscalar>& tensor,
    const feature_mask_t& mask,
    const indices_cmap_t& samples,
    tensor_mem_t<tvalue, trank>& values)
{
    if constexpr (trank == 1)
    {
        values.resize(samples.size());
        values.constant(-1);

        for (tensor_size_t i = 0; i < samples.size(); ++ i)
        {
            if (::nano::getbit(mask, samples(i)))
            {
                values(i) = static_cast<tensor_size_t>(tensor(samples(i)));
            }
        }
    }
    else
    {
        critical0("cannot access single-label feature <", feature.name(), ">!");
    }
}

template <typename tscalar, typename tvalue, size_t trank>
void get(
    const feature_t& feature,
    const mclass_storage_t<tscalar>& tensor,
    const feature_mask_t& mask,
    const indices_cmap_t& samples,
    tensor_mem_t<tvalue, trank>& values)
{
    if constexpr (trank == 2)
    {
        const auto labels = static_cast<tensor_size_t>(feature.labels().size());

        values.resize(make_dims(samples.size(), labels));
        values.constant(-1);

        for (tensor_size_t i = 0; i < samples.size(); ++ i)
        {
            if (::nano::getbit(mask, samples(i)))
            {
                values.vector(i) = tensor.vector(samples(i)).template cast<tensor_size_t>();
            }
        }
    }
    else
    {
        critical0("cannot access multi-label feature <", feature.name(), ">!");
    }
}

template <typename tstats, typename tscalar>
tstats stats(
    const feature_t& feature,
    const scalar_storage_t<tscalar>& tensor,
    const feature_mask_t& mask,
    const indices_cmap_t& samples)
{
    tstats stats;

    if constexpr (std::is_same<tstats, feature_scalar_stats_t>::value)
    {
        stats.m_min.resize(feature.dims());
        stats.m_max.resize(feature.dims());
        stats.m_mean.resize(feature.dims());
        stats.m_stdev.resize(feature.dims());

        stats.m_count = 0;
        stats.m_mean.zero();
        stats.m_stdev.zero();
        stats.m_min.constant(std::numeric_limits<scalar_t>::max());
        stats.m_max.constant(std::numeric_limits<scalar_t>::lowest());

        for (const auto sample : samples)
        {
            if (::nano::getbit(mask, sample))
            {
                const auto values = tensor.vector(sample).template cast<scalar_t>();

                stats.m_count ++;
                stats.m_mean.array() += values.array();
                stats.m_stdev.array() += values.array().square();
                stats.m_min.array() = stats.m_min.array().min(values.array());
                stats.m_max.array() = stats.m_max.array().max(values.array());
            }
        }

        if (stats.m_count > 1)
        {
            const auto N = stats.m_count;
            stats.m_stdev.array() = ((stats.m_stdev.array() - stats.m_mean.array().square() / N) / (N - 1)).sqrt();
            stats.m_mean.array() /= static_cast<scalar_t>(N);
        }
    }

    else
    {
        critical0("cannot access scalar feature <", feature.name(), ">!");
    }

    return stats;
}

template <typename tstats, typename tscalar>
tstats stats(
    const feature_t& feature,
    const sclass_storage_t<tscalar>& tensor,
    const feature_mask_t& mask,
    const indices_cmap_t& samples)
{
    tstats stats;

    if constexpr (std::is_same<tstats, feature_sclass_stats_t>::value)
    {
        stats.m_class_counts.resize(static_cast<tensor_size_t>(feature.labels().size()));
        stats.m_class_counts.zero();

        for (const auto sample : samples)
        {
            if (::nano::getbit(mask, sample))
            {
                const auto label = static_cast<tensor_size_t>(tensor(sample));

                stats.m_class_counts(label) ++;
            }
        }
    }
    else
    {
        critical0("cannot access single-label feature <", feature.name(), ">!");
    }

    return stats;
}

template <typename tstats, typename tscalar>
tstats stats(
    const feature_t& feature,
    const mclass_storage_t<tscalar>& tensor,
    const feature_mask_t& mask,
    const indices_cmap_t& samples)
{
    tstats stats;

    if constexpr (std::is_same<tstats, feature_mclass_stats_t>::value)
    {
        stats.m_class_counts.resize(static_cast<tensor_size_t>(feature.labels().size()));
        stats.m_class_counts.zero();

        for (const auto sample : samples)
        {
            if (::nano::getbit(mask, sample))
            {
                stats.m_class_counts.array() += tensor.array(sample).template cast<tensor_size_t>();
            }
        }
    }
    else
    {
        critical0("cannot access multi-label feature <", feature.name(), ">!");
    }

    return stats;
}

}

void feature_storage_t::get(indices_cmap_t samples, tensor_mem_t<scalar_t, 4>& values) const
{
    visit([&] (const auto& tensor) { ::get(m_feature, tensor, m_mask, samples, values); });
}

void feature_storage_t::get(indices_cmap_t samples, tensor_mem_t<tensor_size_t, 1>& labels) const
{
    visit([&] (const auto& tensor) { ::get(m_feature, tensor, m_mask, samples, labels); });
}

void feature_storage_t::get(indices_cmap_t samples, tensor_mem_t<tensor_size_t, 2>& labels) const
{
    visit([&] (const auto& tensor) { ::get(m_feature, tensor, m_mask, samples, labels); });
}

bool feature_storage_t::optional() const
{
    const auto samples = this->samples();
    const auto bytes = samples / 8;

    for (tensor_size_t byte = 0; byte < bytes; ++ byte)
    {
        if (m_mask(byte) != 0xFF)
        {
            return true;
        }
    }

    for (tensor_size_t sample = 8 * bytes; sample < samples; ++ sample)
    {
        if (!::nano::getbit(m_mask, sample))
        {
            return true;
        }
    }

    return false;
}

feature_scalar_stats_t feature_storage_t::scalar_stats(indices_cmap_t samples) const
{
    return visit([&] (const auto& tensor)
    {
        return ::stats<feature_scalar_stats_t>(m_feature, tensor, m_mask, samples);
    });
}

feature_sclass_stats_t feature_storage_t::sclass_stats(indices_cmap_t samples) const
{
    return visit([&] (const auto& tensor)
    {
        return ::stats<feature_sclass_stats_t>(m_feature, tensor, m_mask, samples);
    });
}

feature_mclass_stats_t feature_storage_t::mclass_stats(indices_cmap_t samples) const
{
    return visit([&] (const auto& tensor)
    {
        return ::stats<feature_mclass_stats_t>(m_feature, tensor, m_mask, samples);
    });
}
