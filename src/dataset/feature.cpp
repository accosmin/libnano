#include <nano/dataset/feature.h>

using namespace nano;

feature_t::feature_t() = default;

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

size_t feature_t::set_label(const string_t& label) const
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

feature_t::operator task_type() const
{
    if (!static_cast<bool>(*this))
    {
        return task_type::unsupervised;
    }
    else
    {
        switch (m_type)
        {
        case feature_type::sclass:      return task_type::sclassification;
        case feature_type::mclass:      return task_type::mclassification;
        default:                        return task_type::regression;
        }
    }
}

bool ::nano::operator==(const feature_t& f1, const feature_t& f2)
{
    return  f1.type() == f2.type() &&
            f1.name() == f2.name() &&
            f1.dims() == f2.dims() &&
            f1.labels() == f2.labels();
}

bool ::nano::operator!=(const feature_t& f1, const feature_t& f2)
{
    return  f1.type() != f2.type() ||
            f1.name() != f2.name() ||
            f1.dims() != f2.dims() ||
            f1.labels() != f2.labels();
}

std::ostream& ::nano::operator<<(std::ostream& stream, const feature_t& feature)
{
    stream << "name=" << feature.name() << ",type=" << feature.type() << ",dims=" << feature.dims() << ",labels[";
    for (const auto& label : feature.labels())
    {
        stream << label;
        if (&label != &(*(feature.labels().rbegin())))
        {
            stream << ",";
        }
    }
    return stream << "]";
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
