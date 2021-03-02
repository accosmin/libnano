#include <nano/dataset/memory.h>

using namespace nano;

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
}

feature_scalar_stats_t feature_storage_t::scalar_stats(const indices_cmap_t& samples, const mask_cmap_t& mask) const
{
    return visit([&] (const auto& tensor)
    {
        feature_scalar_stats_t stats;

        if constexpr (tensor.rank() == 4)
        {
            stats.m_min.resize(dims());
            stats.m_max.resize(dims());
            stats.m_mean.resize(dims());
            stats.m_stdev.resize(dims());

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
            critical0("cannot access scalar feature <", name(), ">!");
        }
        return stats;
    });
}

feature_sclass_stats_t feature_storage_t::sclass_stats(const indices_cmap_t& samples, const mask_cmap_t& mask) const
{
    return visit([&] (const auto& tensor)
    {
        feature_sclass_stats_t stats;

        if constexpr (tensor.rank() == 1)
        {
            stats.m_class_counts.resize(static_cast<tensor_size_t>(m_feature.labels().size()));
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
            critical0("cannot access single-label feature <", name(), ">!");
        }
        return stats;
    });
}

feature_mclass_stats_t feature_storage_t::mclass_stats(const indices_cmap_t& samples, const mask_cmap_t& mask) const
{
    return visit([&] (const auto& tensor)
    {
        feature_mclass_stats_t stats;

        if constexpr (tensor.rank() == 2)
        {
            stats.m_class_counts.resize(static_cast<tensor_size_t>(m_feature.labels().size()));
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
            critical0("cannot access multi-label feature <", name(), ">!");
        }
        return stats;
    });
}

memory_dataset_t::memory_dataset_t() = default;

void memory_dataset_t::resize(tensor_size_t samples, const features_t& features)
{
    this->resize(samples, features, string_t::npos);
}

void memory_dataset_t::resize(tensor_size_t samples, const features_t& features, size_t target)
{
    m_target = target;

    m_storage.clear();
    for (const auto& feature : features)
    {
        m_storage.emplace_back(feature, samples);
    }

    m_mask.resize(static_cast<tensor_size_t>(features.size()), (samples + 7) / 8);
    m_mask.zero();
}

rfeature_dataset_iterator_t memory_dataset_t::feature_iterator(indices_t samples) const
{
    return std::make_unique<memory_feature_dataset_iterator_t>(*this, std::move(samples));
}

rflatten_dataset_iterator_t memory_dataset_t::flatten_iterator(indices_t) const
{
    // TODO: return std::make_unique<memory_flatten_dataset_iterator_t>(*this, std::move(samples));
    return nullptr;
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

tensor_size_t memory_feature_dataset_iterator_t::features() const
{
    return m_dataset.features();
}

feature_t memory_feature_dataset_iterator_t::target() const
{
    return m_dataset.tstorage().feature();
}

tensor3d_dims_t memory_feature_dataset_iterator_t::target_dims() const
{
    // TODO: handle classification tasks!!!
    return m_dataset.tstorage().feature().dims();
}

tensor4d_cmap_t memory_feature_dataset_iterator_t::targets(tensor4d_t& buffer) const
{
    const auto& fs = m_dataset.tstorage();
    fs.get(m_samples, m_dataset.tmask(), buffer);
    return buffer.tensor();
}

namespace
{
    template <typename top>
    indices_t filter(const std::vector<feature_storage_t>& storage, size_t target, const top& op)
    {
        tensor_size_t count = 0;
        for (size_t i = 0, size = storage.size(); i < size; ++ i)
        {
            if (i != target && op(storage[i]))
            {
                ++ count;
            }
        }

        indices_t indices(count);

        tensor_size_t feature = 0, index = 0;
        for (size_t i = 0, size = storage.size(); i < size; ++ i)
        {
            if (i != target)
            {
                if (op(storage[i]))
                {
                    indices(index ++) = feature;
                }
                ++ feature;
            }
        }

        return indices;
    }
}

indices_t memory_feature_dataset_iterator_t::scalar_features() const
{
    const auto op = [] (const feature_storage_t& fs)
    {
        return  fs.feature().type() != feature_type::sclass &&
                fs.feature().type() != feature_type::mclass &&
                ::nano::size(fs.feature().dims()) == 1;
    };

    return ::filter(m_dataset.storage(), m_dataset.target(), op);
}

indices_t memory_feature_dataset_iterator_t::struct_features() const
{
    const auto op = [] (const feature_storage_t& fs)
    {
        return  fs.feature().type() != feature_type::sclass &&
                fs.feature().type() != feature_type::mclass &&
                ::nano::size(fs.feature().dims()) > 1;
    };

    return ::filter(m_dataset.storage(), m_dataset.target(), op);
}

indices_t memory_feature_dataset_iterator_t::sclass_features() const
{
    const auto op = [] (const feature_storage_t& fs)
    {
        return  fs.feature().type() == feature_type::sclass;
    };

    return ::filter(m_dataset.storage(), m_dataset.target(), op);
}

indices_t memory_feature_dataset_iterator_t::mclass_features() const
{
    const auto op = [] (const feature_storage_t& fs)
    {
        return  fs.feature().type() == feature_type::mclass;
    };

    return ::filter(m_dataset.storage(), m_dataset.target(), op);
}

feature_t memory_feature_dataset_iterator_t::feature(tensor_size_t feature) const
{
    const auto& fs = m_dataset.istorage(feature);
    return fs.feature();
}

sindices_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t feature, sindices_t& buffer) const
{
    const auto& fs = m_dataset.istorage(feature);
    fs.get(m_samples, m_dataset.imask(feature), buffer);
    return buffer.tensor();
}

mindices_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t feature, mindices_t& buffer) const
{
    const auto& fs = m_dataset.istorage(feature);
    fs.get(m_samples, m_dataset.imask(feature), buffer);
    return buffer.tensor();
}

tensor1d_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t feature, tensor1d_t& buffer) const
{
    const auto& fs = m_dataset.istorage(feature);
    fs.get(m_samples, m_dataset.imask(feature), buffer);
    return buffer.tensor();
}

tensor4d_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t feature, tensor4d_t& buffer) const
{
    const auto& fs = m_dataset.istorage(feature);
    fs.get(m_samples, m_dataset.imask(feature), buffer);
    return buffer.tensor();
}

bool memory_feature_dataset_iterator_t::cache_inputs(int64_t, execution)
{
    // TODO
    return false;
}

bool memory_feature_dataset_iterator_t::cache_targets(int64_t, execution)
{
    // TODO
    return false;
}

bool memory_feature_dataset_iterator_t::cache_inputs(int64_t, indices_cmap_t, execution)
{
    // TODO
    return false;
}
