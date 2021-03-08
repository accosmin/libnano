#include <nano/dataset/memory_dataset.h>
#include <nano/dataset/memory_iterator.h>

using namespace nano;

memory_dataset_t::memory_dataset_t() = default;

void memory_dataset_t::resize(tensor_size_t samples, const features_t& features)
{
    this->resize(samples, features, string_t::npos);
}

void memory_dataset_t::resize(tensor_size_t samples, const features_t& features, size_t target)
{
    std::map<feature_type, tensor_size_t> size_storage;
    const auto update_size_storage = [&] (feature_type type, auto size)
    {
        const auto begin = size_storage[type];
        const auto end = begin + static_cast<tensor_size_t>(size);
        size_storage[type] = end;
        return std::make_pair(begin, end);
    };

    m_storage_type.resize(features.size());
    m_storage_range.resize(static_cast<tensor_size_t>(features.size()), 2);

    for (size_t i = 0, size = features.size(); i < size; ++ i)
    {
        const auto& feature = features[i];

        feature_type type;
        std::pair<tensor_size_t, tensor_size_t> range;
        switch (feature.type())
        {
        case feature_type::mclass:
            type = feature_type::uint8;
            range = update_size_storage(type, feature.classes());
            break;

        case feature_type::sclass:
            type =
                (feature.classes() <= (tensor_size_t(1) << 8)) ? feature_type::uint8 :
                (feature.classes() <= (tensor_size_t(1) << 16)) ? feature_type::uint16 :
                (feature.classes() <= (tensor_size_t(1) << 32)) ? feature_type::uint32 : feature_type::uint64;
            range = update_size_storage(type, 1);
            break;

        default:
            type = feature.type();
            range = update_size_storage(type, ::nano::size(feature.dims()));
            break;
        }

        m_storage_type[i] = type;

        const auto [begin, end] = range;
        m_storage_range(static_cast<tensor_size_t>(i), 0) = begin;
        m_storage_range(static_cast<tensor_size_t>(i), 1) = end;
    }

    m_samples = samples;
    m_features = features;
    m_target = (target < features.size()) ? static_cast<tensor_size_t>(target) : m_storage_range.size();

    m_storage_f32.resize(size_storage[feature_type::float32], samples);
    m_storage_f64.resize(size_storage[feature_type::float64], samples);
    m_storage_i08.resize(size_storage[feature_type::int8], samples);
    m_storage_i16.resize(size_storage[feature_type::int16], samples);
    m_storage_i32.resize(size_storage[feature_type::int32], samples);
    m_storage_i64.resize(size_storage[feature_type::int64], samples);
    m_storage_u08.resize(size_storage[feature_type::uint8], samples);
    m_storage_u16.resize(size_storage[feature_type::uint16], samples);
    m_storage_u32.resize(size_storage[feature_type::uint32], samples);
    m_storage_u64.resize(size_storage[feature_type::uint64], samples);

    m_storage_mask.resize(static_cast<tensor_size_t>(features.size()), (samples + 7) / 8);
    m_storage_mask.zero();
}

rfeature_dataset_iterator_t memory_dataset_t::feature_iterator(indices_t samples) const
{
    return std::make_unique<memory_feature_dataset_iterator_t>(*this, std::move(samples));
}

rflatten_dataset_iterator_t memory_dataset_t::flatten_iterator(indices_t samples) const
{
    return std::make_unique<memory_flatten_dataset_iterator_t>(*this, std::move(samples));
}

feature_t memory_dataset_t::target() const
{
    if (!has_target())
    {
        return feature_t{};
    }
    else
    {
        return visit_target([] (const feature_t& feature, const auto&, const auto&)
        {
            return feature;
        });
    }
}

tensor3d_dims_t memory_dataset_t::target_dims() const
{
    if (!has_target())
    {
        return make_dims(0, 0, 0);
    }
    else
    {
        return visit_target([] (const feature_t& feature, const auto&, const auto&)
        {
            switch (feature.type())
            {
            case feature_type::sclass:  return make_dims(feature.classes(), 1, 1);
            case feature_type::mclass:  return make_dims(feature.classes(), 1, 1);
            default:                    return feature.dims();
            }
        });
    }
}

tensor4d_cmap_t memory_dataset_t::targets(const indices_cmap_t& samples, tensor4d_t& buffer) const
{
    visit_target([&] (const feature_t& feature, const auto& tensor, const auto& mask)
    {
        if constexpr (tensor.rank() == 1)
        {
            buffer.resize(samples.size(), feature.classes(), 1, 1);
            buffer.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            loop_masked(mask, samples, [&] (tensor_size_t i, tensor_size_t sample)
            {
                buffer.array(i).setConstant(-1.0);
                buffer(i, tensor(sample), 0, 0) = +1.0;
            });
        }
        else if constexpr (tensor.rank() == 2)
        {
            buffer.resize(samples.size(), feature.classes(), 1, 1);
            buffer.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            loop_masked(mask, samples, [&] (tensor_size_t i, tensor_size_t sample)
            {
                buffer.array(i) = tensor.array(sample).template cast<scalar_t>() * 2.0 - 1.0;
            });
        }
        else
        {
            buffer.resize(cat_dims(samples.size(), feature.dims()));
            buffer.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            loop_masked(mask, samples, [&] (tensor_size_t i, tensor_size_t sample)
            {
                buffer.array(i) = tensor.array(sample).template cast<scalar_t>();
            });
        }
    });
    return buffer.tensor();
}
