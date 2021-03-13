#include <nano/dataset/dataset.h>

using namespace nano;

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

    m_storage_f32.zero();
    m_storage_f64.zero();
    m_storage_i08.zero();
    m_storage_i16.zero();
    m_storage_i32.zero();
    m_storage_i64.zero();
    m_storage_u08.zero();
    m_storage_u16.zero();
    m_storage_u32.zero();
    m_storage_u64.zero();

    m_storage_mask.resize(static_cast<tensor_size_t>(features.size()), (samples + 7) / 8);
    m_storage_mask.zero();
}

task_type memory_dataset_t::type() const
{
    return  has_target() ?
            static_cast<task_type>(m_features[static_cast<size_t>(m_target)]) :
            task_type::unsupervised;
}
