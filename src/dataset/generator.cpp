#include <nano/dataset/generator.h>
#include <nano/dataset/iterator.h>

using namespace nano;

template <typename tscalar, size_t trank, typename... tindices>
static auto resize_and_map(tensor_mem_t<tscalar, trank>& buffer, tindices... dims)
{
    if (buffer.size() < ::nano::size(make_dims(dims...)))
    {
        buffer.resize(dims...);
    }
    return map_tensor(buffer.data(), dims...);
}

generator_t::generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
    m_dataset(dataset),
    m_samples(samples)
{
}

sclass_cmap_t generator_t::select(tensor_size_t f, indices_cmap_t, sclass_mem_t& buffer) const
{
    critical0("generator_t: unhandled categorical feature <", f, ":", feature(f), ">!");
    return buffer.tensor();
}

scalar_cmap_t generator_t::select(tensor_size_t f, indices_cmap_t, scalar_mem_t& buffer) const
{
    critical0("generator_t: unhandled scalar feature <", f, ":", feature(f), ">!");
    return buffer.tensor();
}

struct_cmap_t generator_t::select(tensor_size_t f, indices_cmap_t, struct_mem_t& buffer) const
{
    critical0("generator_t: unhandled structured feature <", f, ":", feature(f), ">!");
    return buffer.tensor();
}

identity_generator_t::identity_generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
    generator_t(dataset, samples)
{
    tensor_size_t count = 0;
    for (tensor_size_t f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::sclass:
            count += 1;
            m_columns += feature.classes();
            break;

        case feature_type::mclass:
            count += feature.classes();
            m_columns += feature.classes();
            break;

        default:
            {
                const auto size = ::nano::size(feature.dims());
                count += size;
                if (size > 1)
                {
                    count += 1;
                }
                m_columns += size;
            }
            break;
        }
    }

    m_mapping.resize(count, 2);
    for (tensor_size_t i = 0, f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::sclass:
            map1(i, f, -1);
            break;

        case feature_type::mclass:
            mapN(i, f, feature.classes());
            break;

        default:
            map1(i, f, -1);
            if (size(feature.dims()) > 1)
            {
                mapN(i, f, size(feature.dims()));
            }
            break;
        }
    }
}

feature_t identity_generator_t::feature(tensor_size_t i) const
{
    const auto component = mapped_component(i);
    switch (const auto& feature = mapped_feature(i); feature.type())
    {
    case feature_type::sclass:
        assert(component < 0);
        return feature;

    case feature_type::mclass:
        assert(component >= 0);
        return feature_t{scat(feature.name(), "_", feature.labels()[component])}.sclass(2);

    default:
        if (component == -1)
        {
            return feature;
        }
        else
        {
            return feature_t{scat(feature.name(), "_", component)}.scalar(feature.type(), make_dims(1, 1, 1));
        }
    }
}

void identity_generator_t::original(tensor_size_t i, cluster_t& original_features) const
{
    assert(original_features.groups() == 1);
    assert(original_features.samples() == dataset().features());

    original_features.assign(mapped_index(i), 0);
}

sclass_cmap_t identity_generator_t::select(tensor_size_t i, indices_cmap_t samples, sclass_mem_t& buffer) const
{
    return dataset().visit_inputs(mapped_index(i), [&] (const auto&, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 1)
        {
            const auto storage = resize_and_map(buffer, samples.size());
            for (auto it = make_iterator(data, mask, samples); it; ++ it)
            {
                if (const auto [index, sample, given, label] = *it; given)
                {
                    storage(index) = static_cast<int32_t>(label);
                }
                else
                {
                    storage(index) = -1;
                }
            }
            return sclass_cmap_t{storage};
        }
        else if constexpr (data.rank() == 2)
        {
            const auto component = mapped_component(i);
            const auto storage = resize_and_map(buffer, samples.size());
            for (auto it = make_iterator(data, mask, samples); it; ++ it)
            {
                if (const auto [index, sample, given, hits] = *it; given)
                {
                    storage(index) = hits(component);
                }
                else
                {
                    storage(index) = -1;
                }
            }
            return sclass_cmap_t{storage};
        }
        else
        {
            return generator_t::select(i, samples, buffer);
        }
    });
}

scalar_cmap_t identity_generator_t::select(tensor_size_t i, indices_cmap_t samples, scalar_mem_t& buffer) const
{
    return dataset().visit_inputs(mapped_index(i), [&] (const auto& feature, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 4)
        {
            auto component = mapped_component(i);
            if (component < 0)
            {
                if (size(feature.dims()) > 1)
                {
                    return generator_t::select(i, samples, buffer);
                }
                else
                {
                    component = 0;
                }
            }
            const auto storage = resize_and_map(buffer, samples.size());
            for (auto it = make_iterator(data, mask, samples); it; ++ it)
            {
                if (const auto [index, sample, given, values] = *it; given)
                {
                    storage(index) = static_cast<scalar_t>(values(component));
                }
                else
                {
                    storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                }
            }
            return scalar_cmap_t{storage};
        }
        else
        {
            return generator_t::select(i, samples, buffer);
        }
    });
}

struct_cmap_t identity_generator_t::select(tensor_size_t i, indices_cmap_t samples, struct_mem_t& buffer) const
{
    return dataset().visit_inputs(mapped_index(i), [&] (const auto& feature, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 4)
        {
            const auto component = mapped_component(i);
            if (component >= 0 || size(feature.dims()) <= 1)
            {
                return generator_t::select(i, samples, buffer);
            }
            const auto [dim1, dim2, dim3] = feature.dims();
            const auto storage = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
            for (auto it = make_iterator(data, mask, samples); it; ++ it)
            {
                if (const auto [index, sample, given, values] = *it; given)
                {
                    storage.array(index) = values.array().template cast<scalar_t>();
                }
                else
                {
                    storage.tensor(index).constant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return struct_cmap_t{storage};
        }
        else
        {
            return generator_t::select(i, samples, buffer);
        }
    });
}

void identity_generator_t::flatten(indices_cmap_t samples, tensor2d_map_t buffer, tensor_size_t column_offset) const
{
    auto storage = buffer.matrix();
    for (tensor_size_t f = 0, features = dataset().features(); f < features; ++ f)
    {
        dataset().visit_inputs(f, [&] (const auto& feature, const auto& data, const auto& mask)
        {
            const auto classes = feature.classes();
            const auto scalars = size(feature.dims());

            if constexpr (data.rank() == 1)
            {
                for (auto it = make_iterator(data, mask, samples); it; ++ it)
                {
                    if (const auto [index, sample, given, label] = *it; given)
                    {
                        auto segment = storage.row(index).segment(column_offset, classes);
                        segment.setConstant(-1.0);
                        segment(label) = +1.0;
                    }
                    else
                    {
                        auto segment = storage.row(index).segment(column_offset, classes);
                        segment.setConstant(+0.0);
                    }
                }
                column_offset += classes;
            }
            else if constexpr (data.rank() == 2)
            {
                for (auto it = make_iterator(data, mask, samples); it; ++ it)
                {
                    if (const auto [index, sample, given, hits] = *it; given)
                    {
                        auto segment = storage.row(index).segment(column_offset, classes);
                        segment.array() = 2.0 * hits.array().template cast<scalar_t>() - 1.0;
                    }
                    else
                    {
                        auto segment = storage.row(index).segment(column_offset, classes);
                        segment.setConstant(+0.0);
                    }
                }
                column_offset += classes;
            }
            else
            {
                for (auto it = make_iterator(data, mask, samples); it; ++ it)
                {
                    if (const auto [index, sample, given, values] = *it; given)
                    {
                        auto segment = storage.row(index).segment(column_offset, scalars);
                        segment.array() = values.array().template cast<scalar_t>();
                    }
                    else
                    {
                        auto segment = storage.row(index).segment(column_offset, scalars);
                        segment.setConstant(+0.0);
                    }
                }
                column_offset += scalars;
            }
        });
    }
}

dataset_generator_t::dataset_generator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
}

void dataset_generator_t::update()
{
    const auto features = std::accumulate(
        m_generators.begin(), m_generators.end(), tensor_size_t(0),
        [] (tensor_size_t features, const rgenerator_t& generator) { return features + generator->features(); });

    m_mapping.resize(features, 2);

    tensor_size_t offset = 0, index = 0;
    for (const auto& generator : m_generators)
    {
        const auto features = generator->features();
        for (tensor_size_t feature = 0; feature < features; ++ feature)
        {
            m_mapping(offset + feature, 0) = index;
            m_mapping(offset + feature, 1) = feature;
        }

        ++ index;
        offset += features;
    }

}

tensor_size_t dataset_generator_t::features() const
{
    return m_mapping.size<0>();
}

feature_t dataset_generator_t::feature(tensor_size_t feature) const
{
    assert(feature >= 0 && feature < features());

    const auto& generator = m_generators[static_cast<size_t>(m_mapping(feature, 0))];
    return generator->feature(m_mapping(feature, 1));
}

indices_t dataset_generator_t::original_features(const indices_t& features) const
{
    cluster_t original_features(m_dataset.features(), 1);

    for (const auto& feature : features)
    {
        const auto& generator = m_generators[static_cast<size_t>(m_mapping(feature, 0))];
        generator->original(m_mapping(feature, 1), original_features);
    }

    return original_features.indices(0);
}

sclass_cmap_t dataset_generator_t::select(tensor_size_t feature, indices_cmap_t samples, sclass_mem_t& buffer) const
{
    const auto& generator = m_generators[static_cast<size_t>(m_mapping(feature, 0))];
    return generator->select(m_mapping(feature, 1), samples, buffer);
}

scalar_cmap_t dataset_generator_t::select(tensor_size_t feature, indices_cmap_t samples, scalar_mem_t& buffer) const
{
    const auto& generator = m_generators[static_cast<size_t>(m_mapping(feature, 0))];
    return generator->select(m_mapping(feature, 1), samples, buffer);
}

struct_cmap_t dataset_generator_t::select(tensor_size_t feature, indices_cmap_t samples, struct_mem_t& buffer) const
{
    const auto& generator = m_generators[static_cast<size_t>(m_mapping(feature, 0))];
    return generator->select(m_mapping(feature, 1), samples, buffer);
}

tensor_size_t dataset_generator_t::columns() const
{
    return std::accumulate(
        m_generators.begin(), m_generators.end(), tensor_size_t(0),
        [] (tensor_size_t columns, const rgenerator_t& generator) { return columns + generator->columns(); });
}

tensor2d_cmap_t dataset_generator_t::flatten(tensor_range_t sample_range, tensor2d_t& buffer) const
{
    const auto storage = resize_and_map(buffer, sample_range.size(), columns());

    tensor_size_t offset = 0;
    for (const auto& generator : m_generators)
    {
        generator->flatten(m_samples.slice(sample_range), storage, offset);
        offset += generator->columns();
    }
    return storage;
}

feature_t dataset_generator_t::target() const
{
    switch (m_dataset.type())
    {
    case task_type::unsupervised:
        return feature_t{};

    default:
        return m_dataset.visit_target([] (const feature_t& feature, const auto&, const auto&)
        {
            return feature;
        });
    }
}

tensor3d_dims_t dataset_generator_t::target_dims() const
{
    switch (m_dataset.type())
    {
    case task_type::unsupervised:
        return make_dims(0, 0, 0);

    default:
        return m_dataset.visit_target([] (const feature_t& feature, const auto&, const auto&)
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

tensor4d_cmap_t dataset_generator_t::targets(tensor_range_t sample_range, tensor4d_t& buffer) const
{
    const auto samples = m_samples.slice(sample_range);

    return m_dataset.visit_target([&] (const feature_t& feature, const auto& tensor, const auto& mask)
    {
        if constexpr (tensor.rank() == 1)
        {
            const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
            for (auto it = feature_iterator_t{tensor, mask, samples}; it; ++ it)
            {
                if (const auto [index, sample, given, label] = *it; given)
                {
                    storage.array(index).setConstant(-1.0);
                    storage.array(index)(label) = +1.0;
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_cmap_t{storage};
        }
        else if constexpr (tensor.rank() == 2)
        {
            const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
            for (auto it = feature_iterator_t{tensor, mask, samples}; it; ++ it)
            {
                if (const auto [index, sample, given, hits] = *it; given)
                {
                    storage.array(index) = hits.array().template cast<scalar_t>() * 2.0 - 1.0;
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_cmap_t{storage};
        }
        else
        {
            const auto [dim1, dim2, dim3] = feature.dims();
            const auto storage = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
            for (auto it = feature_iterator_t{tensor, mask, samples}; it; ++ it)
            {
                if (const auto [index, sample, given, values] = *it; given)
                {
                    storage.array(index) = values.array().template cast<scalar_t>();
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_cmap_t{storage};
        }
    });
}
