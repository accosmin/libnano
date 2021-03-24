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

generator1_t::generator1_t(const memory_dataset_t& dataset, const indices_t& samples) :
    generator_t(dataset, samples)
{
}

feature_t generator1_t::feature(tensor_size_t i) const
{
    const auto component = mapped_component(i);
    switch (const auto& feature = mapped_feature(i); feature.type())
    {
    case feature_type::sclass:
        if (component == -1)
        {
            return feature;
        }
        else
        {
            return feature_t{scat(feature.name(), "_", feature.labels()[component])}.sclass(2);
        }

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

void generator1_t::original(tensor_size_t i, cluster_t& original_features) const
{
    assert(original_features.groups() == 1);
    assert(original_features.samples() == dataset().features());

    original_features.assign(mapped_index(i), 0);
}

sclass_generator_t::sclass_generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
    generator1_t(dataset, samples)
{
    tensor_size_t features = 0, columns = 0;
    for (tensor_size_t f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::sclass:  features += 1; columns += feature.classes(); break;
        default:                    break;
        }
    }

    resize(features, columns);
    for (tensor_size_t i = 0, f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::sclass:  map1(i, f, -1); break;
        default:                    break;
        }
    }
}

sclass_cmap_t sclass_generator_t::select(tensor_size_t i, indices_cmap_t samples, sclass_mem_t& buffer) const
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
        else
        {
            return generator_t::select(i, samples, buffer);
        }
    });
}

void sclass_generator_t::flatten(indices_cmap_t samples, tensor2d_map_t buffer, tensor_size_t column_offset) const
{
    auto storage = buffer.matrix();
    for (tensor_size_t i = 0, features = this->features(); i < features; ++ i)
    {
        dataset().visit_inputs(mapped_index(i), [&] (const auto& feature, const auto& data, const auto& mask)
        {
            if constexpr (data.rank() == 1)
            {
                for (auto it = make_iterator(data, mask, samples); it; ++ it)
                {
                    if (const auto [index, sample, given, label] = *it; given)
                    {
                        auto segment = storage.row(index).segment(column_offset, feature.classes());
                        segment.setConstant(+0.0);
                        segment(label) = +1.0;
                    }
                    else
                    {
                        auto segment = storage.row(index).segment(column_offset, feature.classes());
                        segment.setConstant(+0.0);
                    }
                }
                column_offset += feature.classes();
            }
        });
    }
}

sclass2binary_generator_t::sclass2binary_generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
    sclass_generator_t(dataset, samples)
{
    tensor_size_t features = 0, columns = 0;
    for (tensor_size_t f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::sclass:  features += feature.classes(); columns += feature.classes(); break;
        default:                    break;
        }
    }

    resize(features, columns);
    for (tensor_size_t i = 0, f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::sclass:  mapN(i, f, feature.classes()); break;
        default:                    break;
        }
    }
}

sclass_cmap_t sclass2binary_generator_t::select(tensor_size_t i, indices_cmap_t samples, sclass_mem_t& buffer) const
{
    return dataset().visit_inputs(mapped_index(i), [&] (const auto&, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 1)
        {
            const auto component = mapped_component(i);
            const auto storage = resize_and_map(buffer, samples.size());
            for (auto it = make_iterator(data, mask, samples); it; ++ it)
            {
                if (const auto [index, sample, given, label] = *it; given)
                {
                    storage(index) = (static_cast<tensor_size_t>(label) == component) ? 0 : 1;
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

mclass_generator_t::mclass_generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
    generator1_t(dataset, samples)
{
    tensor_size_t features = 0, columns = 0;
    for (tensor_size_t f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::mclass:  features += feature.classes(); columns += feature.classes(); break;
        default:                    break;
        }
    }

    resize(features, columns);
    for (tensor_size_t i = 0, f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::mclass:  mapN(i, f, feature.classes()); break;
        default:                    break;
        }
    }
}

sclass_cmap_t mclass_generator_t::select(tensor_size_t i, indices_cmap_t samples, sclass_mem_t& buffer) const
{
    return dataset().visit_inputs(mapped_index(i), [&] (const auto&, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 2)
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

void mclass_generator_t::flatten(indices_cmap_t samples, tensor2d_map_t buffer, tensor_size_t column_offset) const
{
    auto storage = buffer.matrix();
    for (tensor_size_t i = 0, features = this->features(); i < features; ++ i)
    {
        dataset().visit_inputs(mapped_index(i), [&] (const auto& feature, const auto& data, const auto& mask)
        {
            if constexpr (data.rank() == 2)
            {
                for (auto it = make_iterator(data, mask, samples); it; ++ it)
                {
                    if (const auto [index, sample, given, hits] = *it; given)
                    {
                        auto segment = storage.row(index).segment(column_offset, feature.classes());
                        segment.array() = hits.array().template cast<scalar_t>();
                    }
                    else
                    {
                        auto segment = storage.row(index).segment(column_offset, feature.classes());
                        segment.setConstant(+0.0);
                    }
                }
                column_offset += feature.classes();
            }
        });
    }
}

/*
identity_generator_t::identity_generator_t(const memory_dataset_t& dataset) :
    generator_t(dataset)
{
    tensor_size_t count = 0;
    for (tensor_size_t i = 0, size = m_dataset.features(); i < size; ++ i)
    {
        const auto& feature = m_dataset.feature(i);
        switch (feature.type())
        {
        case feature_type::sclass:  count += 1; break;
        case feature_type::mclass:  count += feature.classes(); break;
        default:                    count += 1 + size(feature.dims()); break;
        }
    }

    m_mapping.resize(count);
    for (tensor_size_t i = 0, f = 0, size = m_dataset.features(); i < size; ++ i)
    {
        const auto& feature = m_dataset.feature(i);
        switch (feature.type())
        {
        case feature_type::sclass:  map1(f, i, -1); break;
        case feature_type::mclass:  mapN(f, i, feature.classes()); break;
        default:                    mapN(f, i, size(feature.dims())); map1(f, i, -1); break;
        }
    }
}

tensor_size_t identity_generator_t::features() const
{
    return m_mapping.size<0>();
}

feature_t identity_generator_t::feature(tensor_size_t f) const
{
    const auto& feature = m_dataset.feature(m_mapping(f, 0));
    const auto component = m_mapping(f, 1);

    switch (feature.type())
    {
    case feature_type::sclass:
        return feature;

    case feature_type::mclass:
        return feature_t{scat(feature.name(), "_", component)}.sclass(2);

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

void identity_generator_t::original(tensor_size_t f, cluster_t& original_features) const
{
    assert(original_features.groups() == 1);
    assert(original_features.samples() == m_dataset.features());

    original_features.assign(m_mapping(f, 0), 0);
}

sclass_cmap_t identity_generator_t::select(tensor_size_t f, indices_cmap_t samples, sclass_mem_t& buffer) const
{
    return m_dataset.visit_inputs(m_mapping(f, 0), [&] (const feature_t& feature, const auto& data, const auto& mask)
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
            return storage;
        }
        else if constexpr (data.rank() == 2)
        {
            const auto component = m_mapping(f, 1);
            const auto storage = resize_and_map(buffer, samples.size());
            for (auto it = make_iterator(data, mask, samples); it; ++ it)
            {
                if (const auto [index, sample, given, labels] = *it; given)
                {
                    storage(index) = static_cast<int32_t>(labels(component));
                }
                else
                {
                    storage(index) = -1;
                }
            }
            return storage;
        }
        else
        {
            critical0("identity_generator_t: selected invalid feature type <", feature, ">!");
            return buffer.tensor();
        }
    });
}

scalar_cmap_t identity_generator_t::select(tensor_size_t f, indices_cmap_t samples, scalar_mem_t& buffer) const
{
    return m_dataset.visit_inputs(m_mapping(f, 0), [&] (const feature_t& feature, const auto& data, const mask_cmap_t& mask)
    {
        if constexpr (data.rank() == 1)
        {
            const auto storage = resize_and_map(buffer, samples.size());
            for (auto it = make_iterator(data, mask, samples); it; ++ it)
            {
                if (const auto [index, sample, given, value] = *it; given)
                {
                    storage(index) = static_cast<scalar_t>(value);
                }
                else
                {
                    storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                }
            }
            return storage;
        }
        else if constexpr (data.rank() == 4)
        {
            const auto component = m_mapping(f, 1);
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
            return storage;
        }
        else
        {
            critical0("identity_generator_t: selected invalid feature type <", feature, ">!");
            return buffer.tensor();
        }
    });
}

struct_cmap_t identity_generator_t::select(tensor_size_t feature, indices_cmap_t samples, struct_mem_t& buffer) const
{
    return m_dataset.visit_inputs(feature, [&] (const feature_t& feature, const auto& data, const mask_cmap_t& mask)
    {
        if constexpr (data.rank() == 4)
        {
            const auto [dim1, dim2, dim3] = feature.dims();
            const auto storage = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
            for (auto it = make_iterator(data, mask, samples); it; ++ it)
            {
                if (const auto [index, sample, given, values] = *it; given)
                {
                    storage.vector(index) = values.vector().template cast<scalar_t>();
                }
                else
                {
                    storage.tensor(index).constant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return storage;
        }
        else
        {
            critical0("identity_generator_t: selected invalid feature type <", feature, ">!");
            return buffer.tensor();
        }
    });
}

tensor_size_t identity_generator_t::flatten_size() const
{
    tensor_size_t size = 0;
    for (tensor_size_t i = 0; i < m_dataset.features(); ++ f)
    {
        const auto& feature = m_dataset.feature(i);
        switch (feature.type())
        {
        case feature::sclass:   size += feature.classes(); break;
        case feature::mclass:   size += feature.classes(); break;
        default:                size += ::nano::size(feature.dims()); break;
        }
    }
    return size;
}

void identity_generator_t::flatten(indices_cmap_t samples, tensor2d_cmap_t inputs) const
{
    assert(inputs.dims() == make_dims(samples.size(), flatten_size()));

    for (tensor_size_t i = 0, offset = 0; i < m_dataset.features(); ++ f)
    {
        m_dataset.visit_inputs(i, [&] (const feature_t& feature, const auto& data, const mask_cmap_t& mask)
        {
            if constexpr (data.rank() == 1)
            {
                for (auto it = make_iterator(data, mask, samples); it; ++ it)
                {
                    if (const auto [index, sample, given, values] = *it; given)
                    {
                        storage.vector(index) = values.vector().template cast<scalar_t>();
                    }
                    else
                    {
                        storage.tensor(index).constant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
                offset += feature.classes();
            }
            return storage;
            }
        const auto& feature = m_dataset.feature(i);
        switch (feature.type())
        {
        case feature::sclass:   size += feature.classes(); break;
        case feature::mclass:   size += feature.classes(); break;
        default:                size += ::nano::size(feature.dims()); break;
        }
    }
    return size;
}
*/

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
