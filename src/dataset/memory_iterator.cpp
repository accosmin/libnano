#include <nano/dataset/memory_iterator.h>
#include <nano/dataset/dataset.h>

using namespace nano;

template <typename tmapping>
static void map_component(tmapping& mapping, tensor_size_t& f, tensor_size_t original, tensor_size_t component)
{
    mapping(f, 0) = original;
    mapping(f ++, 1) = component;
}

template <typename tmapping>
static void map_components(tmapping& mapping, tensor_size_t& f, tensor_size_t original, tensor_size_t components)
{
    for (tensor_size_t component = 0; component < components; ++ component)
    {
        map_component(mapping, f, original, component);
    }
}

memory_feature_dataset_iterator_t::memory_feature_dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
    tensor_size_t features = 0;
    for (tensor_size_t i = 0, size = m_dataset.features(); i < size; ++ i)
    {
        const auto& feature = m_dataset.feature(i);
        switch (feature.type())
        {
        case feature_type::sclass:
            {
                const auto classes = feature.classes();
                features += (classes <= 2) ? classes : (classes + 1);
            }
            break;

        case feature_type::mclass:
            features += feature.classes();
            break;

        default:
            {
                const auto values = ::nano::size(feature.dims());
                features += (values == 1) ? tensor_size_t{1} : (values + 1);
            }
            break;
        }
    }

    m_mapping.resize(features, 2);
    for (tensor_size_t i = 0, f = 0, size = m_dataset.features(); i < size; ++ i)
    {
        const auto& feature = m_dataset.feature(i);
        switch (feature.type())
        {
        case feature_type::sclass:
            if (feature.classes() <= 2)
            {
                map_component(m_mapping, f, i, -1);
            }
            else
            {
                map_component(m_mapping, f, i, -1);
                map_components(m_mapping, f, i, feature.classes());
            }
            break;

        case feature_type::mclass:
            map_components(m_mapping, f, i, feature.classes());
            break;

        default:
            {
                const auto values = ::nano::size(feature.dims());
                if (values == 1)
                {
                    map_component(m_mapping, f, i, -1);
                }
                else
                {
                    map_component(m_mapping, f, i, -1);
                    map_components(m_mapping, f, i, values);
                }
            }
        }
    }
}

const indices_t& memory_feature_dataset_iterator_t::samples() const
{
    return m_samples;
}

tensor_size_t memory_feature_dataset_iterator_t::features() const
{
    return m_mapping.size<0>();
}

feature_t memory_feature_dataset_iterator_t::feature(tensor_size_t f) const
{
    const auto& feature = m_dataset.feature(m_mapping(f, 0));
    const auto component = m_mapping(f, 1);

    switch (feature.type())
    {
    case feature_type::sclass:
        if (component == -1)
        {
            return feature;
        }
        else
        {
            return feature_t{scat(feature.name(), "_", component)}.sclass(2);
        }

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

feature_t memory_feature_dataset_iterator_t::original_feature(tensor_size_t f) const
{
    return m_dataset.feature(m_mapping(f, 0));
}

feature_t memory_feature_dataset_iterator_t::target() const
{
    return m_dataset.target();
}

tensor3d_dims_t memory_feature_dataset_iterator_t::target_dims() const
{
    return m_dataset.target_dims();
}

tensor4d_cmap_t memory_feature_dataset_iterator_t::targets(tensor4d_t& buffer) const
{
    return m_dataset.targets(m_samples, buffer);
}

indices_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t f, indices_t& buffer) const
{
    m_dataset.visit_inputs(m_mapping(f, 0), [&] (const feature_t& feature, const auto& tensor, const auto& mask)
    {
        const auto component = m_mapping(f, 1);
        if constexpr (tensor.rank() == 1)
        {
            buffer.resize(m_samples.size());
            buffer.constant(-1);
            loop_masked(mask, m_samples, [&] (tensor_size_t i, tensor_size_t sample)
            {
                buffer(i) = (component == -1) ? tensor(sample) : (static_cast<tensor_size_t>(tensor(sample)) == component ? 1 : 0);
            });
        }
        else if constexpr (tensor.rank() == 2)
        {
            buffer.resize(m_samples.size());
            buffer.constant(-1);
            loop_masked(mask, m_samples, [&] (tensor_size_t i, tensor_size_t sample)
            {
                buffer(i) = tensor(sample, component);
            });
        }
        else
        {
            critical0("in-memory dataset: unexpected feature type <", feature.name(), ">!");
        }
    });
    return buffer.tensor();
}

tensor1d_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t f, tensor1d_t& buffer) const
{
    m_dataset.visit_inputs(m_mapping(f, 0), [&] (const feature_t& feature, const auto& tensor, const auto& mask)
    {
        const auto component = m_mapping(f, 1);
        if constexpr (tensor.rank() == 4)
        {
            if (component == -1)
            {
                critical0("in-memory dataset: unexpected feature type <", feature.name(), ">!");
            }
            const auto matrix = tensor.reshape(m_samples.size(), -1);
            buffer.resize(m_samples.size());
            buffer.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            loop_masked(mask, m_samples, [&] (tensor_size_t i, tensor_size_t sample)
            {
                buffer(i) = static_cast<scalar_t>(matrix(sample, component));
            });
        }
        else
        {
            critical0("in-memory dataset: unexpected feature type <", feature.name(), ">!");
        }
    });
    return buffer.tensor();
}

tensor4d_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t f, tensor4d_t& buffer) const
{
    m_dataset.visit_inputs(m_mapping(f, 0), [&] (const feature_t& feature, const auto& tensor, const auto& mask)
    {
        const auto component = m_mapping(f, 1);
        if constexpr (tensor.rank() == 4)
        {
            if (component != -1)
            {
                critical0("in-memory dataset: unexpected feature type <", feature.name(), ">!");
            }
            buffer.resize(cat_dims(m_samples.size(), feature.dims()));
            buffer.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            loop_masked(mask, m_samples, [&] (tensor_size_t i, tensor_size_t sample)
            {
                buffer.vector(i) = tensor.vector(sample).template cast<scalar_t>();
            });
        }
        else
        {
            critical0("in-memory dataset: unexpected feature type <", feature.name(), ">!");
        }
    });
    return buffer.tensor();
}

memory_flatten_dataset_iterator_t::memory_flatten_dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
    tensor_size_t features = 0;
    for (tensor_size_t i = 0, size = m_dataset.features(); i < size; ++ i)
    {
        const auto& feature = m_dataset.feature(i);
        switch (feature.type())
        {
        case feature_type::sclass:
            features += feature.classes();
            break;

        case feature_type::mclass:
            features += feature.classes();
            break;

        default:
            features += ::nano::size(feature.dims());
            break;
        }
    }

    m_mapping.resize(features, 2);
    for (tensor_size_t i = 0, f = 0, size = m_dataset.features(); i < size; ++ i)
    {
        const auto& feature = m_dataset.feature(i);
        switch (feature.type())
        {
        case feature_type::sclass:
            map_components(m_mapping, f, i, feature.classes());
            break;

        case feature_type::mclass:
            map_components(m_mapping, f, i, feature.classes());
            break;

        default:
            map_components(m_mapping, f, i, ::nano::size(feature.dims()));
        }
    }
}

const indices_t& memory_flatten_dataset_iterator_t::samples() const
{
    return m_samples;
}

feature_t memory_flatten_dataset_iterator_t::original_feature(tensor_size_t input) const
{
    return m_dataset.feature(m_mapping(input, 0));
}

feature_t memory_flatten_dataset_iterator_t::target() const
{
    return m_dataset.target();
}

tensor3d_dims_t memory_flatten_dataset_iterator_t::target_dims() const
{
    return m_dataset.target_dims();
}

tensor4d_cmap_t memory_flatten_dataset_iterator_t::targets(tensor_range_t range, tensor4d_t& buffer) const
{
    return m_dataset.targets(m_samples.slice(range), buffer);
}

tensor1d_dims_t memory_flatten_dataset_iterator_t::inputs_dims() const
{
    return make_dims(m_mapping.size<0>());
}

tensor2d_cmap_t memory_flatten_dataset_iterator_t::inputs(tensor_range_t range, tensor2d_t& buffer) const
{
    buffer.resize(range.size(), m_mapping.size<0>());
    buffer.zero();

    const auto samples = m_samples.slice(range);
    for (tensor_size_t ff = 0, f = 0, size = m_dataset.features(); ff < size; ++ ff)
    {
        m_dataset.visit_inputs(ff, [&] (const feature_t& feature, const auto& tensor, const auto& mask)
        {
            if constexpr (tensor.rank() == 1)
            {
                loop_masked(mask, samples, [&] (tensor_size_t i, tensor_size_t sample)
                {
                    for (tensor_size_t c = 0; c < feature.classes(); ++ c)
                    {
                        buffer(i, f ++) = static_cast<tensor_size_t>(tensor(sample)) == c ? +1.0 : -1.0;
                    }
                });
            }
            else if constexpr (tensor.rank() == 2)
            {
                loop_masked(mask, samples, [&] (tensor_size_t i, tensor_size_t sample)
                {
                    for (tensor_size_t c = 0; c < feature.classes(); ++ c)
                    {
                        buffer(i, f ++) = tensor(sample, c) != 0U ? +1.0 : -1.0;
                    }
                });
            }
            else
            {
                loop_masked(mask, samples, [&] (tensor_size_t i, tensor_size_t sample)
                {
                    buffer.vector(i).segment(f, ::nano::size(feature.dims())) = tensor.vector(sample).template cast<scalar_t>();
                    f += ::nano::size(feature.dims());
                });
            }
        });
    }

    return buffer.tensor();
}

tensor2d_t memory_flatten_dataset_iterator_t::normalize(normalization) const
{
    tensor2d_t weights(m_mapping.size<0>(), 2);
    weights.matrix().col(0).array() = 1.0;
    weights.matrix().col(1).array() = 0.0;
    // TODO
    return weights;
}
