#include <nano/dataset/generators.h>

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

identity_generator_t::identity_generator_t(const memory_dataset_t& dataset) :
    generator_t(dataset)
{
    tensor_size_t features = 0, columns = 0;
    for (tensor_size_t f = 0, fsize = dataset.features(); f < fsize; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::sclass:
        case feature_type::mclass:
            features += 1;
            columns += feature.classes();
            break;

        default:
            features += 1;
            columns += ::nano::size(feature.dims());
            break;
        }
    }

    m_column_mapping.resize(columns, 1);
    m_feature_mapping.resize(features, 4);

    for (tensor_size_t c = 0, f = 0, features = dataset.features(); f < features; ++ f)
    {
        switch (const auto& feature = dataset.feature(f); feature.type())
        {
        case feature_type::sclass:
        case feature_type::mclass:
            m_feature_mapping(f, 0) = f;
            m_feature_mapping(f, 1) = 0;
            for ( ; c < feature.classes(); ++ c)
            {
                m_column_mapping(c, 0) = f;
            }
            break;

        default:
            m_feature_mapping(f, 0) = f;
            m_feature_mapping(f, 1) = 0;
            for ( ; c < ::nano::size(feature.dims()); ++ c)
            {
                m_column_mapping(c, 0) = f;
            }
            break;
        }
    }
}

feature_t identity_generator_t::feature(tensor_size_t feature) const
{
    return dataset().feature(m_feature_mapping(feature, 0));
}

tensor_size_t identity_generator_t::column2feature(tensor_size_t column) const
{
    return m_column_mapping(column, 0);
}

sclass_cmap_t identity_generator_t::select(tensor_size_t f, indices_cmap_t samples, sclass_mem_t& buffer) const
{
    return dataset().visit_inputs(m_feature_mapping(f, 0), [&] (const auto&, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 1)
        {
            auto storage = resize_and_map(buffer, samples.size());
            if (should_drop(f))
            {
                storage.constant(-1);
            }
            else
            {
                for (auto it = !should_shuffle(f) ?
                    make_iterator(data, mask, samples) :
                    make_iterator(data, mask, samples, m_shuffle_indices.find(f)->second); it; ++ it)
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
            }
            return sclass_cmap_t{storage};
        }
        else
        {
            return generator_t::select(f, samples, buffer);
        }
    });
}

mclass_cmap_t identity_generator_t::select(tensor_size_t f, indices_cmap_t samples, mclass_mem_t& buffer) const
{
    return dataset().visit_inputs(m_feature_mapping(f, 0), [&] (const auto& feature, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 2)
        {
            auto storage = resize_and_map(buffer, samples.size(), feature.classes());
            if (should_drop(f))
            {
                storage.constant(-1);
            }
            else
            {
                for (auto it = !should_shuffle(f) ?
                    make_iterator(data, mask, samples) :
                    make_iterator(data, mask, samples, m_shuffle_indices.find(f)->second); it; ++ it)
                {
                    if (const auto [index, sample, given, hits] = *it; given)
                    {
                        storage.vector(index) = hits.array().template cast<int8_t>();
                    }
                    else
                    {
                        storage.vector(index).setConstant(-1);
                    }
                }
            }
            return mclass_cmap_t{storage};
        }
        else
        {
            return generator_t::select(f, samples, buffer);
        }
    });
}

scalar_cmap_t identity_generator_t::select(tensor_size_t f, indices_cmap_t samples, scalar_mem_t& buffer) const
{
    return dataset().visit_inputs(m_feature_mapping(f, 0), [&] (const auto& feature, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 4)
        {
            if (size(feature.dims()) > 1)
            {
                return generator_t::select(f, samples, buffer);
            }
            auto storage = resize_and_map(buffer, samples.size());
            if (should_drop(f))
            {
                storage.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            }
            else
            {
                for (auto it = make_iterator(data, mask, samples); it; ++ it)
                {
                    if (const auto [index, sample, given, values] = *it; given)
                    {
                        storage(index) = static_cast<scalar_t>(values(0));
                    }
                    else
                    {
                        storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                    }
                }
            }
            return scalar_cmap_t{storage};
        }
        else
        {
            return generator_t::select(f, samples, buffer);
        }
    });
}

struct_cmap_t identity_generator_t::select(tensor_size_t f, indices_cmap_t samples, struct_mem_t& buffer) const
{
    return dataset().visit_inputs(m_feature_mapping(f, 0), [&] (const auto& feature, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 4)
        {
            if (size(feature.dims()) <= 1)
            {
                return generator_t::select(f, samples, buffer);
            }

            const auto [dim1, dim2, dim3] = feature.dims();
            auto storage = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
            if (should_drop(f))
            {
                storage.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            }
            else
            {
                for (auto it = !should_shuffle(f) ?
                    make_iterator(data, mask, samples) :
                    make_iterator(data, mask, samples, m_shuffle_indices.find(f)->second); it; ++ it)
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
            }
            return struct_cmap_t{storage};
        }
        else
        {
            return generator_t::select(f, samples, buffer);
        }
    });
}

void identity_generator_t::flatten(indices_cmap_t samples, tensor2d_map_t storage, tensor_size_t column_offset) const
{
    for (tensor_size_t f = 0, column_size = 0, features = this->features(); f < features; ++ f)
    {
        dataset().visit_inputs(m_feature_mapping(f, 0), [&] (const auto& feature, const auto& data, const auto& mask)
        {
            if constexpr (data.rank() == 1)
            {
                column_size = feature.classes();
                for (auto it = !should_shuffle(f) ?
                    make_iterator(data, mask, samples) :
                    make_iterator(data, mask, samples, m_shuffle_indices.find(f)->second); it; ++ it)
                {
                    if (const auto [index, sample, given, label] = *it; given)
                    {
                        auto segment = storage.array(index).segment(column_offset, column_size);
                        if (should_drop(f))
                        {
                            segment.setConstant(+0.0);
                        }
                        else
                        {
                            segment.setConstant(-1.0);
                            segment(label) = +1.0;
                        }
                    }
                    else
                    {
                        auto segment = storage.array(index).segment(column_offset, column_size);
                        segment.setConstant(+0.0);
                    }
                }
            }
            else if constexpr (data.rank() == 2)
            {
                column_size = feature.classes();
                for (auto it = !should_shuffle(f) ?
                    make_iterator(data, mask, samples) :
                    make_iterator(data, mask, samples, m_shuffle_indices.find(f)->second); it; ++ it)
                {
                    if (const auto [index, sample, given, hits] = *it; given)
                    {
                        auto segment = storage.array(index).segment(column_offset, column_size);
                        if (should_drop(f))
                        {
                            segment.setConstant(+0.0);
                        }
                        else
                        {
                            segment.array() = 2.0 * hits.array().template cast<scalar_t>() - 1.0;
                        }
                    }
                    else
                    {
                        auto segment = storage.array(index).segment(column_offset, column_size);
                        segment.setConstant(+0.0);
                    }
                }
            }
            else
            {
                column_size = size(feature.dims());
                for (auto it = !should_shuffle(f) ?
                    make_iterator(data, mask, samples) :
                    make_iterator(data, mask, samples, m_shuffle_indices.find(f)->second); it; ++ it)
                {
                    if (const auto [index, sample, given, values] = *it; given)
                    {
                        auto segment = storage.array(index).segment(column_offset, column_size);
                        if (should_drop(f))
                        {
                            segment.setConstant(+0.0);
                        }
                        else
                        {
                            segment.array() = values.array().template cast<scalar_t>();
                        }
                    }
                    else
                    {
                        auto segment = storage.array(index).segment(column_offset, column_size);
                        segment.setConstant(+0.0);
                    }
                }
            }
            column_offset += column_size;
        });
    }
}

void identity_generator_t::undrop()
{
    m_feature_mapping.matrix().col(1).array() = 0;
}

void identity_generator_t::unshuffle()
{
    m_feature_mapping.matrix().col(1).array() = 0;
}

void identity_generator_t::drop(tensor_size_t feature)
{
    m_feature_mapping(feature, 1) = 1;
}

indices_t identity_generator_t::shuffle(tensor_size_t feature)
{
    m_feature_mapping(feature, 1) = 2;

    indices_t indices = arange(0, dataset().samples());
    std::shuffle(indices.begin(), indices.end(), make_rng());
    m_shuffle_indices[feature] = indices;
    return indices;
}
