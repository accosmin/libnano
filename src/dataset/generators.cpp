#include <nano/dataset/generators.h>

using namespace nano;

identity_generator_t::identity_generator_t(
    const memory_dataset_t& dataset, const indices_t& samples,
    mclass2binary m2b, sclass2binary s2b, struct2scalar s2s) :
    generator_t(dataset, samples),
    m_mclass2binary(m2b),
    m_sclass2binary(s2b),
    m_struct2scalar(s2s)
{
    tensor_size_t count = 0;
    for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
    {
        switch (const auto& feature = dataset.feature(ifeature); feature.type())
        {
        case feature_type::mclass:
            count += (m_mclass2binary == mclass2binary::on) ? (1 + feature.classes()) : tensor_size_t(1);
            break;

        case feature_type::sclass:
            count += (m_sclass2binary == sclass2binary::on) ? (1 + feature.classes()) : tensor_size_t(1);
            break;

        default:
            count += 1;
            if (size(feature.dims()) > 1 && m_struct2scalar == struct2scalar::on)
            {
                count += size(feature.dims());
            }
            break;
        }
    }

    m_feature_mapping.resize(count, 2);
    for (tensor_size_t ifeature = 0, index = 0, features = dataset.features(); ifeature < features; ++ ifeature)
    {
        switch (const auto& feature = dataset.feature(ifeature); feature.type())
        {
        case feature_type::mclass:
            map1(index, ifeature, -1);
            if (m_mclass2binary == mclass2binary::on)
            {
                mapN(index, ifeature, feature.classes());
            }
            break;

        case feature_type::sclass:
            map1(index, ifeature, -1);
            if (m_sclass2binary == sclass2binary::on)
            {
                mapN(index, ifeature, feature.classes());
            }
            break;

        default:
            map1(index, ifeature, -1);
            if (size(feature.dims()) > 1 && m_struct2scalar == struct2scalar::on)
            {
                mapN(index, ifeature, size(feature.dims()));
            }
            break;
        }
    }

    allocate(this->features());
}

tensor_size_t identity_generator_t::features() const
{
    return m_feature_mapping.size<0>();
}

feature_t identity_generator_t::feature(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < m_feature_mapping.size<0>());

    const auto component = m_feature_mapping(ifeature, 1);

    switch (const auto& feature = dataset().feature(m_feature_mapping(ifeature, 0)); feature.type())
    {
    case feature_type::sclass:
        if (component < 0)
        {
            return feature;
        }
        else
        {
            return feature_t{
                scat(feature.name(), "_", feature.labels()[static_cast<size_t>(component)])}
                .sclass(strings_t{"off", "on"});
        }

    case feature_type::mclass:
        if (component < 0)
        {
            return feature;
        }
        else
        {
            return feature_t{
                scat(feature.name(), "_", feature.labels()[static_cast<size_t>(component)])}
                .sclass(strings_t{"off", "on"});
        }

    default:
        if (component < 0)
        {
            return feature;
        }
        else
        {
            return feature_t{scat(feature.name(), "_", component)}.scalar(feature.type(), make_dims(1, 1, 1));
        }
    }
}

void identity_generator_t::select(tensor_size_t ifeature, tensor_range_t sample_range, sclass_map_t storage) const
{
    dataset().visit_inputs(ifeature, [&] (const auto&, const auto& data, const auto& mask)
    {
        loop_samples<1U>(data, mask, samples(ifeature, sample_range),
            [&] (auto it)
            {
                if (should_drop(ifeature))
                {
                    storage.full(-1);
                }
                else
                {
                    for (; it; ++ it)
                    {
                        if (const auto [index, given, label] = *it; given)
                        {
                            storage(index) = static_cast<int32_t>(label);
                        }
                        else
                        {
                            storage(index) = -1;
                        }
                    }
                }
            },
            [&] ()
            {
                generator_t::select(ifeature, sample_range, storage);
            });
    });
}

void identity_generator_t::select(tensor_size_t ifeature, tensor_range_t sample_range, mclass_map_t storage) const
{
    dataset().visit_inputs(ifeature, [&] (const auto&, const auto& data, const auto& mask)
    {
        loop_samples<2U>(data, mask, samples(ifeature, sample_range),
            [&] (auto it)
            {
                if (should_drop(ifeature))
                {
                    storage.full(-1);
                }
                else
                {
                    for (; it; ++ it)
                    {
                        if (const auto [index, given, hits] = *it; given)
                        {
                            storage.vector(index) = hits.array().template cast<int8_t>();
                        }
                        else
                        {
                            storage.vector(index).setConstant(-1);
                        }
                    }
                }
            },
            [&] ()
            {
                generator_t::select(ifeature, sample_range, storage);
            });
    });
}

void identity_generator_t::select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const
{
    dataset().visit_inputs(ifeature, [&] (const auto& feature, const auto& data, const auto& mask)
    {
        loop_samples<4U>(data, mask, samples(ifeature, sample_range),
            [&] (auto it)
            {
                if (size(feature.dims()) > 1)
                {
                    generator_t::select(ifeature, sample_range, storage);
                }
                else if (should_drop(ifeature))
                {
                    storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
                }
                else
                {
                    for (; it; ++ it)
                    {
                        if (const auto [index, given, values] = *it; given)
                        {
                            storage(index) = static_cast<scalar_t>(values(0));
                        }
                        else
                        {
                            storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                        }
                    }
                }
            },
            [&] ()
            {
                generator_t::select(ifeature, sample_range, storage);
            });
    });
}

void identity_generator_t::select(tensor_size_t ifeature, tensor_range_t sample_range, struct_map_t storage) const
{
    dataset().visit_inputs(ifeature, [&] (const auto& feature, const auto& data, const auto& mask)
    {
        loop_samples<4U>(data, mask, samples(ifeature, sample_range),
            [&] (auto it)
            {
                if (size(feature.dims()) <= 1)
                {
                    generator_t::select(ifeature, sample_range, storage);
                }
                else if (should_drop(ifeature))
                {
                    storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
                }
                else
                {
                    for (; it; ++ it)
                    {
                        if (const auto [index, given, values] = *it; given)
                        {
                            storage.array(index) = values.array().template cast<scalar_t>();
                        }
                        else
                        {
                            storage.tensor(index).full(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                    }
                }
            },
            [&]
            {
                generator_t::select(ifeature, sample_range, storage);
            });
    });
}

void identity_generator_t::flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column_offset) const
{
    for (tensor_size_t ifeature = 0, column_size = 0, features = this->features();
         ifeature < features; ++ ifeature, column_offset += column_size)
    {
        dataset().visit_inputs(ifeature, [&] (const auto& feature, const auto& data, const auto& mask)
        {
            loop_samples(data, mask, samples(ifeature, sample_range),
                [&] (auto it)
                {
                    column_size = feature.classes();
                    for (; it; ++ it)
                    {
                        if (const auto [index, given, label] = *it; given)
                        {
                            auto segment = storage.array(index).segment(column_offset, column_size);
                            if (should_drop(ifeature))
                            {
                                segment.setConstant(+0.0);
                            }
                            else
                            {
                                segment.setConstant(-1.0);
                                segment(static_cast<tensor_size_t>(label)) = +1.0;
                            }
                        }
                        else
                        {
                            auto segment = storage.array(index).segment(column_offset, column_size);
                            segment.setConstant(+0.0);
                        }
                    }
                },
                [&] (auto it)
                {
                    column_size = feature.classes();
                    for (; it; ++ it)
                    {
                        if (const auto [index, given, hits] = *it; given)
                        {
                            auto segment = storage.array(index).segment(column_offset, column_size);
                            if (should_drop(ifeature))
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
                },
                [&] (auto it)
                {
                    column_size = size(feature.dims());
                    for (; it; ++ it)
                    {
                        if (const auto [index, given, values] = *it; given)
                        {
                            auto segment = storage.array(index).segment(column_offset, column_size);
                            if (should_drop(ifeature))
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
                });
        });
    }
}
