#include <nano/dataset/generators.h>

using namespace nano;

std::vector<tensor_size_t> nano::select_scalar_components(
    const memory_dataset_t& dataset, struct2scalar s2s, const indices_t& feature_indices)
{
    std::vector<tensor_size_t> mapping;

    const auto check = [&] (tensor_size_t ifeature)
    {
        const auto& feature = dataset.feature(ifeature);
        if (feature.type() != feature_type::mclass &&
            feature.type() != feature_type::sclass)
        {
            const auto components = size(feature.dims());

            if (components == 1)
            {
                mapping.push_back(ifeature);
                mapping.push_back(0);
            }
            else if (s2s == struct2scalar::on)
            {
                for (tensor_size_t icomponent = 0; icomponent < components; ++ icomponent)
                {
                    mapping.push_back(ifeature);
                    mapping.push_back(icomponent);
                }
            }
        }
    };

    if (feature_indices.size() > 0U)
    {
        for (const auto ifeature : feature_indices)
        {
            check(ifeature);
        }
    }
    else
    {
        for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
        {
            check(ifeature);
        }
    }

    return mapping;
}

identity_generator_t::identity_generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
    generator_t(dataset, samples)
{
    allocate(this->features());
}

tensor_size_t identity_generator_t::features() const
{
    return dataset().features();
}

feature_t identity_generator_t::feature(tensor_size_t ifeature) const
{
    return dataset().feature(ifeature);
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
            return false;
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
