#include <nano/dataset/generators.h>

using namespace nano;

identity_generator_t::identity_generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
    generator_t(dataset, samples)
{
    allocate(this->features());
}

tensor_size_t identity_generator_t::features() const
{
    return dataset().features();
}

feature_t identity_generator_t::feature(tensor_size_t feature) const
{
    return dataset().feature(feature);
}

void identity_generator_t::select(tensor_size_t feature, tensor_range_t sample_range, sclass_map_t storage) const
{
    return dataset().visit_inputs(feature, [&] (const auto&, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 1)
        {
            if (should_drop(feature))
            {
                storage.constant(-1);
            }
            else
            {
                for (auto it = make_iterator(data, mask, samples(feature, sample_range)); it; ++ it)
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
        }
        else
        {
            generator_t::select(feature, sample_range, storage);
        }
    });
}

void identity_generator_t::select(tensor_size_t feature, tensor_range_t sample_range, mclass_map_t storage) const
{
    return dataset().visit_inputs(feature, [&] (const auto&, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 2)
        {
            if (should_drop(feature))
            {
                storage.constant(-1);
            }
            else
            {
                for (auto it = make_iterator(data, mask, samples(feature, sample_range)); it; ++ it)
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
        }
        else
        {
            generator_t::select(feature, sample_range, storage);
        }
    });
}

void identity_generator_t::select(tensor_size_t feature, tensor_range_t sample_range, scalar_map_t storage) const
{
    return dataset().visit_inputs(feature, [&] (const auto& feature_, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 4)
        {
            if (size(feature_.dims()) > 1)
            {
                generator_t::select(feature, sample_range, storage);
            }
            else if (should_drop(feature))
            {
                storage.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            }
            else
            {
                for (auto it = make_iterator(data, mask, samples(feature, sample_range)); it; ++ it)
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
        }
        else
        {
            generator_t::select(feature, sample_range, storage);
        }
    });
}

void identity_generator_t::select(tensor_size_t feature, tensor_range_t sample_range, struct_map_t storage) const
{
    return dataset().visit_inputs(feature, [&] (const auto& feature_, const auto& data, const auto& mask)
    {
        if constexpr (data.rank() == 4)
        {
            if (size(feature_.dims()) <= 1)
            {
                generator_t::select(feature, sample_range, storage);
            }
            else if (should_drop(feature))
            {
                storage.constant(std::numeric_limits<scalar_t>::quiet_NaN());
            }
            else
            {
                for (auto it = make_iterator(data, mask, samples(feature, sample_range)); it; ++ it)
                {
                    if (const auto [index, given, values] = *it; given)
                    {
                        storage.array(index) = values.array().template cast<scalar_t>();
                    }
                    else
                    {
                        storage.tensor(index).constant(std::numeric_limits<scalar_t>::quiet_NaN());
                    }
                }
            }
        }
        else
        {
            generator_t::select(feature, sample_range, storage);
        }
    });
}

void identity_generator_t::flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column_offset) const
{
    for (tensor_size_t ifeature = 0, column_size = 0, features = this->features(); ifeature < features; ++ ifeature)
    {
        dataset().visit_inputs(ifeature, [&] (const auto& feature, const auto& data, const auto& mask)
        {
            if constexpr (data.rank() == 1)
            {
                column_size = feature.classes();
                for (auto it = make_iterator(data, mask, samples(ifeature, sample_range)); it; ++ it)
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
                for (auto it = make_iterator(data, mask, samples(ifeature, sample_range)); it; ++ it)
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
            }
            else
            {
                column_size = size(feature.dims());
                for (auto it = make_iterator(data, mask, samples(ifeature, sample_range)); it; ++ it)
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
            }
            column_offset += column_size;
        });
    }
}
