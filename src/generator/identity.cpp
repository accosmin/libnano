#include <nano/generator/util.h>
#include <nano/generator/identity.h>

using namespace nano;

identity_generator_t::identity_generator_t(const memory_dataset_t& dataset) :
    generator_t(dataset)
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

void identity_generator_t::select(indices_cmap_t samples, tensor_size_t ifeature, sclass_map_t storage) const
{
    iterate1(samples, ifeature, ifeature, [&] (const auto&, const auto& data, const auto& mask, indices_cmap_t samples)
    {
        loop_samples<1U>(data, mask, samples, [&] (auto it)
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
        });
    });
}

void identity_generator_t::select(indices_cmap_t samples, tensor_size_t ifeature, mclass_map_t storage) const
{
    iterate1(samples, ifeature, ifeature, [&] (const auto&, const auto& data, const auto& mask, indices_cmap_t samples)
    {
        loop_samples<2U>(data, mask, samples, [&] (auto it)
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
        });
    });
}

void identity_generator_t::select(indices_cmap_t samples, tensor_size_t ifeature, scalar_map_t storage) const
{
    iterate1(samples, ifeature, ifeature, [&] (const auto& feature, const auto& data, const auto& mask, indices_cmap_t samples)
    {
        loop_samples<4U>(data, mask, samples, [&] (auto it)
        {
            if (size(feature.dims()) > 1)
            {
                generator_t::select(samples, ifeature, storage);
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
        });
    });
}

void identity_generator_t::select(indices_cmap_t samples, tensor_size_t ifeature, struct_map_t storage) const
{
    iterate1(samples, ifeature, ifeature, [&] (const auto& feature, const auto& data, const auto& mask, indices_cmap_t samples)
    {
        loop_samples<4U>(data, mask, samples, [&] (auto it)
        {
            if (size(feature.dims()) <= 1)
            {
                generator_t::select(samples, ifeature, storage);
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
        });
    });
}

void identity_generator_t::flatten(indices_cmap_t samples, tensor2d_map_t storage, tensor_size_t column) const
{
    for (tensor_size_t ifeature = 0, colsize = 0, features = this->features();
         ifeature < features; ++ ifeature, column += colsize)
    {
        iterate1(samples, ifeature, ifeature, [&] (const auto& feature, const auto& data, const auto& mask, indices_cmap_t samples)
        {
            loop_samples(data, mask, samples,
            [&] (auto it)
            {
                colsize = feature.classes();
                for (; it; ++ it)
                {
                    if (const auto [index, given, label] = *it; given)
                    {
                        auto segment = storage.array(index).segment(column, colsize);
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
                        auto segment = storage.array(index).segment(column, colsize);
                        segment.setConstant(+0.0);
                    }
                }
            },
            [&] (auto it)
            {
                colsize = feature.classes();
                for (; it; ++ it)
                {
                    if (const auto [index, given, hits] = *it; given)
                    {
                        auto segment = storage.array(index).segment(column, colsize);
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
                        auto segment = storage.array(index).segment(column, colsize);
                        segment.setConstant(+0.0);
                    }
                }
            },
            [&] (auto it)
            {
                colsize = size(feature.dims());
                for (; it; ++ it)
                {
                    if (const auto [index, given, values] = *it; given)
                    {
                        auto segment = storage.array(index).segment(column, colsize);
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
                        auto segment = storage.array(index).segment(column, colsize);
                        segment.setConstant(+0.0);
                    }
                }
            });
        });
    }
}
