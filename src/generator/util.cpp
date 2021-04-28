#include <nano/generator/util.h>

using namespace nano;

feature_mapping_t nano::select_scalar_components(
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

    // TODO: use a single memory allocation!

    return map_tensor(mapping.data(), static_cast<tensor_size_t>(mapping.size()) / 2, 2);
}
