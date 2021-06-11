#include <nano/generator/elemwise_gradient.h>

using namespace nano;

elemwise_gradient_t::elemwise_gradient_t(
    const memory_dataset_t& dataset, gradient_type type, const indices_t& original_features) :
    base_elemwise_generator_t(dataset),
    m_type(type),
    m_original_features(original_features)
{
}

feature_mapping_t elemwise_gradient_t::do_fit(indices_cmap_t, execution)
{
    const auto mapping = select_struct(dataset(), m_original_features);

    tensor_size_t count = 0;
    for (tensor_size_t i = 0; i < mapping.size<0>(); ++ i)
    {
        if (mapping(i, 3) >= 3 && mapping(i, 4) >= 3)
        {
            ++ count;
        }
    }

    auto feature_mapping = feature_mapping_t{count * 4, 7};
    for (tensor_size_t i = 0, k = 0; i < mapping.size<0>(); ++ i)
    {
        if (mapping(i, 3) >= 3 && mapping(i, 4) >= 3)
        {
            for (tensor_size_t f = 0; f < 4; ++ f)
            {
                feature_mapping(k, 0) = mapping(i, 0);
                feature_mapping(k, 1) = mapping(i, 1);
                feature_mapping(k, 2) = mapping(i, 2);
                feature_mapping(k, 3) = mapping(i, 3);
                feature_mapping(k, 4) = mapping(i, 4);
                feature_mapping(k, 5) = mapping(i, 5);
                feature_mapping(k ++, 6) = f;
            }
        }
    }

    return feature_mapping;
}

feature_t elemwise_gradient_t::feature(tensor_size_t ifeature) const
{
    const auto original = mapped_original(ifeature);
    const auto original_dims = mapped_dims(ifeature);

    const auto dims = make_dims(
        std::get<0>(original_dims) - 2, // rows after filtering with a 3x3 kernel
        std::get<1>(original_dims) - 2, // columns after filtering with a 3x3 kernel
        std::get<2>(original_dims));    // #channels

    const auto& feature = dataset().feature(original);

    auto suffix = scat(m_type);
    switch (mapped_feature_type(ifeature))
    {
    case 0:     suffix += "::gx"; break;
    case 1:     suffix += "::gy"; break;
    case 2:     suffix += "::gg"; break;
    default:    suffix += "::theta"; break;
    }

    return  feature_t{scat(suffix, "(", feature.name(), ")")}.
            scalar(feature_type::float64, dims);
}
