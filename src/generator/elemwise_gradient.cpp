#include <nano/generator/select.h>
#include <nano/generator/elemwise_gradient.h>

using namespace nano;

elemwise_gradient_t::elemwise_gradient_t(
    const memory_dataset_t& dataset, kernel3x3_type type, const indices_t& original_features) :
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
            const auto channels = mapping(i, 5);
            count += channels * 4; // NB: input channels * gradient features!
        }
    }

    auto feature_mapping = feature_mapping_t{count, 8};
    for (tensor_size_t i = 0, k = 0; i < mapping.size<0>(); ++ i)
    {
        if (mapping(i, 3) >= 3 && mapping(i, 4) >= 3)
        {
            for (tensor_size_t channel = 0, channels = mapping(i, 5); channel < channels; ++ channel)
            {
                for (tensor_size_t type = 0; type < 4; ++ type)
                {
                    feature_mapping.vector(k).segment(0, 6) = mapping.vector(i).segment(0, 6);
                    feature_mapping(k, 3) -= 2; // rows after filtering with a 3x3 kernel
                    feature_mapping(k, 4) -= 2; // columns after filtering with a 3x3 kernel
                    feature_mapping(k, 5) = 1;  // one channel filtered at a time
                    feature_mapping(k, 6) = channel;
                    feature_mapping(k ++, 7) = type;
                }
            }
        }
    }

    return feature_mapping;
}

feature_t elemwise_gradient_t::feature(tensor_size_t ifeature) const
{
    const auto original = mapped_original(ifeature);
    const auto dims = mapped_dims(ifeature);

    const auto& feature = dataset().feature(original);

    auto suffix = scat(m_type);
    switch (mapped_mode(ifeature))
    {
    case gradient3x3_mode::gradx:       suffix += "::gx"; break;
    case gradient3x3_mode::grady:       suffix += "::gy"; break;
    case gradient3x3_mode::magnitude:   suffix += "::gg"; break;
    default:                            suffix += "::theta"; break;
    }

    const auto channel = mapped_channel(ifeature);

    return  feature_t{scat(suffix, "(", feature.name(), "[channel::", channel, "])")}.
            scalar(feature_type::float64, dims);
}
