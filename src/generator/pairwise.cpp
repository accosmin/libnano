#include <nano/generator/pairwise.h>

using namespace nano;

feature_mapping_t base_pairwise_generator_t::make_pairwise(const feature_mapping_t& mapping)
{
    const auto size = mapping.template size<0>();
    auto feature_mapping = feature_mapping_t{size * (size + 1) / 2, 12};

    for (tensor_size_t k = 0, i = 0; i < size; ++ i)
    {
        for (tensor_size_t j = i; j < size; ++ j, ++ k)
        {
            feature_mapping.array(k).segment(0, 6) = mapping.array(i);
            feature_mapping.array(k).segment(6, 6) = mapping.array(j);
        }
    }

    return feature_mapping;
}

feature_t base_pairwise_generator_t::make_scalar_feature(tensor_size_t ifeature, const char* name) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original1 = mapped_original1(ifeature);
    const auto original2 = mapped_original2(ifeature);
    const auto component1 = mapped_component1(ifeature);
    const auto component2 = mapped_component2(ifeature);

    const auto& feature1 = dataset().feature(original1);
    const auto& feature2 = dataset().feature(original2);

    return  feature_t{scat(name, "(", feature1.name(), "[", component1, "],", feature2.name(), "[", component2, "])")}.
            scalar(feature_type::float64);
}

feature_t base_pairwise_generator_t::make_sclass_feature(tensor_size_t ifeature, const char* name, strings_t labels) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original1 = mapped_original1(ifeature);
    const auto original2 = mapped_original2(ifeature);
    const auto component1 = mapped_component1(ifeature);
    const auto component2 = mapped_component2(ifeature);

    const auto& feature1 = dataset().feature(original1);
    const auto& feature2 = dataset().feature(original2);

    return  feature_t{scat(name, "(", feature1.name(), "[", component1, "],", feature2.name(), "[", component2, "])")}.
            sclass(std::move(labels));
}
