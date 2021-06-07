#include <nano/generator/elemwise.h>

using namespace nano;

feature_t base_elemwise_generator_t::make_scalar_feature(tensor_size_t ifeature, const char* name) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original = mapped_original(ifeature);
    const auto component = mapped_component(ifeature);

    const auto& feature = dataset().feature(original);
    return  feature_t{scat(name, "(", feature.name(), "[", component, "])")}.
            scalar(feature_type::float64);
}

feature_t base_elemwise_generator_t::make_sclass_feature(tensor_size_t ifeature, const char* name, strings_t labels) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original = mapped_original(ifeature);
    const auto component = mapped_component(ifeature);

    const auto& feature = dataset().feature(original);
    return  feature_t{scat(name, "(", feature.name(), "[", component, "])")}.
            sclass(std::move(labels));
}
