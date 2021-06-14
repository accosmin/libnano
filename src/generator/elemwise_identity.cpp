#include <nano/generator/select.h>
#include <nano/generator/elemwise_identity.h>

using namespace nano;

sclass_identity_t::sclass_identity_t(const memory_dataset_t& dataset) :
    base_elemwise_generator_t(dataset)
{
}

feature_t sclass_identity_t::feature(tensor_size_t ifeature) const
{
    return dataset().feature(mapped_original(ifeature));
}

feature_mapping_t sclass_identity_t::do_fit(indices_cmap_t, execution)
{
    return select_sclass(dataset(), sclass2binary::off);
}

mclass_identity_t::mclass_identity_t(const memory_dataset_t& dataset) :
    base_elemwise_generator_t(dataset)
{
}

feature_t mclass_identity_t::feature(tensor_size_t ifeature) const
{
    return dataset().feature(mapped_original(ifeature));
}

feature_mapping_t mclass_identity_t::do_fit(indices_cmap_t, execution)
{
    return select_mclass(dataset(), mclass2binary::off);
}

scalar_identity_t::scalar_identity_t(const memory_dataset_t& dataset) :
    base_elemwise_generator_t(dataset)
{
}

feature_t scalar_identity_t::feature(tensor_size_t ifeature) const
{
    return dataset().feature(mapped_original(ifeature));
}

feature_mapping_t scalar_identity_t::do_fit(indices_cmap_t, execution)
{
    return select_scalar(dataset(), struct2scalar::off);
}

struct_identity_t::struct_identity_t(const memory_dataset_t& dataset) :
    base_elemwise_generator_t(dataset)
{
}

feature_t struct_identity_t::feature(tensor_size_t ifeature) const
{
    return dataset().feature(mapped_original(ifeature));
}

feature_mapping_t struct_identity_t::do_fit(indices_cmap_t, execution)
{
    return select_struct(dataset());
}
