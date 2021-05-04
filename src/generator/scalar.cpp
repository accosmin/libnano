#include <nano/generator/util.h>
#include <nano/generator/scalar.h>

using namespace nano;

scalar_pairwise_generator_t::scalar_pairwise_generator_t(
    const memory_dataset_t& dataset, const indices_t& samples, struct2scalar s2s, const indices_t& feature_indices) :
    pairwise_generator_t(dataset, samples, select_scalar_components(dataset, s2s, feature_indices))
{
}
