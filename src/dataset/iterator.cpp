#include <nano/dataset/iterator.h>

using namespace nano;

dataset_iterator_t::dataset_iterator_t(indices_t samples) :
    m_samples(std::move(samples))
{
}

const indices_t& dataset_iterator_t::samples() const
{
    return m_samples;
}
