#include <nano/core/random.h>
#include <nano/mlearn/kfold.h>

using namespace nano;

kfold_t::kfold_t(indices_t samples, tensor_size_t folds) :
    m_samples(std::move(samples)),
    m_folds(folds)
{
    assert(folds > 1);
    std::shuffle(begin(m_samples), end(m_samples), make_rng());
}

std::pair<indices_t, indices_t> kfold_t::split(tensor_size_t fold) const
{
    assert(fold >= 0 && fold < m_folds);

    const auto chunk = m_samples.size() / m_folds;
    const auto valid_begin = fold * chunk;
    const auto valid_end = (fold + 1 < m_folds) ? (valid_begin + chunk) : m_samples.size();

    indices_t valid(valid_end - valid_begin);
    indices_t train(m_samples.size() - valid.size());

    const auto world = m_samples.vector();

    valid.vector() = world.segment(valid_begin, valid.size());
    train.vector().segment(0, valid_begin) = world.segment(0, valid_begin);
    train.vector().segment(valid_begin, train.size() - valid_begin) = world.segment(valid_end, world.size() - valid_end);

    // NB: sorting samples by index may increase speed!
    std::sort(begin(train), end(train));
    std::sort(begin(valid), end(valid));

    return std::make_pair(train, valid);
}
