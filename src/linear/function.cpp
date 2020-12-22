#include <nano/linear/util.h>
#include <nano/linear/cache.h>
#include <nano/linear/function.h>

using namespace nano;

linear_function_t::linear_function_t(const loss_t& loss, const dataset_t& dataset, const indices_t& samples) :
    function_t("linear", (::nano::size(dataset.idims()) + 1) * ::nano::size(dataset.tdims()), convexity::yes),
    m_loss(loss),
    m_dataset(dataset),
    m_samples(samples),
    m_isize(::nano::size(dataset.idims())),
    m_tsize(::nano::size(dataset.tdims())),
    m_istats(m_dataset.istats(m_samples, batch()))
{
    assert(m_isize > 0);
    assert(m_tsize > 0);
}

scalar_t linear_function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    assert(!gx || gx->size() == x.size());
    assert(x.size() == (m_isize + 1) * m_tsize);

    const auto b = bias(x);
    const auto W = weights(x);

    std::vector<linear_cache_t> caches(tpool_t::size(), linear_cache_t{m_isize, m_tsize, gx != nullptr, vAreg() > 0});

    loopr(m_samples.size(), batch(), [&] (tensor_size_t begin, tensor_size_t end, size_t tnum)
    {
        assert(tnum < caches.size());
        auto& cache = caches[tnum];

        const auto range = make_range(begin, end);
        auto inputs = m_dataset.inputs(m_samples.slice(range));
        const auto targets = m_dataset.targets(m_samples.slice(range));

        m_istats.scale(normalization(), inputs);
        ::nano::linear::predict(inputs, W, b, cache.m_outputs);
        m_loss.value(targets, cache.m_outputs, cache.m_values);

        const auto vvector = cache.m_values.vector();

        cache.m_vm1 += vvector.array().sum();
        if (vAreg() > 0)
        {
            cache.m_vm2 += vvector.array().square().sum();
        }

        if (gx != nullptr)
        {
            m_loss.vgrad(targets, cache.m_outputs, cache.m_vgrads);

            const auto imatrix = inputs.reshape(range.size(), W.rows()).matrix();
            const auto gmatrix = cache.m_vgrads.reshape(range.size(), W.cols()).matrix();

            cache.m_gb1.vector() += gmatrix.colwise().sum();
            cache.m_gW1.matrix() += imatrix.transpose() * gmatrix;

            if (vAreg() > 0)
            {
                cache.m_gb2.vector() += gmatrix.transpose() * vvector;
                cache.m_gW2.matrix() += (imatrix.array().colwise() * vvector.array()).matrix().transpose() * gmatrix;
            }
        }
    });

    const auto& cache0 = linear_cache_t::reduce(caches, m_samples.size());

    // OK, normalize and add the regularization terms
    if (gx != nullptr)
    {
        auto gb = bias(*gx);
        auto gW = weights(*gx);

        gb = cache0.m_gb1;
        gW = cache0.m_gW1;

        if (l1reg() > 0)
        {
            gW.array() += l1reg() * W.array().sign() / W.size();
        }
        if (l2reg() > 0)
        {
            gW.array() += l2reg() * W.array() * 2 / W.size();
        }
        if (vAreg() > 0)
        {
            gb.array() += vAreg() * (cache0.m_gb2.array() - cache0.m_vm1 * cache0.m_gb1.array()) * 2;
            gW.array() += vAreg() * (cache0.m_gW2.array() - cache0.m_vm1 * cache0.m_gW1.array()) * 2;
        }
    }

    return  cache0.m_vm1 +
            ((l1reg() > 0) ? (l1reg() * W.array().abs().mean()) : scalar_t(0)) +
            ((l2reg() > 0) ? (l2reg() * W.array().square().mean()) : scalar_t(0)) +
            ((vAreg() > 0) ? (vAreg() * (cache0.m_vm2 - cache0.m_vm1 * cache0.m_vm1)) : scalar_t(0));
}
