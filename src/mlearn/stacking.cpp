#include <nano/mlearn/stacking.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(tensor_size_t models) :
            m_gx(vector_t::Zero(models))
        {
        }

        // attributes
        scalar_t    m_fx{0};    ///<
        vector_t    m_gx{0};    ///<
        tensor1d_t  m_values;   ///<
        tensor4d_t  m_vgrads;   ///<
        tensor4d_t  m_outputs;  ///<
    };
}

stacking_function_t::stacking_function_t(
    const loss_t& loss, const tensor4d_t& targets, const tensor5d_t& outputs) :
    function_t("stacking", outputs.size<0>(), convexity::no),
    m_loss(loss),
    m_targets(targets),
    m_outputs(outputs)
{
}

void stacking_function_t::batch(tensor_size_t batch)
{
    m_batch.set(batch);
}

vector_t stacking_function_t::as_weights(const vector_t& x)
{
    vector_t expx = x.array().exp();
    const scalar_t norm = expx.sum();
    expx /= norm;
    return expx;
}

scalar_t stacking_function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto models = m_outputs.size<0>();
    const auto samples = m_outputs.size<1>();

    assert(x.size() == models);
    assert(!gx || gx->size() == x.size());
    assert(m_targets.size<0>() == m_outputs.size<1>());
    assert(m_targets.size<1>() == m_outputs.size<2>());
    assert(m_targets.size<2>() == m_outputs.size<3>());
    assert(m_targets.size<3>() == m_outputs.size<4>());

    const auto weights = stacking_function_t::as_weights(x);

    matrix_t gweights = -weights * weights.transpose();
    gweights.diagonal() += weights;

    std::vector<cache_t> caches(tpool_t::size(), cache_t{models});
    loopr(samples, static_cast<tensor_size_t>(batch()), [&] (tensor_size_t begin, tensor_size_t end, size_t tnum)
    {
        auto& cache = caches[tnum];
        auto& values = cache.m_values;
        auto& vgrads = cache.m_vgrads;
        auto& outputs = cache.m_outputs;

        const auto range = make_range(begin, end);

        outputs.resize(make_dims(range.size(), m_outputs.size<2>(), m_outputs.size<3>(), m_outputs.size<4>()));
        outputs.zero();
        for (tensor_size_t model = 0; model < models; ++ model)
        {
            outputs.vector() += weights(model) * m_outputs.tensor(model).slice(range).vector();
        }

        m_loss.value(m_targets.slice(range), outputs.tensor(), values);
        cache.m_fx += values.vector().sum();

        if (gx != nullptr)
        {
            m_loss.vgrad(m_targets.slice(range), outputs.tensor(), vgrads);

            const auto gmatrix = vgrads.reshape(range.size(), -1).matrix();

            for (tensor_size_t model = 0; model < models; ++ model)
            {
                const auto omatrix = m_outputs.tensor(model).slice(range).reshape(range.size(), - 1).matrix();

                cache.m_gx += gweights.row(model) * (gmatrix.array() * omatrix.array()).colwise().sum().sum();
            }
        }
    });

    for (size_t i = 1; i < caches.size(); ++ i)
    {
        caches[0].m_fx += caches[i].m_fx;
        caches[0].m_gx += caches[i].m_gx;
    }

    if (gx != nullptr)
    {
        *gx = caches[0].m_gx / samples;
    }
    return caches[0].m_fx / static_cast<scalar_t>(samples);
}
