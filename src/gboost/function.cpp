#include <nano/gboost/util.h>
#include <nano/gboost/function.h>

using namespace nano;

class cache_t
{
public:

    explicit cache_t(const tensor_size_t tsize = 1) :
        m_gb1(vector_t::Zero(tsize)),
        m_gb2(vector_t::Zero(tsize))
    {
    }

    cache_t& operator+=(const cache_t& other)
    {
        m_vm1 += other.m_vm1;
        m_vm2 += other.m_vm2;
        m_gb1 += other.m_gb1;
        m_gb2 += other.m_gb2;
        return *this;
    }

    cache_t& operator/=(const tensor_size_t samples)
    {
        m_vm1 /= static_cast<scalar_t>(samples);
        m_vm2 /= static_cast<scalar_t>(samples);
        m_gb1 /= static_cast<scalar_t>(samples);
        m_gb2 /= static_cast<scalar_t>(samples);
        return *this;
    }

    void update(const tensor1d_t& values)
    {
        m_vm1 += values.array().sum();
        m_vm2 += values.array().square().sum();
    }

    auto vgrad(scalar_t vAreg, vector_t* gx) const
    {
        if (gx != nullptr)
        {
            *gx = m_gb1 + vAreg * (m_gb2 - m_vm1 * m_gb1) * 2;
        }
        return m_vm1 + vAreg * (m_vm2 - m_vm1 * m_vm1);
    }

    // attributes
    scalar_t    m_vm1{0}, m_vm2{0}; ///< first and second order momentum of the loss values
    vector_t    m_gb1{0}, m_gb2{0}; ///< first and second order momentum of the gradient wrt scale
};

gboost_function_t::gboost_function_t(tensor_size_t dims) :
    function_t("gboost_function", dims, convexity::yes)
{
}

void gboost_function_t::vAreg(scalar_t vAreg)
{
    m_vAreg.set(vAreg);
}

void gboost_function_t::batch(tensor_size_t batch)
{
    m_batch.set(batch);
}

gboost_scale_function_t::gboost_scale_function_t(const loss_t& loss, const dataset_t& dataset, const indices_t& samples,
    const cluster_t& cluster, const tensor4d_t& outputs, const tensor4d_t& woutputs) :
    gboost_function_t(cluster.groups()),
    m_loss(loss),
    m_dataset(dataset),
    m_samples(samples),
    m_cluster(cluster),
    m_outputs(outputs),
    m_woutputs(woutputs)
{
    assert(m_outputs.dims() == m_woutputs.dims());
    assert(m_outputs.dims() == cat_dims(samples.size(), m_dataset.tdims()));
}

scalar_t gboost_scale_function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    assert(!gx || gx->size() == x.size());
    assert(x.size() == m_cluster.groups());

    std::vector<cache_t> caches(tpool_t::size(), cache_t{x.size()});
    loopr(m_samples.size(), batch(), [&] (tensor_size_t begin, tensor_size_t end, size_t tnum)
    {
        assert(tnum < caches.size());
        auto& cache = caches[tnum];

        const auto range = make_range(begin, end);
        const auto targets = m_dataset.targets(m_samples.slice(range));

        // output = output(strong learner) + scale * output(weak learner)
        tensor4d_t outputs(targets.dims());
        for (tensor_size_t i = begin; i < end; ++ i)
        {
            const auto group = m_cluster.group(m_samples(i));
            const auto scale = (group < 0) ? 0.0 : x(group);
            outputs.vector(i - range.begin()) = m_outputs.vector(i) + scale * m_woutputs.vector(i);
        }

        tensor1d_t values;
        m_loss.value(targets, outputs, values);
        cache.update(values);

        if (gx != nullptr)
        {
            tensor4d_t vgrads;
            m_loss.vgrad(targets, outputs, vgrads);

            for (tensor_size_t i = begin; i < end; ++ i)
            {
                const auto group = m_cluster.group(m_samples(i));
                if (group < 0)
                {
                    continue;
                }
                const auto gw = vgrads.vector(i - begin).dot(m_woutputs.vector(i));

                cache.m_gb1(group) += gw;
                cache.m_gb2(group) += gw * values(i - begin);
            }
        }
    });

    // OK
    const auto& cache0 = ::nano::gboost::sum_reduce(caches, m_samples.size());
    return cache0.vgrad(vAreg(), gx);
}

gboost_bias_function_t::gboost_bias_function_t(const loss_t& loss, const dataset_t& dataset, const indices_t& samples) :
    gboost_function_t(::nano::size(dataset.tdims())),
    m_loss(loss),
    m_dataset(dataset),
    m_samples(samples)
{
}

scalar_t gboost_bias_function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto tsize = nano::size(m_dataset.tdims());

    assert(!gx || gx->size() == x.size());
    assert(x.size() == tsize);

    std::vector<cache_t> caches(tpool_t::size(), cache_t{x.size()});
    loopr(m_samples.size(), batch(), [&] (tensor_size_t begin, tensor_size_t end, size_t tnum)
    {
        assert(tnum < caches.size());
        auto& cache = caches[tnum];

        const auto range = make_range(begin, end);
        const auto targets = m_dataset.targets(m_samples.slice(range));

        // output = bias (fixed vector)
        tensor4d_t outputs(targets.dims());
        outputs.reshape(range.size(), -1).matrix().rowwise() = x.transpose();

        tensor1d_t values;
        m_loss.value(targets, outputs, values);
        cache.update(values);

        if (gx != nullptr)
        {
            tensor4d_t vgrads;
            m_loss.vgrad(targets, outputs, vgrads);
            const auto gmatrix = vgrads.reshape(range.size(), tsize).matrix();

            cache.m_gb1 += gmatrix.colwise().sum();
            cache.m_gb2 += gmatrix.transpose() * values.vector();
        }
    });

    // OK
    const auto& cache0 = ::nano::gboost::sum_reduce(caches, m_samples.size());
    return cache0.vgrad(vAreg(), gx);
}

gboost_grads_function_t::gboost_grads_function_t(const loss_t& loss, const dataset_t& dataset, const indices_t& samples) :
    gboost_function_t(samples.size() * nano::size(dataset.tdims())),
    m_loss(loss),
    m_dataset(dataset),
    m_samples(samples),
    m_values(samples.size()),
    m_vgrads(cat_dims(samples.size(), dataset.tdims()))
{
}

scalar_t gboost_grads_function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto odims = cat_dims(m_samples.size(), m_dataset.tdims());

    assert(!gx || gx->size() == x.size());
    assert(x.size() == nano::size(odims));

    const auto& grads = gradients(map_tensor(x.data(), odims));
    if (gx != nullptr)
    {
        *gx = grads.vector();
        *gx /= static_cast<scalar_t>(m_samples.size());
    }

    // OK
    const auto vm1 = m_values.vector().mean();
    const auto vm2 = m_values.array().square().mean();
    return vm1 + vAreg() * (vm2 - vm1 * vm1);
}

const tensor4d_t& gboost_grads_function_t::gradients(const tensor4d_cmap_t& outputs) const
{
    assert(outputs.dims() == m_vgrads.dims());

    loopr(m_samples.size(), batch(), [&] (tensor_size_t begin, tensor_size_t end, size_t)
    {
        const auto range = make_range(begin, end);
        const auto targets = m_dataset.targets(m_samples.slice(range));
        m_loss.value(targets, outputs.slice(range), m_values.slice(range));
        m_loss.vgrad(targets, outputs.slice(range), m_vgrads.slice(range));
    });

    const auto vm1 = m_values.vector().mean();
    loopi(m_values.size(), [&] (tensor_size_t i, size_t)
    {
        m_vgrads.vector(i) *= 1.0 + 2.0 * vAreg() * (m_values(i) - vm1);
    });

    return m_vgrads;
}
