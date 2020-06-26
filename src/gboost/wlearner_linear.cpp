#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/gboost/wlearner_linear.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        void clear(const tensor3d_dim_t& tdim)
        {
            m_x1 = 0;
            m_x2 = 0;
            m_cnt = 0;

            m_r1.resize(tdim);
            m_r2.resize(tdim);
            m_rx.resize(tdim);
            m_tables.resize(cat_dims(2, tdim));

            m_r1.zero();
            m_r2.zero();
            m_rx.zero();
        }

        [[nodiscard]] auto a() const
        {
            return (m_rx.array() * m_cnt - m_x1 * m_r1.array()) / (m_x2 * m_cnt - m_x1 * m_x1);
        }

        [[nodiscard]] auto b() const
        {
            return (m_r1.array() * m_x2 - m_x1 * m_rx.array()) / (m_x2 * m_cnt - m_x1 * m_x1);
        }

        [[nodiscard]] auto score() const
        {
            scalar_t score = 0;
            if (m_cnt > 0)
            {
                score += (a().square() * m_x2 + b().square() * m_cnt + m_r2.array()
                    + 2 * a().array() * b().array() * m_x1
                    - 2 * b().array() * m_r1.array()
                    - 2 * a().array() * m_rx.array()).sum();
            }
            return score;
        }

        // attributes
        tensor_size_t   m_feature{0};                                   ///<
        tensor4d_t      m_tables;                                       ///<
        scalar_t        m_x1{0}, m_x2{0}, m_cnt{0};                     ///<
        tensor3d_t      m_r1, m_r2, m_rx;                               ///<
        scalar_t        m_score{std::numeric_limits<scalar_t>::max()};  ///<
    };
}

wlearner_linear_t::wlearner_linear_t() = default;

rwlearner_t wlearner_linear_t::clone() const
{
    return std::make_unique<wlearner_linear_t>(*this);
}

scalar_t wlearner_linear_t::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients,
    const indices_t& indices)
{
    assert(indices.min() >= 0);
    assert(indices.max() < dataset.samples(fold));
    assert(gradients.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    switch (type())
    {
    case wlearner::real:
        break;

    default:
        critical(true, "linear weak learner: unhandled wlearner");
        break;
    }

    std::vector<cache_t> caches(tpool_t::size());
    loopi(dataset.features(), [&] (tensor_size_t feature, size_t tnum)
    {
        const auto& ifeature = dataset.ifeature(feature);

        // NB: This weak learner works only with continuous features!
        if (ifeature.discrete())
        {
            return;
        }
        const auto fvalues = dataset.inputs(fold, make_range(0, dataset.samples(fold)), feature);

        // update accumulators
        auto& cache = caches[tnum];
        cache.clear(dataset.tdim());
        for (const auto i : indices)
        {
            const auto value = fvalues(i);
            if (feature_t::missing(value))
            {
                continue;
            }

            ++ cache.m_cnt;
            cache.m_x1 += value;
            cache.m_x2 += value * value;
            cache.m_r1.array() -= gradients.array(i);
            cache.m_rx.array() -= value * gradients.array(i);
            cache.m_r2.array() += gradients.array(i) * gradients.array(i);
        }

        // update the parameters if a better feature
        const auto score = cache.score();
        if (score < cache.m_score)
        {
            cache.m_tables.zero();
            cache.m_score = score;
            cache.m_feature = feature;
            if (cache.m_cnt > 0)
            {
                cache.m_tables.array(0) = cache.a();
                cache.m_tables.array(1) = cache.b();
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);
    set(best.m_feature, best.m_tables);
    return best.m_score;
}

void wlearner_linear_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
{
    wlearner_feature1_t::predict(dataset, fold, range, outputs, [&] (scalar_t x, tensor_size_t i)
    {
        outputs.vector(i) = vector(0) * x + vector(1);
    });
}

cluster_t wlearner_linear_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    cluster_t cluster(dataset.samples(fold), 1);
    wlearner_feature1_t::split(dataset, fold, indices, [&] (scalar_t, tensor_size_t i)
    {
        cluster.assign(i, 0);
    });

    return cluster;
}
