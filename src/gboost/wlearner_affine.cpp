#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/gboost/wlearner_affine.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        cache_t(const tensor3d_dim_t& tdim) :
            m_r1(tdim),
            m_r2(tdim),
            m_rx(tdim),
            m_tables(cat_dims(2, tdim))
        {
        }

        void clear()
        {
            m_x1 = 0;
            m_x2 = 0;
            m_cnt = 0;

            m_r1.zero();
            m_r2.zero();
            m_rx.zero();
        }

        template <typename tgradient>
        void update(scalar_t value, tgradient&& gradient)
        {
            ++ m_cnt;
            m_x1 += value;
            m_x2 += value * value;
            m_r1.array() -= gradient;
            m_rx.array() -= value * gradient;
            m_r2.array() += gradient * gradient;
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
        scalar_t        m_x1{0}, m_x2{0}, m_cnt{0};                     ///<
        tensor3d_t      m_r1, m_r2, m_rx;                               ///<
        tensor_size_t   m_feature{0};                                   ///<
        tensor4d_t      m_tables;                                       ///<
        scalar_t        m_score{std::numeric_limits<scalar_t>::max()};  ///<
    };
}

template <typename tfun1>
wlearner_affine_t<tfun1>::wlearner_affine_t() = default;

template <typename tfun1>
rwlearner_t wlearner_affine_t<tfun1>::clone() const
{
    return std::make_unique<wlearner_affine_t>(*this);
}

template <typename tfun1>
scalar_t wlearner_affine_t<tfun1>::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients,
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

    std::vector<cache_t> caches(tpool_t::size(), cache_t{dataset.tdim()});
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
        cache.clear();
        for (const auto i : indices)
        {
            const auto value = fvalues(i);
            if (!feature_t::missing(value))
            {
                cache.update(value, gradients.array(i));
            }
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

template <typename tfun1>
void wlearner_affine_t<tfun1>::predict(
    const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
{
    wlearner_feature1_t::predict(dataset, fold, range, outputs, [&] (scalar_t x, tensor_size_t i)
    {
        outputs.vector(i) = vector(0) * tfun1::get(x) + vector(1);
    });
}

template <typename tfun1>
cluster_t wlearner_affine_t<tfun1>::split(
    const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    cluster_t cluster(dataset.samples(fold), 1);
    wlearner_feature1_t::split(dataset, fold, indices, [&] (scalar_t, tensor_size_t i)
    {
        cluster.assign(i, 0);
    });

    return cluster;
}

template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::wlearner_fun1_cos_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::wlearner_fun1_lin_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::wlearner_fun1_log_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::wlearner_fun1_sin_t>;
