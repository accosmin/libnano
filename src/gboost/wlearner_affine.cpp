#include <iomanip>
#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/gboost/wlearner_affine.h>

using namespace nano;
using namespace nano::gboost;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(const tensor3d_dims_t& tdims) :
            m_acc(tdims),
            m_tables(cat_dims(2, tdims))
        {
        }

        auto x0() const { return m_acc.x0(); }
        auto x1() const { return m_acc.x1(); }
        auto x2() const { return m_acc.x2(); }
        auto r1() const { return m_acc.r1(); }
        auto rx() const { return m_acc.rx(); }
        auto r2() const { return m_acc.r2(); }

        void clear()
        {
            m_acc.clear();
        }

        auto a() const
        {
            return (rx() * x0() - r1() * x1()) / (x2() * x0() - x1() * x1());
        }

        auto b() const
        {
            return (r1() * x2() - rx() * x1()) / (x2() * x0() - x1() * x1());
        }

        auto score() const
        {
            return (r2() + a().square() * x2() + b().square() * x0() -
                    2 * a() * rx() - 2 * b() * r1() + 2 * a() * b() * x1()).sum();
        }

        // attributes
        accumulator_t   m_acc;                                  ///<
        tensor4d_t      m_tables;                               ///<
        tensor_size_t   m_feature{0};                           ///<
        scalar_t        m_score{wlearner_t::no_fit_score()};    ///<
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
scalar_t wlearner_affine_t<tfun1>::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.tdims()));

    std::vector<cache_t> caches(tpool_t::size(), cache_t{dataset.tdims()});
    wlearner_feature1_t::loopc(dataset, samples,
        [&] (tensor_size_t feature, const tensor1d_t& fvalues, size_t tnum)
    {
        // update accumulators
        auto& cache = caches[tnum];
        cache.clear();
        for (tensor_size_t i = 0; i < samples.size(); ++ i)
        {
            const auto value = fvalues(i);
            if (!feature_t::missing(value))
            {
                cache.m_acc.update(tfun1::get(value), gradients.array(samples(i)));
            }
        }

        // update the parameters if a better feature
        const auto score = cache.score();
        if (std::isfinite(score) && score < cache.m_score)
        {
            cache.m_score = score;
            cache.m_feature = feature;
            cache.m_tables.array(0) = cache.a();
            cache.m_tables.array(1) = cache.b();
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === affine(feature=" << best.m_feature << "|"
        << (best.m_feature >= 0 ? dataset.feature(best.m_feature).name() : string_t("N/A"))
        << "),samples=" << samples.size()
        << ",score=" << (best.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(best.m_score)) << ".";

    set(best.m_feature, best.m_tables);
    return best.m_score;
}

template <typename tfun1>
void wlearner_affine_t<tfun1>::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    wlearner_feature1_t::predict(dataset, samples, outputs, [&] (scalar_t x, tensor3d_map_t&& outputs)
    {
        outputs.vector() += vector(0) * tfun1::get(x) + vector(1);
    });
}

template <typename tfun1>
cluster_t wlearner_affine_t<tfun1>::split(const dataset_t& dataset, const indices_t& samples) const
{
    return wlearner_feature1_t::split(dataset, samples, 1, [&] (scalar_t)
    {
        return 0;
    });
}

template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_cos_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_lin_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_log_t>;
template class NANO_PUBLIC ::nano::wlearner_affine_t<::nano::fun1_sin_t>;
