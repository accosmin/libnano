#include <iomanip>
#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_dstep.h>

using namespace nano;
using namespace nano::gboost;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(const tensor3d_dims_t& tdims) :
            m_beta0(tdims),
            m_acc(tdims)
        {
            m_beta0.zero();
        }

        auto& x0(tensor_size_t fv) { return m_acc.x0(fv); }
        auto r1(tensor_size_t fv) { return m_acc.r1(fv); }
        auto r2(tensor_size_t fv) { return m_acc.r2(fv); }

        auto x0(tensor_size_t fv) const { return m_acc.x0(fv); }
        auto r1(tensor_size_t fv) const { return m_acc.r1(fv); }
        auto r2(tensor_size_t fv) const { return m_acc.r2(fv); }

        void clear(tensor_size_t n_fvalues)
        {
            m_acc.clear(n_fvalues);
            m_scores.resize(2, n_fvalues);
        }

        auto beta0() const
        {
            return m_beta0.array();
        }

        auto beta(tensor_size_t fv) const
        {
            return r1(fv) / x0(fv);
        }

        template <typename tbarray>
        scalar_t score(const tensor_size_t fv, const tbarray& beta) const
        {
            return (r2(fv) + beta.square() * x0(fv) - 2 * beta * r1(fv)).sum();
        }

        // attributes
        tensor3d_t      m_beta0;                                ///<
        accumulator_t   m_acc;                                  ///<
        tensor4d_t      m_tables;                               ///<
        tensor_size_t   m_feature{-1};                          ///<
        tensor_size_t   m_fvalue{-1};                           ///<
        scalar_t        m_score{wlearner_t::no_fit_score()};    ///<
        tensor2d_t      m_scores;                               ///<
    };
}

wlearner_dstep_t::wlearner_dstep_t() = default;

rwlearner_t wlearner_dstep_t::clone() const
{
    return std::make_unique<wlearner_dstep_t>(*this);
}

void wlearner_dstep_t::read(std::istream& stream)
{
    wlearner_feature1_t::read(stream);

    critical(
        !::nano::read_cast<int64_t>(stream, m_fvalue),
        "dstep weak learner: failed to read from stream!");
}

void wlearner_dstep_t::write(std::ostream& stream) const
{
    wlearner_feature1_t::write(stream);

    critical(
        !::nano::write(stream, static_cast<int64_t>(m_fvalue)),
        "dstep weak learner: failed to write to stream!");
}

scalar_t wlearner_dstep_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.tdims()));

    std::vector<cache_t> caches(tpool_t::size(), cache_t{dataset.tdims()});
    wlearner_feature1_t::loopd(dataset, samples,
        [&] (tensor_size_t feature, const tensor1d_t& fvalues, tensor_size_t n_fvalues, size_t tnum)
    {
        // update accumulators
        auto& cache = caches[tnum];
        cache.clear(n_fvalues);
        for (tensor_size_t i = 0; i < fvalues.size(); ++ i)
        {
            const auto value = fvalues(i);
            if (feature_t::missing(value))
            {
                continue;
            }

            const auto fv = static_cast<tensor_size_t>(value);
            critical(fv < 0 || fv >= n_fvalues,
                scat("dstep weak learner: invalid feature value ", fv, ", expecting [0, ", n_fvalues, ")"));

            cache.m_acc.update(gradients.array(samples(i)), fv);
        }

        // update the parameters if a better feature
        for (tensor_size_t fv = 0; fv < n_fvalues; ++ fv)
        {
            cache.m_scores(0, fv) = cache.score(fv, cache.beta0());
            cache.m_scores(1, fv) = cache.score(fv, cache.beta(fv));
        }
        for (tensor_size_t fv = 0; fv < n_fvalues; ++ fv)
        {
            const auto score = cache.m_scores.array(0).sum() - cache.m_scores(0, fv) + cache.m_scores(1, fv);
            if (std::isfinite(score) && score < cache.m_score)
            {
                cache.m_score = score;
                cache.m_fvalue = fv;
                cache.m_feature = feature;
                cache.m_tables.resize(cat_dims(n_fvalues, dataset.tdims()));
                cache.m_tables.zero();
                cache.m_tables.array(fv) = cache.beta(fv);
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === dstep(feature=" << best.m_feature << "|"
        << (best.m_feature >= 0 ? dataset.feature(best.m_feature).name() : string_t("N/A"))
        << ",fvalues=" << best.m_tables.size<0>() << ",fvalue=" << best.m_fvalue << "),samples=" << samples.size()
        << ",score=" << (best.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(best.m_score)) << ".";

    set(best.m_feature, best.m_tables, static_cast<size_t>(best.m_tables.size<0>()));
    m_fvalue = best.m_fvalue;
    return best.m_score;
}

void wlearner_dstep_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    wlearner_feature1_t::predict(dataset, samples, outputs, [&] (scalar_t x, tensor3d_map_t&& outputs)
    {
        const auto index = static_cast<tensor_size_t>(x);
        critical(
            index < 0 || index >= fvalues(),
            scat("dstep weak learner: invalid feature value ", x, ", expecting [0, ", fvalues(), ")"));
        outputs.vector() += vector(index);
    });
}

cluster_t wlearner_dstep_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    return wlearner_feature1_t::split(dataset, samples, 1, [&] (scalar_t)
    {
        return 0;
    });
}
