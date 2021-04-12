#include <iomanip>
#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_stump.h>

using namespace nano;
using namespace nano::gboost;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(const tensor3d_dims_t& tdims) :
            m_acc_sum(tdims),
            m_acc_neg(tdims),
            m_tables(cat_dims(2, tdims))
        {
        }

        auto x0_neg() const { return m_acc_neg.x0(); }
        auto r1_neg() const { return m_acc_neg.r1(); }
        auto r2_neg() const { return m_acc_neg.r2(); }

        auto x0_pos() const { return m_acc_sum.x0() - m_acc_neg.x0(); }
        auto r1_pos() const { return m_acc_sum.r1() - m_acc_neg.r1(); }
        auto r2_pos() const { return m_acc_sum.r2() - m_acc_neg.r2(); }

        void clear(const tensor4d_t& gradients, const tensor1d_t& values, const indices_t& samples)
        {
            m_acc_sum.clear();
            m_acc_neg.clear();

            m_ivalues.clear();
            m_ivalues.reserve(static_cast<size_t>(values.size()));
            for (tensor_size_t i = 0; i < values.size(); ++ i)
            {
                if (!feature_t::missing(values(i)))
                {
                    m_ivalues.emplace_back(values(i), samples(i));
                    m_acc_sum.update(gradients.array(samples(i)));
                }
            }
            std::sort(m_ivalues.begin(), m_ivalues.end());
        }

        auto output_neg() const
        {
            return r1_neg() / x0_neg();
        }

        auto output_pos() const
        {
            return r1_pos() / x0_pos();
        }

        template <typename tarray, typename toutputs>
        static auto score(const scalar_t r0, const tarray& r1, const tarray& r2, const toutputs& outputs)
        {
            return (r2 + outputs.square() * r0 - 2 * outputs * r1).sum();
        }

        auto score() const
        {
            return
                cache_t::score(x0_neg(), r1_neg(), r2_neg(), output_neg()) +
                cache_t::score(x0_pos(), r1_pos(), r2_pos(), output_pos());
        }

        using ivalues_t = std::vector<std::pair<scalar_t, tensor_size_t>>;

        // attributes
        ivalues_t       m_ivalues;                              ///<
        accumulator_t   m_acc_sum, m_acc_neg;                   ///<
        tensor4d_t      m_tables;                               ///<
        tensor_size_t   m_feature{-1};                          ///<
        scalar_t        m_threshold{0};                         ///<
        scalar_t        m_score{wlearner_t::no_fit_score()};    ///<
    };
}

wlearner_stump_t::wlearner_stump_t() = default;

void wlearner_stump_t::read(std::istream& stream)
{
    wlearner_feature1_t::read(stream);

    critical(
        !::nano::read(stream, m_threshold),
        "stump weak learner: failed to read from stream!");
}

void wlearner_stump_t::write(std::ostream& stream) const
{
    wlearner_feature1_t::write(stream);

    critical(
        !::nano::write(stream, m_threshold),
        "stump weak learner: failed to write to stream!");
}

rwlearner_t wlearner_stump_t::clone() const
{
    return std::make_unique<wlearner_stump_t>(*this);
}

scalar_t wlearner_stump_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.tdims()));

    std::vector<cache_t> caches(tpool_t::size(), cache_t{dataset.tdims()});
    wlearner_feature1_t::loopc(dataset, samples, [&] (tensor_size_t feature, const tensor1d_t& fvalues, size_t tnum)
    {
        // update accumulators
        auto& cache = caches[tnum];
        cache.clear(gradients, fvalues, samples);
        for (size_t iv = 0, sv = cache.m_ivalues.size(); iv + 1 < sv; ++ iv)
        {
            const auto& ivalue1 = cache.m_ivalues[iv + 0];
            const auto& ivalue2 = cache.m_ivalues[iv + 1];

            cache.m_acc_neg.update(gradients.array(ivalue1.second));

            if (ivalue1.first < ivalue2.first)
            {
                // update the parameters if a better feature
                const auto score = cache.score();
                if (std::isfinite(score) && score < cache.m_score)
                {
                    cache.m_score = score;
                    cache.m_feature = feature;
                    cache.m_threshold = 0.5 * (ivalue1.first + ivalue2.first);
                    cache.m_tables.array(0) = cache.output_neg();
                    cache.m_tables.array(1) = cache.output_pos();
                }
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);

    log_info() << std::fixed << std::setprecision(8) << " === stump(feature=" << best.m_feature << "|"
        << (best.m_feature >= 0 ? dataset.feature(best.m_feature).name() : string_t("N/A"))
        << ",threshold=" << best.m_threshold << "),samples=" << samples.size()
        << ",score=" << (best.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(best.m_score)) << ".";

    set(best.m_feature, best.m_tables);
    m_threshold = best.m_threshold;
    return best.m_score;
}

void wlearner_stump_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    wlearner_feature1_t::predict(dataset, samples, outputs, [&] (scalar_t x, tensor3d_map_t&& outputs)
    {
        outputs.vector() += vector(x < m_threshold ? 0 : 1);
    });
}

cluster_t wlearner_stump_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    return wlearner_feature1_t::split(dataset, samples, 2, [&] (scalar_t x)
    {
        return (x < m_threshold) ? 0 : 1;
    });
}
