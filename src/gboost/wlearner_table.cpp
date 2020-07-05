#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_table.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        auto r0(tensor_size_t fv) { return m_accumulators.array(fv, 0); }
        auto r1(tensor_size_t fv) { return m_accumulators.array(fv, 1); }
        auto r2(tensor_size_t fv) { return m_accumulators.array(fv, 2); }

        [[nodiscard]] auto r0(tensor_size_t fv) const { return m_accumulators.array(fv, 0); }
        [[nodiscard]] auto r1(tensor_size_t fv) const { return m_accumulators.array(fv, 1); }
        [[nodiscard]] auto r2(tensor_size_t fv) const { return m_accumulators.array(fv, 2); }

        void clear(const tensor_size_t n_fvalues, const tensor3d_dim_t tdim)
        {
            m_accumulators.resize(cat_dims(n_fvalues, cat_dims(3, tdim)));
            m_accumulators.zero();
        }

        [[nodiscard]] auto outputs_real(const tensor_size_t fv) const
        {
            return r1(fv) / r0(fv);
        }

        [[nodiscard]] auto outputs_discrete(const tensor_size_t fv) const
        {
            return outputs_real(fv).sign();
        }

        template <typename toutputs>
        [[nodiscard]] scalar_t score(const tensor_size_t fv, const toutputs& outputs) const
        {
            return (r2(fv) + outputs.square() * r0(fv) - 2 * outputs * r1(fv)).sum();
        }

        [[nodiscard]] auto score(const wlearner type) const
        {
            scalar_t score = 0;
            switch (type)
            {
            case wlearner::real:
                for (tensor_size_t fv = 0, n_fvalues = m_accumulators.size<0>(); fv < n_fvalues; ++ fv)
                {
                    score += this->score(fv, outputs_real(fv));
                }
                break;

            case wlearner::discrete:
                for (tensor_size_t fv = 0, n_fvalues = m_accumulators.size<0>(); fv < n_fvalues; ++ fv)
                {
                    score += this->score(fv, outputs_discrete(fv));
                }
                break;

            default:
                assert(false);
                break;
            }
            return score;
        }

        template <typename tgarray, typename tsarray>
        void update(tensor_size_t fv, tgarray&& vgrad, tsarray&& scale)
        {
            r0(fv) += scale.square();
            r1(fv) -= scale * vgrad;
            r2(fv) += vgrad * vgrad;
        }

        // attributes
        tensor_size_t   m_feature{-1};                                  ///<
        tensor4d_t      m_tables;                                       ///<
        tensor5d_t      m_accumulators;                                 ///< (#feature_values, 3, tdim)
        scalar_t        m_score{std::numeric_limits<scalar_t>::max()};  ///<
    };
}

wlearner_table_t::wlearner_table_t() = default;

rwlearner_t wlearner_table_t::clone() const
{
    return std::make_unique<wlearner_table_t>(*this);
}

scalar_t wlearner_table_t::fit(const dataset_t& dataset, fold_t fold, const tensor4d_t& gradients,
    const indices_t& indices, const tensor4d_t& scales)
{
    assert(indices.min() >= 0);
    assert(indices.max() < dataset.samples(fold));
    assert(scales.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));
    assert(gradients.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    switch (type())
    {
    case wlearner::real:
    case wlearner::discrete:
        break;

    default:
        critical(true, "table weak learner: unhandled wlearner");
        break;
    }

    std::vector<cache_t> caches(tpool_t::size());
    loopi(dataset.features(), [&] (tensor_size_t feature, size_t tnum)
    {
        const auto& ifeature = dataset.ifeature(feature);

        // NB: This weak learner works only with discrete features!
        if (!ifeature.discrete())
        {
            return;
        }

        const auto n_fvalues = static_cast<tensor_size_t>(ifeature.labels().size());
        const auto fvalues = dataset.inputs(fold, make_range(0, dataset.samples(fold)), feature);

        // update accumulators
        auto& cache = caches[tnum];
        cache.clear(n_fvalues, dataset.tdim());
        for (const auto i : indices)
        {
            const auto value = fvalues(i);
            if (feature_t::missing(value))
            {
                continue;
            }

            const auto fv = static_cast<tensor_size_t>(value);
            critical(fv < 0 || fv >= n_fvalues,
                scat("table weak learner: invalid feature value ", fv, ", expecting [0, ", n_fvalues, ")"));

            cache.update(fv, gradients.array(i), scales.array(i));
        }

        // update the parameters if a better feature
        const auto score = cache.score(type());
        if (std::isfinite(score) && score < cache.m_score)
        {
            cache.m_score = score;
            cache.m_feature = feature;
            cache.m_tables.resize(cat_dims(n_fvalues, dataset.tdim()));
            switch (type())
            {
            case wlearner::real:
                for (tensor_size_t fv = 0; fv < n_fvalues; ++ fv)
                {
                    cache.m_tables.array(fv) = cache.outputs_real(fv);
                }
                break;

            case wlearner::discrete:
                for (tensor_size_t fv = 0; fv < n_fvalues; ++ fv)
                {
                    cache.m_tables.array(fv) = cache.outputs_discrete(fv);
                }
                break;

            default:
                assert(false);
                break;
            }
        }
    });

    // OK, return and store the optimum feature across threads
    const auto& best = ::nano::gboost::min_reduce(caches);
    set(best.m_feature, best.m_tables, static_cast<size_t>(best.m_tables.size<0>()));
    return best.m_score;
}

void wlearner_table_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t&& outputs) const
{
    wlearner_feature1_t::predict(dataset, fold, range, outputs, [&] (scalar_t x, tensor_size_t i)
    {
        const auto index = static_cast<tensor_size_t>(x);
        critical(
            index < 0 || index >= n_fvalues(),
            scat("table weak learner: invalid feature value ", x, ", expecting [0, ", n_fvalues(), ")"));
        outputs.vector(i) = vector(index);
    });
}

cluster_t wlearner_table_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    cluster_t cluster(dataset.samples(fold), n_fvalues());
    wlearner_feature1_t::split(dataset, fold, indices, [&] (scalar_t x, tensor_size_t i)
    {
        const auto index = static_cast<tensor_size_t>(x);
        critical(
            index < 0 || index >= n_fvalues(),
            scat("table weak learner: invalid feature value ", x, ", expecting [0, ", n_fvalues(), ")"));
        cluster.assign(i, index);
    });

    return cluster;
}
