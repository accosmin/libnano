#include <nano/logger.h>
#include <nano/gboost/wlearner_product.h>

using namespace nano;

wlearner_product_t::wlearner_product_t() = default;

wlearner_product_t::wlearner_product_t(wlearner_product_t&&) = default;

wlearner_product_t::wlearner_product_t(const wlearner_product_t&) = default;

wlearner_product_t& wlearner_product_t::operator=(wlearner_product_t&&) = default;

wlearner_product_t& wlearner_product_t::operator=(const wlearner_product_t&) = default;

void wlearner_product_t::add(const string_t& id)
{
    add(id, wlearner_t::all().get(id));
}

void wlearner_product_t::add(string_t id, rwlearner_t&& prototype)
{
    critical(prototype == nullptr, "product weak learner: invalid prototype weak learner!");

    m_protos.emplace_back(std::move(id), std::move(prototype));
}

void wlearner_product_t::degree(int degree)
{
    m_degree = degree;
}

void wlearner_product_t::read(std::istream& stream)
{
    wlearner_t::read(stream);

    int32_t idegree = 0;

    critical(
        !::nano::detail::read(stream, idegree),
        "product weak learner: failed to read from stream!");

    degree(idegree);

    iwlearner_t::read(stream, m_protos);
    iwlearner_t::read(stream, m_terms);
}

void wlearner_product_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(
        !::nano::detail::write(stream, static_cast<int32_t>(degree())),
        "product weak learner: failed to write to stream!");

    iwlearner_t::write(stream, m_protos);
    iwlearner_t::write(stream, m_terms);
}

rwlearner_t wlearner_product_t::clone() const
{
    return std::make_unique<wlearner_product_t>(*this);
}

void wlearner_product_t::scale(const vector_t& scale)
{
    critical(
        m_terms.empty(),
        "product weak learner: no term fitted!");

    m_terms.begin()->m_wlearner->scale(scale);
}

indices_t wlearner_product_t::features() const
{
    std::vector<tensor_size_t> features;
    for (const auto& term : m_terms)
    {
        const auto term_features = term.m_wlearner->features();
        features.insert(features.end(), begin(term_features), end(term_features));
    }

    std::sort(features.begin(), features.end());
    features.erase(std::unique(features.begin(), features.end()), features.end());

    return map_tensor(features.data(), static_cast<tensor_size_t>(features.size()));
}

scalar_t wlearner_product_t::fit(const dataset_t& dataset, fold_t fold,
    const tensor4d_t& gradients, const indices_t& indices, const tensor4d_t&)
{
    const auto samples = dataset.samples(fold);

    tensor4d_t scales(cat_dims(samples, dataset.tdim()));
    tensor4d_t woutputs(cat_dims(samples, dataset.tdim()));

    scales.constant(1.0);
    woutputs.constant(0.0);

    auto best_score = std::numeric_limits<scalar_t>::max();
    for (auto deg = degree(); deg > 0; -- deg)
    {
        auto best_id = std::string{};
        auto best_wlearner = rwlearner_t{};
        best_score = std::numeric_limits<scalar_t>::max();
        for (const auto& prototype : m_protos)
        {
            auto wlearner = prototype.m_wlearner->clone();
            assert(wlearner);

            const auto score = wlearner->fit(dataset, fold, gradients, indices, scales);
            if (score < best_score)
            {
                best_id = prototype.m_id;
                best_score = score;
                best_wlearner = std::move(wlearner);
            }
        }

        if (!best_wlearner)
        {
            log_warning() << "cannot fit any new weak learner, stopping.";
            break;
        }

        // update scales to fit to if not the last term
        if (deg > 1)
        {
            dataset.loop(execution::par, fold, batch(), [&] (tensor_range_t range, size_t)
            {
                best_wlearner->predict(dataset, fold, range, woutputs.slice(range));
            });

            scales.array() *= woutputs.array();
        }

        // OK, store the weak learner as a new term in the product
        m_terms.emplace_back(std::move(best_id), std::move(best_wlearner));
    }

    // OK
    return best_score;
}

cluster_t wlearner_product_t::split(const dataset_t& dataset, fold_t fold, const indices_t& indices) const
{
    cluster_t cluster(dataset.samples(fold), 1);

    // the resulting split is the intersection of the terms' splits
    tensor_size_t deg = 0;
    for (const auto& term : m_terms)
    {
        const auto wcluster = term.m_wlearner->split(dataset, fold, indices);
        assert(wcluster.samples() == cluster.samples());

        if (deg == 0)
        {
            cluster = wcluster;
        }
        else
        {
            for (tensor_size_t i = 0; i < cluster.samples(); ++ i)
            {
                if (wcluster.group(i) < 0)
                {
                    cluster.assign(i, -1);
                }
            }
        }
        ++ deg;
    }

    return cluster;
}

void wlearner_product_t::predict(const dataset_t& dataset, fold_t fold, tensor_range_t range,
    tensor4d_map_t&& outputs) const
{
    outputs.constant(1.0);
    tensor4d_t woutputs(outputs.dims());
    for (const auto& term : m_terms)
    {
        term.m_wlearner->predict(dataset, fold, range, woutputs.tensor());
        outputs.array() *= woutputs.array();
    }
}
