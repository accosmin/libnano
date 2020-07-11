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

tensor3d_dim_t wlearner_product_t::odim() const
{
    critical(
        m_terms.empty(),
        "product weak learner: no term fitted!");

    return m_terms.begin()->m_wlearner->odim();
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

//scalar_t fit(const dataset_t&, fold_t, const tensor4d_t&, const indices_t&, const tensor4d_t&);
//cluster_t split(const dataset_t&, fold_t, const indices_t&) const;
//void predict(const dataset_t&, fold_t, tensor_range_t, tensor4d_map_t&& outputs) const;
