#include <nano/logger.h>
#include <nano/gboost/wlearner_product.h>

using namespace nano;

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
