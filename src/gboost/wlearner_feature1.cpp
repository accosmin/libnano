#include <nano/logger.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_feature1.h>

using namespace nano;

wlearner_feature1_t::wlearner_feature1_t() = default;

void wlearner_feature1_t::set(tensor_size_t feature, const tensor4d_t& tables, size_t labels)
{
    m_tables = tables;
    m_labels = labels;
    m_feature = feature;
}

void wlearner_feature1_t::read(std::istream& stream)
{
    wlearner_t::read(stream);

    critical(
        !::nano::detail::read(stream, m_feature) ||
        !::nano::read(stream, m_tables),
        "feature1 weak learner: failed to read from stream!");
}

void wlearner_feature1_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(
        !::nano::detail::write(stream, m_feature) ||
        !::nano::write(stream, m_tables),
        "feature1 weak learner: failed to write to stream!");
}

tensor3d_dim_t wlearner_feature1_t::odim() const
{
    return make_dims(m_tables.size<1>(), m_tables.size<2>(), m_tables.size<3>());
}

void wlearner_feature1_t::scale(const vector_t& scale)
{
    wlearner_t::scale(m_tables, scale);
}

void wlearner_feature1_t::compatible(const dataset_t& dataset) const
{
    critical(
        m_tables.size<0>() == 0,
        "feature1 weak learner: empty weak learner!");

    critical(
        odim() != dataset.tdim() ||
        m_feature < 0 || m_feature >= dataset.features() ||
        dataset.ifeature(m_feature).labels().size() != m_labels,
        "feature1 weak learner: mis-matching dataset!");
}

indices_t wlearner_feature1_t::features() const
{
    return std::array<tensor_size_t, 1>{{m_feature}};
}
