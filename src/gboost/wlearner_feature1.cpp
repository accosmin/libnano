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
        !::nano::read_cast<int64_t>(stream, m_feature) ||
        !::nano::read(stream, m_tables),
        "feature1 weak learner: failed to read from stream!");
}

void wlearner_feature1_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(
        !::nano::write(stream, static_cast<int64_t>(m_feature)) ||
        !::nano::write(stream, m_tables),
        "feature1 weak learner: failed to write to stream!");
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
        make_dims(m_tables.size<1>(), m_tables.size<2>(), m_tables.size<3>()) != dataset.tdims() ||
        m_feature < 0 || m_feature >= dataset.features() ||
        dataset.feature(m_feature).labels().size() != m_labels,
        "feature1 weak learner: mis-matching dataset!");
}

indices_t wlearner_feature1_t::features() const
{
    return make_tensor<tensor_size_t>(make_dims(1), m_feature);
}
