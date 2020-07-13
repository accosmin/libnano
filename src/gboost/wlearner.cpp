#include <mutex>
#include <nano/logger.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_dtree.h>
#include <nano/gboost/wlearner_stump.h>
#include <nano/gboost/wlearner_table.h>
#include <nano/gboost/wlearner_affine.h>

using namespace nano;

void wlearner_t::batch(int batch)
{
    m_batch = batch;
}

void wlearner_t::type(wlearner type)
{
    m_type = type;
}

void wlearner_t::read(std::istream& stream)
{
    serializable_t::read(stream);

    int32_t itype = 0;
    int32_t ibatch = 0;

    critical(
        !::nano::detail::read(stream, itype) ||
        !::nano::detail::read(stream, ibatch),
        "weak learner: failed to read from stream!");

    type(static_cast<wlearner>(itype));
    batch(ibatch);
}

void wlearner_t::write(std::ostream& stream) const
{
    serializable_t::write(stream);

    critical(
        !::nano::detail::write(stream, static_cast<int32_t>(type())) ||
        !::nano::detail::write(stream, static_cast<int32_t>(batch())),
        "weak learner: failed to write to stream!");
}

void wlearner_t::check(const indices_t& indices)
{
    critical(
        !std::is_sorted(::nano::begin(indices), ::nano::end(indices)),
        "weak learner: indices must be sorted!");
}

void wlearner_t::scale(tensor4d_t& tables, const vector_t& scale)
{
    critical(
        scale.size() != 1 && scale.size() != tables.size<0>(),
        "weak learner: mis-matching scale!");

    critical(
        scale.minCoeff() < 0,
        "weak learner: invalid scale factors!");

    for (tensor_size_t i = 0; i < tables.size<0>(); ++ i)
    {
        tables.array(i) *= scale(std::min(i, scale.size() - 1));
    }
}

wlearner_factory_t& wlearner_t::all()
{
    static wlearner_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add_by_type<wlearner_lin1_t>();
        manager.add_by_type<wlearner_log1_t>();
        manager.add_by_type<wlearner_cos1_t>();
        manager.add_by_type<wlearner_sin1_t>();
        manager.add_by_type<wlearner_stump_t>();
        manager.add_by_type<wlearner_table_t>();
        manager.add_by_type<wlearner_dtree_t>();
    });

    return manager;
}

iwlearner_t::iwlearner_t() = default;

iwlearner_t::iwlearner_t(iwlearner_t&&) noexcept = default;

iwlearner_t::iwlearner_t(const iwlearner_t& other) :
    m_id(other.m_id)
{
    if (static_cast<bool>(other.m_wlearner))
    {
        m_wlearner = other.m_wlearner->clone();
    }
}

iwlearner_t& iwlearner_t::operator=(iwlearner_t&&) noexcept = default;

iwlearner_t& iwlearner_t::operator=(const iwlearner_t& other)
{
    if (this != &other)
    {
        m_id = other.m_id;
        if (static_cast<bool>(other.m_wlearner))
        {
            m_wlearner = other.m_wlearner->clone();
        }
    }

    return *this;
}

iwlearner_t::iwlearner_t(string_t&& id, rwlearner_t&& wlearner) :
    m_id(std::move(id)),
    m_wlearner(std::move(wlearner))
{
}

void iwlearner_t::read(std::istream& stream)
{
    critical(
        !::nano::detail::read(stream, m_id),
        "wlearner wid: failed to read from stream!");

    m_wlearner = wlearner_t::all().get(m_id);
    critical(
        m_wlearner == nullptr,
        scat("wlearner wid: invalid weak learner id <", m_id, "> read from stream!"));

    m_wlearner->read(stream);
}

void iwlearner_t::write(std::ostream& stream) const
{
    critical(
        !::nano::detail::write(stream, m_id),
        "wlearner wid: failed to write to stream!");

    m_wlearner->write(stream);
}

void iwlearner_t::read(std::istream& stream, std::vector<iwlearner_t>& protos)
{
    uint32_t size = 0;
    critical(
        !::nano::detail::read(stream, size),
        "weak learner: failed to read from stream!");

    protos.resize(size);
    for (auto& proto : protos)
    {
        proto.read(stream);
    }
}

void iwlearner_t::write(std::ostream& stream, const std::vector<iwlearner_t>& protos)
{
    critical(
        !::nano::detail::write(stream, static_cast<uint32_t>(protos.size())),
        "weak learner: failed to write to stream!");

    for (const auto& proto : protos)
    {
        proto.write(stream);
    }
}
