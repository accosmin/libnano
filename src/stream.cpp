#include <nano/stream.h>
#include <nano/core/logger.h>
#include <nano/tensor/stream.h>

using namespace nano;

void serializable_t::read(std::istream& stream)
{
    critical(
        !::nano::read(stream, m_major_version) ||
        !::nano::read(stream, m_minor_version) ||
        !::nano::read(stream, m_patch_version),
        "serializable: failed to read from stream!");

    critical(
        m_major_version > nano::major_version ||
        (m_major_version == nano::major_version &&
         m_minor_version > nano::minor_version) ||
        (m_major_version == nano::major_version &&
         m_minor_version == nano::minor_version &&
         m_patch_version > nano::patch_version),
        "serializable: version mismatch!");
}

void serializable_t::write(std::ostream& stream) const
{
    critical(
        !::nano::write(stream, static_cast<int32_t>(nano::major_version)) ||
        !::nano::write(stream, static_cast<int32_t>(nano::minor_version)) ||
        !::nano::write(stream, static_cast<int32_t>(nano::patch_version)),
        "serializable: failed to write to stream");
}
