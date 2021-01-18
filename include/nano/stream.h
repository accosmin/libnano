#pragma once

#include <istream>
#include <ostream>
#include <type_traits>
#include <nano/logger.h>
#include <nano/string.h>
#include <nano/version.h>

namespace nano
{
    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
    >
    // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
    std::ostream& write(std::ostream& stream, const tscalar scalar)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return stream.write(reinterpret_cast<const char*>(&scalar), sizeof(tscalar));
    }

    template
    <
        typename tscalar, typename tcount,
        typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
    >
    std::ostream& write(std::ostream& stream, const tscalar* data, const tcount count)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return stream.write(reinterpret_cast<const char*>(data), sizeof(tscalar) * count);
    }

    template
    <
        typename twscalar, typename tscalar, typename tcount,
        typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
    >
    std::ostream& write_cast(std::ostream& stream, const tscalar* data, const tcount count)
    {
        for (tcount i = 0; i < count; ++ i)
        {
            write(stream, static_cast<twscalar>(data[i]));
        }
        return stream;
    }

    inline std::ostream& write(std::ostream& stream, const std::string& string)
    {
        write(stream, static_cast<uint32_t>(string.size()));
        return write(stream, string.data(), string.size());
    }

    template <typename tvalue>
    std::ostream& write(std::ostream& stream, const std::vector<tvalue>& values)
    {
        if (!write(stream, static_cast<uint64_t>(values.size())))
        {
            return stream;
        }

        for (const auto& value : values)
        {
            if (!write(stream, value))
            {
                return stream;
            }
        }
        return stream;
    }

    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
    >
    std::istream& read(std::istream& stream, tscalar& scalar)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return stream.read(reinterpret_cast<char*>(&scalar), sizeof(tscalar));
    }

    template
    <
        typename tscalar, typename tcount,
        typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
    >
    std::istream& read(std::istream& stream, tscalar* data, const tcount count)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return stream.read(reinterpret_cast<char*>(data), sizeof(tscalar) * count);
    }

    template
    <
        typename trscalar, typename tscalar,
        typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
    >
    std::istream& read_cast(std::istream& stream, tscalar& scalar)
    {
        trscalar value{};
        read(stream, value);
        scalar = static_cast<tscalar>(value);
        return stream;
    }

    template
    <
        typename trscalar, typename tscalar, typename tcount,
        typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
    >
    std::istream& read_cast(std::istream& stream, tscalar* data, const tcount count)
    {
        for (tcount i = 0; i < count; ++ i)
        {
            read_cast<trscalar>(stream, data[i]);
        }
        return stream;
    }

    inline std::istream& read(std::istream& stream, std::string& string)
    {
        uint32_t size = 0;
        if (!read(stream, size))
        {
            return stream;
        }

        string.resize(size);
        for (char& c : string)
        {
            read(stream, c);
        }
        return stream;
    }

    template <typename tvalue>
    std::istream& read(std::istream& stream, std::vector<tvalue>& values)
    {
        uint64_t size = 0;
        if (!read(stream, size))
        {
            return stream;
        }

        values.resize(size);
        for (auto& value : values)
        {
            if (!read(stream, value))
            {
                return stream;
            }
        }
        return stream;
    }

    ///
    /// \brief interface for serializable object with support for:
    ///     - versioning with automatic checks.
    ///
    class NANO_PUBLIC serializable_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        serializable_t() = default;

        ///
        /// \brief enable copying
        ///
        serializable_t(const serializable_t&) = default;
        serializable_t& operator=(const serializable_t&) = default;

        ///
        /// \brief enable moving
        ///
        serializable_t(serializable_t&&) noexcept = default;
        serializable_t& operator=(serializable_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        virtual ~serializable_t() = default;

        ///
        /// \brief serialize from the given binary stream.
        ///
        /// NB: any error is considered critical and expected to result in an exception.
        ///
        virtual void read(std::istream&);

        ///
        /// \brief serialize to the given binary stream.
        ///
        /// NB: any error is considered critical and expected to result in an exception.
        ///
        virtual void write(std::ostream&) const;

        ///
        /// \brief access functions
        ///
        auto major_version() const { return m_major_version; }
        auto minor_version() const { return m_minor_version; }
        auto patch_version() const { return m_patch_version; }

    private:

        // attributes
        int32_t     m_major_version{::nano::major_version}; ///<
        int32_t     m_minor_version{::nano::minor_version}; ///<
        int32_t     m_patch_version{::nano::patch_version}; ///<
    };

    ///
    /// \brief wraps serializable objects with an associated factory ID
    ///     to support type-safe binary serialization.
    ///
    /// NB: the wrapped object must implement:
    ///     - ::all() to return its associated factory and
    ///     - .clone() to create a copy as stored in its associated factory.
    ///
    template
    <
        typename tobject,
        typename trobject = std::unique_ptr<tobject>,
        typename = typename std::enable_if<std::is_base_of<serializable_t, tobject>::value>::type
    >
    class NANO_PUBLIC identifiable_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        identifiable_t() = default;

        ///
        /// \brief constructor
        ///
        identifiable_t(std::string&& id, trobject&& object) :
            m_id(std::move(id)),
            m_object(std::move(object))
        {
        }

        ///
        /// \brief enable moving
        ///
        identifiable_t(identifiable_t&&) noexcept = default;
        identifiable_t& operator=(identifiable_t&&) noexcept = default;

        ///
        /// \brief enable copying by cloning
        ///
        identifiable_t(const identifiable_t& other) :
            m_id(other.m_id)
        {
            if (other.m_object != nullptr)
            {
                m_object = other.m_object->clone();
            }
        }
        identifiable_t& operator=(const identifiable_t& other)// NOLINT(cert-oop54-cpp)
        {
            if (this != &other)
            {
                m_id = other.m_id;
                if (other.m_object != nullptr)
                {
                    m_object = other.m_object->clone();
                }
                else
                {
                    m_object.reset();
                }
            }
            return *this;
        }

        ///
        /// \brief destructor
        ///
        ~identifiable_t() = default;

        ///
        /// \brief serialize from binary stream.
        ///
        void read(std::istream& stream)
        {
            critical(
                !::nano::read(stream, m_id),
                "identifiable: failed to read from stream!");

            m_object = tobject::all().get(m_id);
            critical(
                m_object == nullptr,
                "identifiable: invalid id <", m_id, "> read from stream!");

            m_object->read(stream);
        }

        ///
        /// \brief serialize to binary stream.
        ///
        void write(std::ostream& stream) const
        {
            critical(
                !::nano::write(stream, m_id),
                "identifiable: failed to write to stream!");

            critical(
                m_object == nullptr,
                "identifiable: cannot serialize unitialized object!");
            m_object->write(stream);
        }

        ///
        /// \brief returns true if the wrapped object is initialized.
        ///
        explicit operator bool() const
        {
            return m_object != nullptr;
        }

        ///
        /// \brief access functions
        ///
        const auto& id() const { return m_id; }
        auto& get() { assert(m_object != nullptr); return *m_object; }
        const auto& get() const { assert(m_object != nullptr); return *m_object; }

    private:

        // attributes
        std::string     m_id;           ///<
        trobject        m_object;       ///<
    };

    inline std::istream& read(std::istream& stream, serializable_t& object)
    {
        object.read(stream);
        return stream;
    }

    template <typename tobject>
    std::istream& read(std::istream& stream, identifiable_t<tobject>& object)
    {
        object.read(stream);
        return stream;
    }

    inline std::ostream& write(std::ostream& stream, const serializable_t& object)
    {
        object.write(stream);
        return stream;
    }

    template <typename tobject>
    std::ostream& write(std::ostream& stream, const identifiable_t<tobject>& object)
    {
        object.write(stream);
        return stream;
    }
}
