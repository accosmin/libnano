#pragma once

#include <iostream>
#include <nano/arch.h>
#include <nano/string.h>

namespace nano
{
    ///
    /// \brief logging object.
    ///
    class NANO_PUBLIC logger_t
    {
    public:

        enum class type
        {
            info,
            warn,
            error
        };

        ///
        /// \brief constructor
        ///
        explicit logger_t(type, std::ostream* cout = &std::cout, std::ostream* cerr = &std::cerr);

        ///
        /// \brief disable copying
        ///
        logger_t(const logger_t&) = delete;
        logger_t& operator=(const logger_t&) = delete;

        ///
        /// \brief disable moving
        ///
        logger_t(logger_t&&) noexcept = default;
        logger_t& operator=(logger_t&&) noexcept = delete;

        ///
        /// \brief destructor
        ///
        ~logger_t();

        ///
        /// \brief log tokens.
        ///
        template <typename T>
        const logger_t& operator<<(const T& data) const
        {
            m_stream << data;
            return *this;
        }

        ///
        /// \brief log manipulators.
        ///
        const logger_t& operator<<(std::ostream& (*pf)(std::ostream&)) const
        {
            pf(m_stream);
            return *this;
        }

    private:

        // attributes
        std::ostream&   m_stream;       ///< stream to write into
        std::streamsize m_precision;    ///< original precision to restore
    };

    ///
    /// \brief specific [information, warning, error] line loggers.
    ///
    inline logger_t log_info(std::ostream* cout = &std::cout, std::ostream* cerr = &std::cerr)
    {
        return logger_t(logger_t::type::info, cout, cerr);
    }

    inline logger_t log_warning(std::ostream* cout = &std::cout, std::ostream* cerr = &std::cerr)
    {
        return logger_t(logger_t::type::warn, cout, cerr);
    }

    inline logger_t log_error(std::ostream* cout = &std::cout, std::ostream* cerr = &std::cerr)
    {
        return logger_t(logger_t::type::error, cout, cerr);
    }

    ///
    /// \brief throws an exception as a critical condition is satisfied.
    ///
    template <typename... tmessage>
    [[noreturn]] void critical0(const tmessage&... message)
    {
        (log_error() << ... << message);
        throw std::runtime_error("critical check failed!");
    }

    ///
    /// \brief checks and throws an exception if the given condition is satisfied.
    ///
    template <typename tcondition, typename... tmessage>
    void critical(const tcondition& condition, const tmessage&... message)
    {
        if (static_cast<bool>(condition))
        {
            critical0(message...);
        }
    }

    ///
    /// \brief wraps main function to catch and log all exceptions.
    ///
    template <typename toperator>
    int main(const toperator& op, int argc, const char* argv[])
    {
        try
        {
            return op(argc, argv);
        }
        catch (std::exception& e)
        {
            log_error() << "caught exception (" << e.what() << ")!";
            return EXIT_FAILURE;
        }
        catch (...)
        {
            log_error() << "caught unknown exception!";
            return EXIT_FAILURE;
        }
    }
}
