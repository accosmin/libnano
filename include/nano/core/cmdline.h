#pragma once

#include <iostream>
#include <nano/arch.h>
#include <nano/string.h>

namespace nano
{
    ///
    /// \brief command line processing of the form:
    ///     --option [value]
    ///     -o [value]s
    ///
    /// other properties:
    ///     - -h,--help is built-in
    ///     - any error is considered critical and reported as an exception
    ///             (e.g. duplicated option names, missing option value, invalid option name)
    ///     - each option must have a long name, while the short name (single character) is optional
    ///     - options need not have an associated value (they can be interpreted as boolean flags)
    ///
    class NANO_PUBLIC cmdline_t
    {
    public:

        ///
        /// \brief command line option.
        ///
        struct option_t
        {
            option_t() = default;

            option_t(string_t short_name, string_t name, string_t description, string_t default_value) :
                m_short_name(std::move(short_name)),
                m_name(std::move(name)),
                m_description(std::move(description)),
                m_value(std::move(default_value))
            {
            }

            string_t describe() const;
            auto has() const { return m_given; }
            const auto& get() const { return m_value; }

            // attributes
            string_t    m_short_name;       ///<
            string_t    m_name;             ///<
            string_t    m_description;      ///<
            string_t    m_value;            ///<
            bool        m_given{false};     ///<
        };

        ///
        /// \brief constructor
        ///
        explicit cmdline_t(string_t title);

        ///
        /// \brief add new option by name and short name (without dash).
        ///
        void add(string_t short_name, string_t name, string_t description);

        ///
        /// \brief add new option with default value by name and short name (without dash)
        ///
        template <typename tvalue>
        void add(string_t short_name, string_t name, string_t description, tvalue default_value)
        {
            add(std::move(short_name), std::move(name), std::move(description), scat(default_value));
        }

        ///
        /// \brief process the command line arguments
        ///
        void process(int argc, const char* argv[]);

        ///
        /// \brief process the command line arguments
        ///
        void process(const string_t& config);

        ///
        /// \brief process the command line arguments from configuration file
        ///
        void process_config_file(const string_t& path);

        ///
        /// \brief check if an option was set
        ///
        bool has(const string_t& name_or_short_name) const;

        ///
        /// \brief get the value of an option
        ///
        string_t get(const string_t& name_or_short_name) const;

        ///
        /// \brief get the value of an option as a given type
        ///
        template <typename tvalue>
        tvalue get(const string_t& name_or_short_name) const
        {
            return nano::from_string<tvalue>(get(name_or_short_name));
        }

        ///
        /// \brief print help menu
        ///
        void usage(std::ostream& = std::cout, size_t indent = 2) const;

    private:

        auto find(const string_t& name_or_short_name)
        {
            return std::find(m_options.begin(), m_options.end(), name_or_short_name);
        }

        auto find(const string_t& name_or_short_name) const
        {
            return std::find(m_options.begin(), m_options.end(), name_or_short_name);
        }

        void add(string_t short_name, string_t name, string_t description, string_t default_value);
        void store(const string_t& name_or_short_name, const string_t& value = string_t());

        // attributes
        using options_t = std::vector<option_t>;
        string_t        m_title;        ///<
        options_t       m_options;      ///<
    };

    inline bool operator==(const cmdline_t::option_t& option, const string_t& name_or_short_name)
    {
        return  option.m_short_name == name_or_short_name ||
                option.m_name == name_or_short_name;
    }
}
