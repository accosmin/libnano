#pragma once

#include <string>

namespace nano
{
    ///
    /// \brief iterator for splitting strings by delimiting characters.
    ///
    class tokenizer_t
    {
    public:

        ///
        /// \brief constructor
        ///
        tokenizer_t(const std::string& str, const char* delims, const size_t pos = 0) :
            m_str(str), m_delims(delims),
            m_pos(pos), m_end(pos)
        {
            next();
        }

        ///
        /// \brief enable copying
        ///
        tokenizer_t(const tokenizer_t&) = default;
        tokenizer_t& operator=(const tokenizer_t&) = delete;

        ///
        /// \brief enable moving
        ///
        tokenizer_t(tokenizer_t&&) noexcept = default;
        tokenizer_t& operator=(tokenizer_t&&) noexcept = delete;

        ///
        /// \brief default destructor
        ///
        ~tokenizer_t() = default;

        ///
        /// \brief returns true if parsing didn't finished
        ///
        operator bool() const // NOLINT(hicpp-explicit-conversions)
        {
            return (m_pos != std::string::npos) && (m_pos < m_end);
        }

        ///
        /// \brief move to the next token
        ///
        tokenizer_t& operator++()
        {
            next();
            return *this;
        }

        ///
        /// \brief move to the next token
        ///
        const tokenizer_t operator++(int) // NOLINT(readability-const-return-type)
        {
            tokenizer_t tmp(*this);
            next();
            return tmp;
        }

        ///
        /// \brief returns the current token
        ///
        auto get() const
        {
            // todo: return a std::string_view when moving to C++17
            return m_str.substr(m_pos, m_end - m_pos);
        }

        ///
        /// \brief returns the begining of the current token
        ///
        auto pos() const { return m_pos; }

        ///
        /// \brief returns the number of tokens found so far
        ///
        auto count() const { return m_count; }

    private:

        void next()
        {
            m_pos = m_str.find_first_not_of(m_delims, m_end);
            if ((m_pos == std::string::npos) ||
                ((m_end = m_str.find_first_of(m_delims, m_pos + 1)) == std::string::npos))
            {
                m_end = m_str.size();
            }

            if (this->operator bool())
            {
                ++ m_count;
            }
        }

        // attributes
        const std::string&  m_str;      ///< string to parse
        const char*         m_delims;   ///< delimiting characters
        size_t              m_pos{0};   ///< the begining of the current token
        size_t              m_end{0};   ///< the end of the current token
        size_t              m_count{0}; ///< the number of tokens found so far
    };
}
