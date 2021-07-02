#pragma once

#include <regex>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <typeinfo>
#include <algorithm>
#include <stdexcept>

namespace nano
{
    using string_t = std::string;
    using strings_t = std::vector<string_t>;

    ///
    /// \brief text alignment options
    ///
    enum class alignment : int
    {
        left,
        center,
        right
    };

    template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
    using enum_map_t = std::vector<std::pair<tenum, const char*>>;

    ///
    /// \brief maps all possible values of an enum to string.
    ///
    template <typename tenum>
    enum_map_t<tenum> enum_string();

    ///
    /// \brief collect all the values of an enum type, optionally filtered by the given regular expression.
    ///
    template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
    auto enum_values(const std::regex& enum_regex = std::regex(".+"))
    {
        std::vector<tenum> enums;
        for (const auto& elem : enum_string<tenum>())
        {
            if (std::regex_match(elem.second, enum_regex))
            {
                enums.push_back(elem.first);
            }
        }
        return enums;
    }

    ///
    /// \brief concatenate a list of potentially heterogeneous values into a formatted string.
    ///
    namespace detail
    {
        template <typename tvalue>
        void scat(std::ostringstream& stream, const tvalue& value)
        {
            if constexpr (std::is_enum_v<std::remove_reference_t<tvalue>>)
            {
                for (const auto& elem : enum_string<tvalue>())
                {
                    if (elem.first == value)
                    {   // cppcheck-suppress useStlAlgorithm
                        stream << elem.second;
                        return;
                    }
                }

                const auto str = std::to_string(static_cast<int>(value));
                const auto msg = string_t("missing mapping for enumeration ") + typeid(tvalue).name() + " <" + str + ">!";
                throw std::invalid_argument(msg);
            }
            else
            {
                stream << value;
            }
        }

        template <typename tvalue>
        void scat(std::ostringstream& stream, const std::vector<tvalue>& values)
        {
            for (auto begin = values.begin(), end = values.end(); begin != end; )
            {
                scat(stream, *begin);
                if (++ begin != end)
                {
                    stream << ",";
                }
            }
        }

        template <typename... tvalues>
        void scat(std::ostringstream& stream, const tvalues&... values)
        {
            (scat(stream, values), ...);
        }
    }

    template <typename... tvalues>
    string_t scat(const tvalues&... values)
    {
        std::ostringstream stream;
        detail::scat(stream, values...);
        return stream.str();
    }

    ///
    /// \brief cast string to value.
    /// FIXME: use std::string_view instead of std::string (more efficient when used with tokenizer) when
    ///        gcc (and clang) supports properly std::from_chars.
    ///
    template <typename tvalue>
    tvalue from_string(const string_t& str)
    {
        if constexpr (std::is_integral_v<tvalue>)
        {
            if constexpr (std::is_signed_v<tvalue>)
            {
                return static_cast<tvalue>(std::stoll(str));
            }
            else
            {
                return static_cast<tvalue>(std::stoull(str));
            }
        }
        else if constexpr (std::is_floating_point_v<tvalue>)
        {
            return static_cast<tvalue>(std::stold(str));
        }
        else if constexpr (std::is_same_v<tvalue, string_t>)
        {
            return str;
        }
        else if constexpr (std::is_enum_v<std::remove_reference_t<tvalue>>)
        {
            const auto options = enum_string<tvalue>();
            for (const auto& option : options)
            {
                if (option.second == str)
                {   // cppcheck-suppress useStlAlgorithm
                    return option.first;
                }
            }
            for (const auto& option : options)
            {
                if (str.find(option.second) == 0)
                {
                    return option.first;
                }
            }
            throw std::invalid_argument(scat("invalid ", typeid(tvalue).name(), " <", ">!"));
        }
    }

    ///
    /// \brief cast string to value and use the given default value if casting fails.
    ///
    template <typename tvalue>
    tvalue from_string(const string_t& str, const tvalue& default_value)
    {
        try
        {
            return from_string<tvalue>(str);
        }
        catch (std::exception&)
        {
            return default_value;
        }
    }

    ///
    /// \brief check if a string starts with a token (case sensitive)
    ///
    inline bool starts_with(const string_t& str, const string_t& token)
    {
        return str.size() >= token.size() && std::equal(token.begin(), token.end(), str.begin());
    }

    ///
    /// \brief check if a string ends with a token (case sensitive)
    ///
    inline bool ends_with(const string_t& str, const string_t& token)
    {
        return str.size() >= token.size() && std::equal(token.rbegin(), token.rend(), str.rbegin());
    }

    ///
    /// \brief align a string to fill the given size (if possible).
    ///
    inline string_t align(const string_t& str, const size_t str_size,
        const alignment mode = alignment::left, const char fill_char = ' ')
    {
        const auto fill_size = (str.size() > str_size) ? (0) : (str_size - str.size());

        string_t ret;
        switch (mode)
        {
        case alignment::center:
            ret.append(fill_size / 2, fill_char);
            ret.append(str);
            ret.append(fill_size - fill_size / 2, fill_char);
            break;

        case alignment::right:
            ret.append(fill_size, fill_char);
            ret.append(str);
            break;

        case alignment::left:
        default:
            ret.append(str);
            ret.append(fill_size, fill_char);
            break;
        }

        return ret;
    }
}
