#include <cassert>
#include <fstream>
#include <nano/core/cmdline.h>
#include <nano/core/tokenizer.h>

using namespace nano;

static const auto str_dash = string_t{"-"}; // NOLINT(cert-err58-cpp)
static const auto str_dash_dash = string_t{"--"}; // NOLINT(cert-err58-cpp)

string_t cmdline_t::option_t::describe() const
{
    string_t str;
    if (!m_short_name.empty())
    {
        str += "-";
        str += m_short_name;
        str += ",";
    }
    str += "--";
    str += m_name;
    if (!m_value.empty())
    {
        str += "(";
        str += m_value;
        str += ")";
    }

    return str;
} // LCOV_EXCL_LINE

cmdline_t::cmdline_t(string_t title) :
    m_title(std::move(title))
{
    add("h", "help", "usage");
}

void cmdline_t::add(string_t short_name, string_t name, string_t description)
{
    add(std::move(short_name), std::move(name), std::move(description), string_t());
}

void cmdline_t::add(string_t short_name, string_t name, string_t description, string_t default_value)
{
    if (name.empty() || nano::starts_with(name, str_dash_dash) || nano::starts_with(name, str_dash))
    {
        throw std::runtime_error(scat("cmdline: invalid option name [", name, "]"));
    }
    if (find(name) != m_options.end())
    {
        throw std::runtime_error(scat("cmdline: duplicated option [", name, "]"));
    }

    if (!short_name.empty())
    {
        if (short_name.size() != 1 || short_name[0] == '-')
        {
            throw std::runtime_error(scat("cmdline: invalid short option name [", short_name, "]"));
        }
        if (find(short_name) != m_options.end())
        {
            throw std::runtime_error(scat("cmdline: duplicated short option [", short_name, "]"));
        }
    }

    m_options.emplace_back(std::move(short_name), std::move(name), std::move(description), std::move(default_value));
}

void cmdline_t::store(const string_t& name_or_short_name, const string_t& value)
{
    auto it = find(name_or_short_name);
    if (it == m_options.end())
    {
        throw std::runtime_error(scat("cmdline: unrecognized option [", name_or_short_name, "]"));
    }
    else
    {
        it->m_given = true;
        if (!value.empty())
        {
            it->m_value = value;
        }
    }
}

void cmdline_t::process(const int argc, const char* argv[])
{
    string_t current_name_or_short_name;

    for (int i = 1; i < argc; ++ i)
    {
        const string_t token = argv[i];
        assert(!token.empty());

        if (nano::starts_with(token, str_dash_dash))
        {
            if (token.size() == 2U)
            {
                throw std::runtime_error(scat("cmdline: invalid option name [", token, "]"));
            }

            auto name = token.substr(2);
            store(name);
            current_name_or_short_name = std::move(name);
        }
        else if (nano::starts_with(token, str_dash))
        {
            if (token.size() != 2U)
            {
                throw std::runtime_error(scat("cmdline: invalid short option name [", token, "]"));
            }

            auto short_name = token.substr(1);
            store(short_name);
            current_name_or_short_name = std::move(short_name);
        }
        else
        {
            const auto& value = token;
            if (current_name_or_short_name.empty())
            {
                throw std::runtime_error(scat("cmdline: missing option before value [", value, "]"));
            }

            store(current_name_or_short_name, value);
            current_name_or_short_name.clear();
        }
    }
}

void cmdline_t::process(const string_t& config)
{
    strings_t tokens;
    for (auto tokenizer = tokenizer_t{config, " \t\n\r"}; tokenizer; ++ tokenizer)
    {
        tokens.push_back(tokenizer.get());
    }

    std::vector<const char*> ptokens(tokens.size() + 1, nullptr);
    std::transform(tokens.begin(), tokens.end(), ptokens.begin() + 1, [&] (const auto& token) { return token.c_str(); });

    process(static_cast<int>(ptokens.size()), ptokens.data());
}

void cmdline_t::process_config_file(const string_t& path)
{
    std::ifstream in(path.c_str());
    const string_t config((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    process(config);
}

bool cmdline_t::has(const string_t& name_or_short_name) const
{
    const auto it = find(name_or_short_name);
    if (it == m_options.end())
    {
        throw std::runtime_error(scat("cmdline: unrecognized option [", name_or_short_name, "]"));
    }
    return it->m_given;
}

string_t cmdline_t::get(const string_t& name_or_short_name) const
{
    const auto it = find(name_or_short_name);
    if (it == m_options.end())
    {
        throw std::runtime_error(scat("cmdline: unrecognized option [", name_or_short_name, "]"));
    }
    else if (!it->m_given && it->m_value.empty())
    {
        throw std::runtime_error(scat("cmdline: no value provided for option [", name_or_short_name, "]"));
    }
    return it->get();
}

void cmdline_t::usage(std::ostream& os, size_t indent) const
{
    os << m_title << std::endl;

    size_t max_option_size = 0;
    for (const auto& option : m_options)
    {
        max_option_size = std::max(max_option_size, option.describe().size());
    }

    max_option_size += 4;
    for (const auto& option : m_options)
    {
        os << string_t(indent, ' ')
           << nano::align(option.describe(), max_option_size) << option.m_description << std::endl;
    }

    os << std::endl;
}
