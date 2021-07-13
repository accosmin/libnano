#include <nano/core/chrono.h>

using namespace nano;

static void append(std::string& str, const char* format, const int value)
{
    char buffer[32];
    snprintf(buffer, sizeof(buffer), format, value); // NOLINT(hicpp-vararg,cppcoreguidelines-pro-type-vararg)
    str.append(buffer);
}

std::string nano::elapsed(int milliseconds)
{
    static constexpr int size_second = 1000;
    static constexpr int size_minute = 60 * size_second;
    static constexpr int size_hour = 60 * size_minute;
    static constexpr int size_day = 24 * size_hour;

    const auto days = milliseconds / size_day; milliseconds -= days * size_day;
    const auto hours = milliseconds / size_hour; milliseconds -= hours * size_hour;
    const auto minutes = milliseconds / size_minute; milliseconds -= minutes * size_minute;
    const auto seconds = milliseconds / size_second; milliseconds -= seconds * size_second;

    std::string str;
    if (days > 0)
    {
        append(str, "%id:", days);
    }
    if (days > 0 || hours > 0)
    {
        append(str, "%.2ih:", hours);
    }
    if (days > 0 || hours > 0 || minutes > 0)
    {
        append(str, "%.2im:", minutes);
    }
    if (days > 0 || hours > 0 || minutes > 0 || seconds > 0)
    {
        append(str, "%.2is:", seconds);
    }
    append(str, "%.3ims", milliseconds);

    return str;
} // LCOV_EXCL_LINE
