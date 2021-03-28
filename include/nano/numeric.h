#pragma once

#include <cmath>
#include <limits>

namespace nano
{
    ///
    /// \brief square: x^2
    ///
    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_arithmetic<tscalar>::value>::type
    >
    tscalar square(tscalar value)
    {
        return value * value;
    }

    ///
    /// \brief cube: x^3
    ///
    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_arithmetic<tscalar>::value>::type
    >
    tscalar cube(tscalar value)
    {
        return value * square(value);
    }

    ///
    /// \brief quartic: x^4
    ///
    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_arithmetic<tscalar>::value>::type
    >
    tscalar quartic(tscalar value)
    {
        return square(square(value));
    }

    ///
    /// \brief integer division with rounding.
    ///
    template
    <
        typename tnominator, typename tdenominator,
        typename = typename std::enable_if<std::is_integral<tnominator>::value>::type,
        typename = typename std::enable_if<std::is_integral<tdenominator>::value>::type
    >
    tnominator idiv(tnominator nominator, tdenominator denominator)
    {
        return (nominator + static_cast<tnominator>(denominator) / 2) / static_cast<tnominator>(denominator);
    }

    ///
    /// \brief integer rounding.
    ///
    template
    <
        typename tvalue, typename tmodulo,
        typename = typename std::enable_if<std::is_integral<tvalue>::value>::type,
        typename = typename std::enable_if<std::is_integral<tmodulo>::value>::type
    >
    tvalue iround(tvalue value, tmodulo modulo)
    {
        return idiv(value, modulo) * modulo;
    }

    ///
    /// \brief check if two scalars are almost equal
    ///
    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_arithmetic<tscalar>::value>::type
    >
    bool close(tscalar x, tscalar y, tscalar epsilon)
    {
        return std::abs(x - y) <= (tscalar(1) + (std::abs(x) + std::abs(y) / 2)) * epsilon;
    }

    ///
    /// \brief round to the closest power of 10
    ///
    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_floating_point<tscalar>::value>::type
    >
    inline auto roundpow10(tscalar v)
    {
        return std::pow(tscalar(10), std::floor(std::log10(v)));
    }

    ///
    /// \brief precision level [0=very precise, 1=quite precise, 2=precise, 3=loose] for different scalars
    ///
    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_floating_point<tscalar>::value>::type
    >
    tscalar epsilon()
    {
        return std::numeric_limits<tscalar>::epsilon();
    }

    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_floating_point<tscalar>::value>::type
    >
    tscalar epsilon0()
    {
        return roundpow10(10 * epsilon<tscalar>());
    }

    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_floating_point<tscalar>::value>::type
    >
    tscalar epsilon1()
    {
        const auto cb = std::cbrt(epsilon<tscalar>());
        return roundpow10(cb * cb);
    }

    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_floating_point<tscalar>::value>::type
    >
    tscalar epsilon2()
    {
        return roundpow10(std::sqrt(epsilon<tscalar>()));
    }

    template
    <
        typename tscalar,
        typename = typename std::enable_if<std::is_floating_point<tscalar>::value>::type
    >
    tscalar epsilon3()
    {
        return roundpow10(std::cbrt(epsilon<tscalar>()));
    }
}
