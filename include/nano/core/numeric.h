#pragma once

#include <cmath>
#include <limits>

namespace nano
{
    ///
    /// \brief square: x^2.
    ///
    template
    <
        typename tscalar,
        std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true
    >
    tscalar square(tscalar value)
    {
        return value * value;
    }

    ///
    /// \brief cube: x^3.
    ///
    template
    <
        typename tscalar,
        std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true
    >
    tscalar cube(tscalar value)
    {
        return value * square(value);
    }

    ///
    /// \brief quartic: x^4.
    ///
    template
    <
        typename tscalar,
        std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true
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
        std::enable_if_t<std::is_integral_v<tnominator>, bool> = true,
        std::enable_if_t<std::is_integral_v<tdenominator>, bool> = true
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
        std::enable_if_t<std::is_integral_v<tvalue>, bool> = true,
        std::enable_if_t<std::is_integral_v<tmodulo>, bool> = true
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
        std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true
    >
    bool close(tscalar x, tscalar y, tscalar epsilon)
    {
        return std::abs(x - y) <= (tscalar(1) + (std::abs(x) + std::abs(y) / 2)) * epsilon;
    }

    ///
    /// \brief round to the closest power of 10.
    ///
    template
    <
        typename tscalar,
        std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true
    >
    inline auto roundpow10(tscalar v)
    {
        return std::pow(tscalar(10), std::round(std::log10(v)));
    }

    ///
    /// \brief precision level [0=very precise, 1=quite precise, 2=precise, 3=loose] for different scalars.
    ///
    template
    <
        typename tscalar,
        std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true
    >
    tscalar epsilon()
    {
        return std::numeric_limits<tscalar>::epsilon();
    }

    template
    <
        typename tscalar,
        std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true
    >
    tscalar epsilon0()
    {
        return roundpow10(10 * epsilon<tscalar>());
    }

    template
    <
        typename tscalar,
        std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true
    >
    tscalar epsilon1()
    {
        const auto cb = std::cbrt(epsilon<tscalar>());
        return roundpow10(cb * cb);
    }

    template
    <
        typename tscalar,
        std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true
    >
    tscalar epsilon2()
    {
        return roundpow10(std::sqrt(epsilon<tscalar>()));
    }

    template
    <
        typename tscalar,
        std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true
    >
    tscalar epsilon3()
    {
        return roundpow10(std::cbrt(epsilon<tscalar>()));
    }
}
