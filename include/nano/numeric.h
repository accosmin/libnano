#pragma once

#include <cmath>
#include <limits>

namespace nano
{
    ///
    /// \brief square: x^2
    ///
    template <typename tscalar>
    tscalar square(const tscalar value)
    {
        return value * value;
    }

    ///
    /// \brief cube: x^3
    ///
    template <typename tscalar>
    tscalar cube(const tscalar value)
    {
        return value * square(value);
    }

    ///
    /// \brief quartic: x^4
    ///
    template <typename tscalar>
    tscalar quartic(const tscalar value)
    {
        return square(square(value));
    }

    ///
    /// \brief integer division with rounding
    ///
    template <typename tinteger, typename tinteger2>
    tinteger idiv(const tinteger nominator, const tinteger2 denominator)
    {
        return (nominator + static_cast<tinteger>(denominator) / 2) / static_cast<tinteger>(denominator);
    }

    ///
    /// \brief integer rounding
    ///
    template <typename tinteger, typename tinteger2>
    tinteger iround(const tinteger value, const tinteger2 modulo)
    {
        return idiv(value, modulo) * modulo;
    }

    ///
    /// \brief absolute value
    ///
    // fixme: this won't be needed in C++17 as std::abs is doing the right thing!
    template <typename tscalar>
    tscalar abs(const tscalar v)
    {
        return std::abs(v);
    }

    template <>
    inline float abs(const float v)
    {
        return std::fabs(v);
    }

    template <>
    inline double abs(const double v)
    {
        return std::fabs(v);
    }

    template <>
    inline long double abs(const long double v)
    {
        return std::fabs(v);
    }

    ///
    /// \brief check if two scalars are almost equal
    ///
    template <typename tscalar>
    bool close(const tscalar x, const tscalar y, const tscalar epsilon)
    {
        return nano::abs(x - y) <= (tscalar(1) + (nano::abs(x) + nano::abs(y) / 2)) * epsilon;
    }

    ///
    /// \brief round to the closest power of 10
    ///
    template <typename tscalar>
    inline auto roundpow10(const tscalar v)
    {
        return std::pow(tscalar(10), std::floor(std::log10(v)));
    }

    ///
    /// \brief precision level [0=very precise, 1=quite precise, 2=precise, 3=loose] for different scalars
    ///
    template <typename tscalar>
    tscalar epsilon()
    {
        return std::numeric_limits<tscalar>::epsilon();
    }

    template <typename tscalar>
    tscalar epsilon0()
    {
        return roundpow10(10 * epsilon<tscalar>());
    }

    template <typename tscalar>
    tscalar epsilon1()
    {
        const auto cb = std::cbrt(epsilon<tscalar>());
        return roundpow10(cb * cb);
    }

    template <typename tscalar>
    tscalar epsilon2()
    {
        return roundpow10(std::sqrt(epsilon<tscalar>()));
    }

    template <typename tscalar>
    tscalar epsilon3()
    {
        return roundpow10(std::cbrt(epsilon<tscalar>()));
    }
}
