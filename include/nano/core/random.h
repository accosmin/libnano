#pragma once

#include <random>
#include <cassert>
#include <type_traits>

namespace nano
{
    using rng_t = std::minstd_rand;

    template
    <
        typename tscalar,
        std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true
    >
    using udist_t = typename std::conditional<
        std::is_integral_v<tscalar>,
        std::uniform_int_distribution<tscalar>,
        std::uniform_real_distribution<tscalar>>::type;

    ///
    /// \brief create & initialize a random number generator.
    ///
    inline auto make_rng()
    {
        // todo: use seed_seq to initialize the RNG (see C++17)
        return rng_t{std::random_device{}()};
    }

    ///
    /// \brief create an uniform distribution for the [min, max] range.
    ///
    template
    <
        typename tscalar,
        std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true
    >
    inline auto make_udist(tscalar min, tscalar max)
    {
        assert(min <= max);
        return udist_t<tscalar>(min, max);
    }

    ///
    /// \brief generate a random value uniformaly distributed in the [min, max] range.
    ///
    template
    <
        typename tscalar, typename trng,
        std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true
    >
    tscalar urand(tscalar min, tscalar max, trng&& rng)
    {
        auto udist = make_udist<tscalar>(min, max);
        return udist(rng);
    }

    ///
    /// \brief fill the [begin, range) range of elements with random values uniformaly distributed in the [min, max] range.
    ///
    template
    <
        typename tscalar, typename titerator, typename trng,
        std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true
    >
    void urand(tscalar min, tscalar max, titerator begin, const titerator end, trng&& rng)
    {
        auto udist = make_udist<tscalar>(min, max);
        for ( ; begin != end; ++ begin)
        {
            *begin = udist(rng);
        }
    }
}
