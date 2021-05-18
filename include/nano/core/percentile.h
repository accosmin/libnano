#pragma once

#include <cmath>
#include <cassert>
#include <iterator>
#include <algorithm>

namespace nano
{
    ///
    /// \brief returns the percentile value.
    ///
    template <typename titerator>
    auto percentile(titerator begin, titerator end, double percentage)
    {
        assert(percentage >= 0.0 && percentage <= 100.0);

        const auto from_position = [&] (auto pos)
        {
            auto middle = begin;
            std::advance(middle, pos);
            std::nth_element(begin, middle, end);
            return static_cast<double>(*middle);
        };

        const auto size = std::distance(begin, end);
        const double position = percentage * static_cast<double>(size - 1) / 100.0;

        const auto lpos = static_cast<decltype(size)>(std::floor(position));
        const auto rpos = static_cast<decltype(size)>(std::ceil(position));

        if (lpos == rpos)
        {
            return from_position(lpos);
        }
        else
        {
            const auto lvalue = from_position(lpos);
            const auto rvalue = from_position(rpos);
            return (lvalue + rvalue) / 2;
        }
    }

    ///
    /// \brief returns the median value.
    ///
    template <typename titerator>
    auto median(titerator begin, titerator end)
    {
        return percentile(begin, end, 50);
    }
}
