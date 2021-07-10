#pragma once

#include <cmath>
#include <vector>
#include <cassert>
#include <numeric>
#include <ostream>
#include <iterator>
#include <algorithm>

namespace nano
{
    namespace detail
    {
        template <typename titerator, typename toperator>
        auto percentile(titerator begin, titerator end, double percentage, const toperator& from_position)
        {
            assert(percentage >= 0.0 && percentage <= 100.0);

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
    }

    ///
    /// \brief returns the percentile value from a potentially not sorted list of values.
    ///
    template <typename titerator>
    auto percentile(titerator begin, titerator end, double percentage)
    {
        const auto from_position = [&] (auto pos)
        {
            auto middle = begin;
            std::advance(middle, pos);
            std::nth_element(begin, middle, end);
            return static_cast<double>(*middle);
        };

        return detail::percentile(begin, end, percentage, from_position);
    }

    ///
    /// \brief returns the percentile value from a sorted list of values.
    ///
    template <typename titerator>
    auto percentile_sorted(titerator begin, titerator end, double percentage)
    {
        assert(std::is_sorted(begin, end));

        const auto from_position = [&] (auto pos)
        {
            auto middle = begin;
            std::advance(middle, pos);
            return static_cast<double>(*middle);
        };

        return detail::percentile(begin, end, percentage, from_position);
    }

    ///
    /// \brief returns the median value from a potentially not sorted list of values.
    ///
    template <typename titerator>
    auto median(titerator begin, titerator end)
    {
        return percentile(begin, end, 50);
    }

    ///
    /// \brief returns the median value from a sorted list of values.
    ///
    template <typename titerator>
    auto median_sorted(titerator begin, titerator end)
    {
        return percentile_sorted(begin, end, 50);
    }

    ///
    /// \brief collects numerical values and computes statistics like:
    ///     - average, standard deviation
    ///     - minimum and maximum values
    ///     - median, percentile etc.
    ///
    class stats_t
    {
    public:

        using value_t = double;
        using values_t = std::vector<value_t>;

        ///
        /// \brief constructor
        ///
        stats_t() = default;

        ///
        /// \brief update statistics with the given [begin, end) range
        ///
        template <typename titerator, typename = typename std::iterator_traits<titerator>::value_type>
        stats_t(titerator begin, const titerator end) // NOLINT(readability-non-const-parameter)
        {
            for ( ; begin != end; ++ begin)
            {
                operator()(*begin);
            }
        }

        ///
        /// \brief update statistics with new values
        ///
        template <typename tscalar>
        void operator()(tscalar value)
        {
            m_values.push_back(static_cast<value_t>(value));
        }

        template <typename... tscalars>
        void operator()(tscalars... values)
        {
            (operator()(values), ...);
        }

        ///
        /// \brief merge statistics
        ///
        void operator()(const stats_t& other)
        {
            m_values.insert(m_values.end(), other.m_values.begin(), other.m_values.end());
        }

        ///
        /// \brief update statistics with the given [begin, end) range
        ///
        template <typename titerator, typename = typename std::iterator_traits<titerator>::value_type>
        void operator()(titerator begin, const titerator end)
        {
            for ( ; begin != end; ++ begin)
            {
                operator()(*begin);
            }
        }

        ///
        /// \brief reset statistics
        ///
        void clear()
        {
            m_values.clear();
        }

        ///
        /// \brief returns the number of values
        ///
        auto count() const
        {
            return m_values.size();
        }

        ///
        /// \brief returns the minimum
        ///
        auto min() const
        {
            assert(!m_values.empty());
            return *std::min_element(m_values.begin(), m_values.end());
        }

        ///
        /// \brief returns the maximum
        ///
        auto max() const
        {
            assert(!m_values.empty());
            return *std::max_element(m_values.begin(), m_values.end());
        }

        ///
        /// \brief returns the sum
        ///
        auto sum1() const
        {
            return  std::accumulate(m_values.begin(), m_values.end(), value_t(0),
                    [] (const auto s, const auto v) { return s + v; });
        }

        ///
        /// \brief returns the sum of squares
        ///
        auto sum2() const
        {
            return  std::accumulate(m_values.begin(), m_values.end(), value_t(0),
                    [] (const auto s, const auto v) { return s + v * v; });
        }

        ///
        /// \brief returns the average
        ///
        auto avg() const
        {
            assert(count() > 0);
            return sum1() / static_cast<value_t>(count());
        }

        ///
        /// \brief returns the variance.
        ///
        auto var() const
        {
            assert(count() > 0);
            return sum2() / static_cast<value_t>(count()) - avg() * avg();
        }

        ///
        /// \brief returns the population standard deviation.
        ///
        auto stdev() const
        {
            return std::sqrt(var());
        }

        ///
        /// \brief returns the percentile.
        ///
        auto percentile(int percentage)
        {
            return ::nano::percentile(m_values.begin(), m_values.end(), percentage);
        }

        ///
        /// \brief returns the median.
        ///
        auto median()
        {
            return percentile(50);
        }

        ///
        /// \brief check if valid (enough values collected)
        ///
        operator bool() const { return count() > 1; } // NOLINT(hicpp-explicit-conversions)

    private:

        // attributes
        values_t    m_values;       ///< store all values
    };

    inline std::ostream& operator<<(std::ostream& os, const stats_t& stats)
    {
        return  !stats ?
                os :
                (os << stats.avg() << "+/-" << stats.stdev() << "[" << stats.min() << "," << stats.max() << "]");
    }
}
