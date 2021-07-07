#pragma once

#include <chrono>
#include <string>
#include <cstdio>
#include <nano/arch.h>
#include <nano/core/stats.h>
#include <nano/core/numeric.h>

namespace nano
{
    using picoseconds_t = std::chrono::duration<long long, std::pico>;
    using nanoseconds_t = std::chrono::duration<long long, std::nano>;
    using microseconds_t = std::chrono::duration<long long, std::micro>;
    using milliseconds_t = std::chrono::duration<long long, std::milli>;
    using seconds_t = std::chrono::duration<long long>;

    using timepoint_t = std::chrono::high_resolution_clock::time_point;

    ///
    /// \brief return a human readable string representation of a duration in milliseconds.
    ///
    NANO_PUBLIC std::string elapsed(int milliseconds);

    ///
    /// \brief utility to measure duration.
    ///
    class timer_t
    {
    public:
        ///
        /// \brief constructor
        ///
        timer_t() : m_start(now())
        {
        }

        ///
        /// \brief reset the current time point.
        ///
        void reset()
        {
            m_start = now();
        }

        ///
        /// \brief retrieve the elapsed time as a string.
        ///
        std::string elapsed() const
        {
            return ::nano::elapsed(static_cast<int>(this->milliseconds().count()));
        }

        ///
        /// \brief retrieve the elapsed time in seconds.
        ///
        seconds_t seconds() const
        {
            return duration<seconds_t>();
        }

        ///
        /// \brief retrieve the elapsed time in milliseconds.
        ///
        milliseconds_t milliseconds() const
        {
            return duration<milliseconds_t>();
        }

        ///
        /// \brief retrieve the elapsed time in microseconds.
        ///
        microseconds_t microseconds() const
        {
            return duration<microseconds_t>();
        }

        ///
        /// \brief retrieve the elapsed time in nanoseconds.
        ///
        nanoseconds_t nanoseconds() const
        {
            return duration<nanoseconds_t>();
        }

    private:

        static timepoint_t now()
        {
            return std::chrono::high_resolution_clock::now();
        }

        template <typename tduration>
        tduration duration() const
        {
            return std::chrono::duration_cast<tduration>(now() - m_start);
        }

        // attributes
        timepoint_t     m_start;        ///< starting time point
    };

    ///
    /// \brief robustly measure a function call (in the given time units).
    ///
    template <typename tunits, typename toperator>
    tunits measure(const toperator& op, int64_t trials, int64_t min_trial_iterations = 1,
        microseconds_t min_trial_duration = microseconds_t(1000))
    {
        const auto run_opx = [&] (int64_t times)
        {
            const timer_t timer;
            for (int64_t i = 0; i < times; ++ i)
            {
                op();
            }
            return timer.microseconds();
        };

        const auto run_trial = [&] (int64_t times)
        {
            return picoseconds_t(nano::idiv(run_opx(times).count() * 1000 * 1000, times));
        };

        // calibrate the number of function calls to achieve the minimum time resolution
        auto trial_iterations = std::max(int64_t(1), min_trial_iterations);
        const auto max_trials = std::numeric_limits<int64_t>::max() / 4;
        for (microseconds_t usecs(0); usecs < min_trial_duration && trial_iterations < max_trials; trial_iterations *= 2)
        {
            usecs = run_opx(trial_iterations);
        }

        // measure multiple times for robustness
        picoseconds_t duration = run_trial(trial_iterations);
        for (int64_t t = 1; t < trials; ++ t)
        {
            duration = std::min(duration, run_trial(trial_iterations));
        }

        return std::chrono::duration_cast<tunits>(duration);
    }

    ///
    /// \brief compute GFLOPS (giga floating point operations per seconds)
    ///     given the number of FLOPs run in the given duration.
    ///
    template <typename tinteger, typename tduration>
    int64_t gflops(const tinteger flops, const tduration& duration)
    {
        const auto div = static_cast<int64_t>(std::chrono::duration_cast<picoseconds_t>(duration).count());
        return nano::idiv(static_cast<int64_t>(flops) * 1000, std::max(div, int64_t(1)));
    }

    ///
    /// \brief accumulate time measurements for a given operation of given complexity (aka flops).
    ///
    class probe_t
    {
    public:

        explicit probe_t(std::string basename = std::string(),
                std::string fullname = std::string(),
                const int64_t flops = 1) :
                m_basename(std::move(basename)),
                m_fullname(std::move(fullname)),
                m_flops(flops)
        {
        }

        template <typename toperator>
        void measure(const toperator& op, const int64_t count = 1)
        {
            assert(count > 0);
            const timer_t timer;
            op();
            m_timings(timer.nanoseconds().count() / count);
        }

        operator bool() const { return m_timings; } // NOLINT(hicpp-explicit-conversions)
        const auto& timings() const { return m_timings; }

        const auto& basename() const { return m_basename; }
        const auto& fullname() const { return m_fullname; }

        auto flops() const { return m_flops; }
        auto kflops() const { return m_flops / 1024; }
        auto gflops() const { return nano::gflops(flops(), nanoseconds_t(static_cast<int64_t>(timings().min()))); }

    private:

        // attributes
        std::string     m_basename;             ///<
        std::string     m_fullname;             ///<
        int64_t         m_flops;                ///< number of floating point operations per call
        stats_t         m_timings;              ///< time measurements
    };

    using probes_t = std::vector<probe_t>;
}
