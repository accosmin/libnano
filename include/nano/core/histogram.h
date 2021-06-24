#pragma once

#include <nano/tensor.h>
#include <nano/core/percentile.h>

namespace nano
{
    ///
    /// \brief construct equidistant percentiles (in the range [0, 100]).
    ///
    inline tensor_mem_t<scalar_t, 1> make_equidistant_percentiles(tensor_size_t bins)
    {
        assert(bins > 1);

        const auto delta = 100.0 / static_cast<scalar_t>(bins);

        tensor_mem_t<scalar_t, 1> percentiles(bins - 1);
        percentiles.lin_spaced(delta, 100.0 - delta);
        return percentiles;
    }

    ///
    /// \brief construct equidistant ratios (in the range [0, 1]).
    ///
    inline tensor_mem_t<scalar_t, 1> make_equidistant_ratios(tensor_size_t bins)
    {
        assert(bins > 1);

        const auto delta = 1.0 / static_cast<scalar_t>(bins);

        tensor_mem_t<scalar_t, 1> ratios(bins - 1);
        ratios.lin_spaced(delta, 1.0 - delta);
        return ratios;
    }

    ///
    /// \brief histogram utility for scalar values.
    ///
    /// the bins can be initialized:
    ///     - from equidistant or user-defined [0, 1] ratios of the [min, max] range of values or
    ///     - from equidistant or user-defined [0, 100] percentiles or
    ///     - from user-defined scalar thresholds in the [min, max] range of values.
    ///
    /// the following values are made available for each bin:
    ///     - the number of samples and
    ///     - the mean and the median of the values falling in the bin.
    ///
    class histogram_t
    {
    public:

        using thresholds_t = tensor_mem_t<scalar_t, 1>;
        using bin_counts_t = tensor_mem_t<tensor_size_t, 1>;
        using bin_means_t = tensor_mem_t<scalar_t, 1>;
        using bin_medians_t = tensor_mem_t<scalar_t, 1>;

        histogram_t() = default;

        template <typename titerator>
        histogram_t(titerator begin, titerator end, tensor_mem_t<scalar_t, 1> thresholds) :
            m_thresholds(std::move(thresholds))
        {
            assert(m_thresholds.size() > 0);

            std::sort(begin, end);
            std::sort(::nano::begin(m_thresholds), ::nano::end(m_thresholds));

            update(begin, end);
        }

        template <typename titerator>
        static histogram_t make_from_percentiles(titerator begin, titerator end, tensor_size_t bins)
        {
            return make_from_percentiles(begin, end, make_equidistant_percentiles(bins));
        }

        template <typename titerator>
        static histogram_t make_from_percentiles(titerator begin, titerator end, tensor_mem_t<scalar_t, 1> percentiles)
        {
            std::sort(begin, end);
            std::sort(::nano::begin(percentiles), ::nano::end(percentiles));

            assert(std::distance(begin, end) > 0);
            assert(percentiles.size() > 0);
            assert(percentiles(0) > 0.0);
            assert(percentiles(percentiles.size() - 1) < 100.0);

            tensor_mem_t<scalar_t, 1> thresholds(percentiles.size());
            for (tensor_size_t i = 0; i < thresholds.size(); ++ i)
            {
                thresholds(i) = percentile_sorted(begin, end, percentiles(i));
            }

            return histogram_t(begin, end, thresholds);
        }

        template <typename titerator>
        static histogram_t make_from_thresholds(titerator begin, titerator end, tensor_mem_t<scalar_t, 1> thresholds)
        {
            return histogram_t(begin, end, std::move(thresholds));
        }

        template <typename titerator>
        static histogram_t make_from_ratios(titerator begin, titerator end, tensor_size_t bins)
        {
            return make_from_ratios(begin, end, make_equidistant_ratios(bins));
        }

        template <typename titerator>
        static histogram_t make_from_ratios(titerator begin, titerator end, tensor_mem_t<scalar_t, 1> ratios)
        {
            std::sort(begin, end);
            std::sort(::nano::begin(ratios), ::nano::end(ratios));

            assert(std::distance(begin, end) > 0);
            assert(ratios.size() > 0);
            assert(ratios(0) > 0.0);
            assert(ratios(ratios.size() - 1) < 1.0);

            auto min = static_cast<scalar_t>(*begin);
            auto max = static_cast<scalar_t>(*--end); ++ end;
            if (max < min + std::numeric_limits<scalar_t>::epsilon())
            {
                min -= std::numeric_limits<scalar_t>::epsilon();
                max += std::numeric_limits<scalar_t>::epsilon();
            }
            const auto delta = (max - min);

            tensor_mem_t<scalar_t, 1> thresholds(ratios.size());
            for (tensor_size_t i = 0; i < thresholds.size(); ++ i)
            {
                thresholds(i) = min + ratios(i) * delta;
            }

            return histogram_t(begin, end, thresholds);
        }

        template <typename titerator>
        static histogram_t make_from_exponents(titerator begin, titerator end, scalar_t base,
            scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon())
        {
            std::sort(begin, end);

            assert(std::distance(begin, end) > 0);
            assert(base > 1.0);
            assert(epsilon > 0.0);

            const auto get_exponent = [=] (scalar_t value)
            {
                const auto log_value = std::log(std::fabs(value)) / std::log(base);
                return static_cast<int>(std::floor(log_value));
            };

            int min_pos_exponent = std::numeric_limits<int>::max(), max_pos_exponent = std::numeric_limits<int>::min();
            int min_neg_exponent = std::numeric_limits<int>::max(), max_neg_exponent = std::numeric_limits<int>::min();

            for (auto it = begin; it != end; ++ it)
            {
                const auto value = static_cast<scalar_t>(*it);
                if (value < 0.0)
                {
                    const auto exponent = get_exponent(std::min(value, -epsilon));
                    min_neg_exponent = std::min(min_neg_exponent, exponent);
                    max_neg_exponent = std::max(max_neg_exponent, exponent);
                }
                else
                {
                    const auto exponent = get_exponent(std::max(value, +epsilon));
                    min_pos_exponent = std::min(min_pos_exponent, exponent);
                    max_pos_exponent = std::max(max_pos_exponent, exponent);
                }
            }

            const auto has_pos = min_pos_exponent != std::numeric_limits<int>::max();
            const auto has_neg = min_neg_exponent != std::numeric_limits<int>::max();

            tensor_mem_t<scalar_t, 1> thresholds(static_cast<tensor_size_t>(
                (has_pos ? (max_pos_exponent - min_pos_exponent + 1) : 0) +
                (has_neg ? (max_neg_exponent - min_neg_exponent + 1) : 0)));

            tensor_size_t i = 0;
            for (auto exponent = max_neg_exponent; has_neg && exponent != min_neg_exponent - 1; -- exponent, ++ i)
            {
                thresholds(i) = -std::pow(base, static_cast<scalar_t>(exponent));
            }
            for (auto exponent = min_pos_exponent; has_pos && exponent != max_pos_exponent + 1; ++ exponent, ++ i)
            {
                thresholds(i) = +std::pow(base, static_cast<scalar_t>(exponent));
            }

            return histogram_t(begin, end, thresholds);
        }

        const auto& means() const { return m_bin_means; }
        const auto& counts() const { return m_bin_counts; }
        const auto& medians() const { return m_bin_medians; }
        const auto& thresholds() const { return m_thresholds; }

        tensor_size_t bins() const { return m_bin_counts.size(); }
        scalar_t mean(tensor_size_t bin) const { return m_bin_means(bin); }
        scalar_t median(tensor_size_t bin) const { return m_bin_medians(bin); }
        tensor_size_t count(tensor_size_t bin) const { return m_bin_counts(bin); }

        template <typename tvalue>
        tensor_size_t bin(tvalue value) const
        {
            const auto svalue = cast_bin(value);

            const auto *const begin = ::nano::begin(m_thresholds);
            const auto *const end = ::nano::end(m_thresholds);

            const auto *const it = std::upper_bound(begin, end, svalue);
            if (it == end)
            {
                return bins() - 1;
            }
            else
            {
                return static_cast<tensor_size_t>(std::distance(begin, it));
            }
        }

    private:

        template <typename tvalue>
        static tensor_size_t cast_bin(tvalue value)
        {
            return static_cast<tensor_size_t>(value);
        }

        static tensor_size_t cast_bin(char value)
        {
            return static_cast<tensor_size_t>(static_cast<unsigned char>(value));
        }

        template <typename titerator>
        void update(titerator begin, titerator end)
        {
            const auto bins = m_thresholds.size() + 1;

            m_bin_means.resize(bins);
            m_bin_counts.resize(bins);
            m_bin_medians.resize(bins);

            m_bin_counts.zero();
            m_bin_means.full(std::numeric_limits<scalar_t>::quiet_NaN());
            m_bin_medians.full(std::numeric_limits<scalar_t>::quiet_NaN());

            for (tensor_size_t bin = 0; bin < bins; ++ bin)
            {
                if (bin + 1 < bins)
                {
                    const auto op = [] (scalar_t threshold, scalar_t value) { return value >= threshold; };
                    const auto it = std::upper_bound(begin, end, m_thresholds(bin), op);
                    update_bin(begin, it, bin);
                    begin = it;
                }
                else
                {
                    update_bin(begin, end, bin);
                }
            }
        }

        template <typename titerator>
        void update_bin(titerator begin, titerator end, tensor_size_t bin)
        {
            const auto count = static_cast<tensor_size_t>(std::distance(begin, end));

            m_bin_counts(bin) = count;
            if (count > 0)
            {
                m_bin_means(bin) = mean(begin, end, count);
                m_bin_medians(bin) = median_sorted(begin, end);
            }
            else
            {
                m_bin_means(bin) = std::numeric_limits<scalar_t>::quiet_NaN();
                m_bin_medians(bin) = std::numeric_limits<scalar_t>::quiet_NaN();
            }
        }

        template <typename titerator>
        static scalar_t mean(titerator begin, titerator end, tensor_size_t count)
        {
            const auto accumulator = [] (scalar_t acc, auto value)
            {
                return acc + static_cast<scalar_t>(value);
            };
            return std::accumulate(begin, end, 0.0, accumulator) / static_cast<scalar_t>(count);
        }

        // attributes
        thresholds_t    m_thresholds;   ///<
        bin_means_t     m_bin_means;    ///<
        bin_counts_t    m_bin_counts;   ///<
        bin_medians_t   m_bin_medians;  ///<
    };
}
