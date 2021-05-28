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
        static histogram_t make_from_percentiles(
            titerator begin, titerator end, tensor_size_t bins)
        {
            return make_from_percentiles(begin, end, make_equidistant_percentiles(bins));
        }

        template <typename titerator>
        static histogram_t make_from_percentiles(
            titerator begin, titerator end, tensor_mem_t<scalar_t, 1> percentiles)
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
        static histogram_t make_from_thresholds(
            titerator begin, titerator end, const tensor_mem_t<scalar_t, 1>& thresholds)
        {
            return histogram_t(begin, end, thresholds);
        }

        template <typename titerator>
        static histogram_t make_from_ratios(
            titerator begin, titerator end, tensor_size_t bins)
        {
            return make_from_ratios(begin, end, make_equidistant_ratios(bins));
        }

        template <typename titerator>
        static histogram_t make_from_ratios(
            titerator begin, titerator end, tensor_mem_t<scalar_t, 1> ratios)
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

        const auto& means() const { return m_bin_means; }
        const auto& counts() const { return m_bin_counts; }
        const auto& medians() const { return m_bin_medians; }
        const auto& thresholds() const { return m_thresholds; }

        tensor_size_t bins() const { return m_bin_counts.size(); }
        scalar_t mean(tensor_size_t bin) const { return m_bin_means(bin); }
        scalar_t count(tensor_size_t bin) const { return m_bin_counts(bin); }
        scalar_t median(tensor_size_t bin) const { return m_bin_medians(bin); }

        template <typename tvalue>
        tensor_size_t bin(tvalue value) const
        {
            const auto svalue = static_cast<tensor_size_t>(value);

            const auto begin = ::nano::begin(m_thresholds);
            const auto end = ::nano::end(m_thresholds);

            const auto it = std::upper_bound(begin, end, svalue);
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
                    const auto it = std::upper_bound(begin, end, m_thresholds(bin));
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
                const auto accumulator = [] (scalar_t acc, auto value) { return acc + static_cast<scalar_t>(value); };
                m_bin_means(bin) = std::accumulate(begin, end, 0.0, accumulator) / static_cast<scalar_t>(count);
                m_bin_medians(bin) = median_sorted(begin, end);
            }
            else
            {
                m_bin_means(bin) = std::numeric_limits<scalar_t>::quiet_NaN();
                m_bin_medians(bin) = std::numeric_limits<scalar_t>::quiet_NaN();
            }
        }

        // attributes
        thresholds_t    m_thresholds;   ///<
        bin_means_t     m_bin_means;    ///<
        bin_counts_t    m_bin_counts;   ///<
        bin_medians_t   m_bin_medians;  ///<
    };
}
