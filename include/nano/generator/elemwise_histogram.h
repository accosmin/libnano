#pragma once

#include <nano/core/histogram.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief generate features by (histogram) binning the original feature values independently.
    ///
    /// the resulting feature value is the median of the bin the original feature value belongs to.
    ///
    class NANO_PUBLIC histogram_medians_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        histogram_medians_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off);

        feature_t feature(tensor_size_t ifeature) const override;
        feature_mapping_t do_fit(indices_cmap_t samples, execution) override;

        auto process(tensor_size_t ifeature) const
        {
            const auto& histogram = m_histograms[static_cast<size_t>(ifeature)];
            const auto component = mapped_component(ifeature);

            const auto colsize = tensor_size_t{1};
            const auto process = [=] (const auto& values)
            {
                return make_value(histogram, values(component));
            };

            return std::make_tuple(process, colsize);
        }

    protected:

        virtual string_t suffix() const = 0;
        virtual histogram_t make_histogram(std::vector<scalar_t>& values) const = 0;

   private:

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static scalar_t make_value(const histogram_t& histogram, tscalar value)
        {
            return histogram.median(histogram.bin(value));
        }

        using histograms_t = std::vector<histogram_t>;

        // attributes
        struct2scalar       m_s2s{struct2scalar::off};  ///<
        histograms_t        m_histograms;               ///< histogram per feature
    };

    ///
    /// \brief construct histograms using equidistant ratios of the original feature values.
    ///
    class NANO_PUBLIC ratio_histogram_medians_t : public histogram_medians_t
    {
    public:

        ratio_histogram_medians_t(
            const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off, tensor_size_t bins = 10) :
            histogram_medians_t(dataset, s2s),
            m_bins(bins)
        {
        }

        string_t suffix() const override
        {
            return scat("ratio_hist[", m_bins, "]");
        }

        histogram_t make_histogram(std::vector<scalar_t>& values) const override
        {
            return histogram_t::make_from_ratios(std::begin(values), std::end(values), m_bins);
        }

    private:

        // attributes
        tensor_size_t       m_bins{10};     ///<
    };

    ///
    /// \brief construct histograms using equidistant percentiles of the original feature values.
    ///
    class NANO_PUBLIC percentile_histogram_medians_t : public histogram_medians_t
    {
    public:

        percentile_histogram_medians_t(
            const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off, tensor_size_t bins = 10) :
            histogram_medians_t(dataset, s2s),
            m_bins(bins)
        {
        }

        string_t suffix() const override
        {
            return scat("perc_hist[", m_bins, "]");
        }

        histogram_t make_histogram(std::vector<scalar_t>& values) const override
        {
            return histogram_t::make_from_percentiles(std::begin(values), std::end(values), m_bins);
        }

    private:

        // attributes
        tensor_size_t       m_bins{10};     ///<
    };
}
