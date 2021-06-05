#pragma once

#include <nano/core/histogram.h>
#include <nano/generator/util.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief
    ///
    class histogram_medians_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        histogram_medians_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off) :
            base_elemwise_generator_t(dataset),
            m_s2s(s2s)
        {
        }

        feature_mapping_t do_fit(indices_cmap_t samples, execution) override
        {
            const auto mapping = select_scalar(dataset(), m_s2s);

            m_histograms.clear();

            for (tensor_size_t ifeature = 0; ifeature < mapping.size<0>(); ++ ifeature)
            {
                const auto original = mapping(ifeature, 0);
                const auto component = std::max(mapping(ifeature, 1), tensor_size_t{0});

                std::vector<scalar_t> allvalues;
                dataset().visit_inputs(original, [&] (const auto&, const auto& data, const auto& mask)
                {
                    loop_samples<input_rank>(data, mask, samples, [&] (auto it)
                    {
                        for (; it; ++ it)
                        {
                            if (const auto [index, given, values] = *it; given)
                            {
                                allvalues.push_back(static_cast<scalar_t>(values(component)));
                            }
                        }
                    });
                });

                m_histograms.push_back(make_histogram(allvalues));
            }

            return mapping;
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original = mapped_original(ifeature);
            const auto component = mapped_component(ifeature);

            const auto& feature = dataset().feature(original);
            return feature_t{scat(suffix(), "(", feature.name(), "[", component, "])")}.scalar(feature_type::float64);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, scalar_map_t storage) const
        {
            const auto& histogram = m_histograms[static_cast<size_t>(ifeature)];

            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = make_value(histogram, values(component));
                }
                else
                {
                    storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                }
            }
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_flatten(dataset_iterator_t<tscalar, input_rank> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto& histogram = m_histograms[static_cast<size_t>(ifeature)];

            const auto should_drop = this->should_drop(ifeature);
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    if (should_drop)
                    {
                        storage(index, column) = 0.0;
                    }
                    else
                    {
                        storage(index, column) = make_value(histogram, values(component));
                    }
                }
                else
                {
                    storage(index, column) = 0.0;
                }
            }
            ++ column;
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
    /// \brief
    ///
    class ratio_histogram_medians_t : public histogram_medians_t
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
    /// \brief
    ///
    class percentile_histogram_medians_t : public histogram_medians_t
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
