#pragma once

#include <nano/core/percentile.h>
#include <nano/generator/util.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    // TODO: generic single and paired generator to handle the mapping and the dropping and shuffling part
    // TODO: polynomial features
    // TODO: basic image-based features: gradients, magnitude, orientation, HoG
    // TODO: support for stateful generators (e.g. automatically find which scalar features need to scaled, percentiles)

    // TODO: generic utilities for
    //  * percentiles & histogram classes
    //  * averages in percentiles and histogram bins
    //  * w/o non-linear scaling functions

    ///
    /// \brief log-scale the feature values while preserving their signs.
    ///
    class slog1p_t : public generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        slog1p_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off) :
            generator_t(dataset),
            m_s2s(s2s)
        {
        }

        void fit(indices_cmap_t, execution) override
        {
            m_feature_mapping = select_scalar(dataset(), m_s2s);

            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original = this->mapped_original(ifeature);
            const auto component = this->mapped_component(ifeature);

            const auto& feature = this->dataset().feature(original);
            return feature_t{scat("slog1p(", feature.name(), "[", component, "])")}.scalar(feature_type::float64);
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor_size_t mapped_component(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 1);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, scalar_map_t storage) const
        {
            const auto component = this->mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = this->make_value(values(component));
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
            const auto should_drop = this->should_drop(ifeature);
            const auto component = this->mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    if (should_drop)
                    {
                        storage(index, column) = +0.0;
                    }
                    else
                    {
                        storage(index, column) = this->make_value(values(component));
                    }
                }
                else
                {
                    storage(index, column) = +0.0;
                }
            }
            ++ column;
        }

    private:

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto make_value(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return (svalue < 0.0 ? -1.0 : +1.0) * std::log1p(std::fabs(svalue));
        }

        // attributes
        struct2scalar       m_s2s{struct2scalar::off};  ///<
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };

    ///
    /// \brief
    ///
    class sign_t : public generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        sign_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off) :
            generator_t(dataset),
            m_s2s(s2s)
        {
        }

        void fit(indices_cmap_t, execution) override
        {
            m_feature_mapping = select_scalar(dataset(), m_s2s);

            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original = this->mapped_original(ifeature);
            const auto component = this->mapped_component(ifeature);

            const auto& feature = this->dataset().feature(original);
            return feature_t{scat("sign(", feature.name(), "[", component, "])")}.scalar(feature_type::float64);
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor_size_t mapped_component(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 1);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, scalar_map_t storage) const
        {
            const auto component = this->mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = this->make_value(values(component));
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
            const auto should_drop = this->should_drop(ifeature);
            const auto component = this->mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    if (should_drop)
                    {
                        storage(index, column) = +0.0;
                    }
                    else
                    {
                        storage(index, column) = this->make_value(values(component));
                    }
                }
                else
                {
                    storage(index, column) = +0.0;
                }
            }
            ++ column;
        }

    private:

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto make_value(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return svalue < 0.0 ? -1.0 : +1.0;
        }

        // attributes
        struct2scalar       m_s2s{struct2scalar::off};  ///<
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };

    ///
    /// \brief
    ///
    class sign_class_t : public generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::sclass;

        sign_class_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off) :
            generator_t(dataset),
            m_s2s(s2s)
        {
        }

        void fit(indices_cmap_t, execution) override
        {
            m_feature_mapping = select_scalar(dataset(), m_s2s);

            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original = this->mapped_original(ifeature);
            const auto component = this->mapped_component(ifeature);

            const auto& feature = this->dataset().feature(original);
            return feature_t{scat("sign_class(", feature.name(), "[", component, "])")}.sclass(strings_t{"neg", "pos"});
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor_size_t mapped_component(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 1);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, sclass_map_t storage) const
        {
            const auto component = this->mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = this->make_value(values(component));
                }
                else
                {
                    storage(index) = -1;
                }
            }
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_flatten(dataset_iterator_t<tscalar, input_rank> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto should_drop = this->should_drop(ifeature);
            const auto component = this->mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    if (should_drop)
                    {
                        storage(index, column + 0) = 0.0;
                        storage(index, column + 1) = 0.0;
                    }
                    else
                    {
                        const auto label = this->make_value(values(component));
                        if (label == 0)
                        {
                            storage(index, column + 0) = +1.0;
                            storage(index, column + 1) = -1.0;
                        }
                        else
                        {
                            storage(index, column + 0) = -1.0;
                            storage(index, column + 1) = +1.0;
                        }
                    }
                }
                else
                {
                    storage(index, column + 0) = 0.0;
                    storage(index, column + 1) = 0.0;
                }
            }
            column += 2;
        }

    private:

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto make_value(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return svalue < 0.0 ? 0 : 1;
        }

        // attributes
        struct2scalar       m_s2s{struct2scalar::off};  ///<
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };

    ///
    /// \brief
    ///
    class percentile_bin_class_t : public generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::sclass;

        percentile_bin_class_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off,
            tensor_size_t bins = 10) :
            generator_t(dataset),
            m_s2s(s2s),
            m_bins(bins)
        {
            assert(bins > 0);
        }

        void fit(indices_cmap_t samples, execution) override
        {
            m_feature_mapping = select_scalar(dataset(), m_s2s);

            m_thresholds.resize(features(), m_bins - 1);

            for (tensor_size_t ifeature = 0; ifeature < features(); ++ ifeature)
            {
                const auto original = mapped_original(ifeature);
                const auto component = mapped_component(ifeature);

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

                std::sort(allvalues.begin(), allvalues.end());

                for (tensor_size_t ibin = 0; ibin + 1 < m_bins; ++ ibin)
                {
                    const auto percentile = 100.0 * (ibin + 1) / m_bins;
                    m_thresholds(ifeature, ibin) = ::nano::percentile(allvalues.begin(), allvalues.end(), percentile);
                }
            }

            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original = this->mapped_original(ifeature);
            const auto component = this->mapped_component(ifeature);

            const auto& feature = this->dataset().feature(original);
            return  feature_t{scat("percbin(", feature.name(), "[", component, "])")}.
                    sclass(static_cast<size_t>(m_bins));
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor_size_t mapped_component(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 1);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, sclass_map_t storage) const
        {
            const auto thresholds = m_thresholds.vector(ifeature);
            const auto component = this->mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = this->make_value(thresholds, values(component));
                }
                else
                {
                    storage(index) = -1;
                }
            }
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_flatten(dataset_iterator_t<tscalar, input_rank> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto should_drop = this->should_drop(ifeature);
            const auto thresholds = m_thresholds.vector(ifeature);
            const auto component = this->mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    auto segment = storage.array(index).segment(column, m_bins);
                    if (should_drop)
                    {
                        segment.setConstant(0.0);
                    }
                    else
                    {
                        const auto label = this->make_value(thresholds, values(component));
                        segment.setConstant(-1.0);
                        segment(label) = +1.0;
                    }
                }
                else
                {
                    auto segment = storage.array(index).segment(column, m_bins);
                    segment.setConstant(0.0);
                }
            }
            column += m_bins;
        }

    private:

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        int32_t make_value(const vector_cmap_t& thresholds, tscalar value)
        {
            int32_t bin = static_cast<int32_t>(m_bins) - 1;
            while (value < thresholds(bin) && bin > 0)
            {
                -- bin;
            }
            return bin;
        }

        // attributes
        struct2scalar       m_s2s{struct2scalar::off};  ///<
        tensor_size_t       m_bins{10};                 ///<
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
        tensor2d_t          m_thresholds;               ///< (feature index, threshold)
    };
}
