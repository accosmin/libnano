#pragma once

#include <nano/generator/util.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    // TODO: generic single and paired generator to handle the mapping and the dropping and shuffling part
    // TODO: feature-wise non-linear transformations of scalar features - sign(x)*log(1+x*x), x/sqrt(1+x*x)
    // TODO: polynomial features
    // TODO: basic image-based features: gradients, magnitude, orientation, HoG
    // TODO: histogram-based scalar features - assign scalar value into its percentile range index
    // TODO: sign -> transform scalar value to its sign class or sign scalar value
    // TODO: clamp_perc -> clamp scalar value outside a given percentile range
    // TODO: clamp -> clamp scalar value to given range
    // TODO: support for stateful generators (e.g. automatically find which scalar features need to scaled, percentiles)

    ///
    /// \brief
    ///
    template <typename tcomputer, std::enable_if_t<std::is_base_of_v<elemwise_generator_t, tcomputer>, bool> = true>
    class NANO_PUBLIC scalar_elemwise_generator_t : public tcomputer
    {
    public:

        scalar_elemwise_generator_t(
            const memory_dataset_t& dataset,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& feature_indices = indices_t{}) :
            tcomputer(dataset, select_scalar_components(dataset, s2s, feature_indices))
        {
            if constexpr (tcomputer::generated_feature_type == feature_type::sclass)
            {
                m_feature_classes.resize(this->features());
                for (tensor_size_t ifeature = 0, features = this->features(); ifeature < features; ++ ifeature)
                {
                    m_feature_classes(ifeature) = this->feature(ifeature).classes();
                }
            }
        }

        void select(indices_cmap_t samples, tensor_size_t ifeature, scalar_map_t storage) const override
        {
            if constexpr (tcomputer::generated_feature_type != feature_type::sclass)
            {
                this->iterate1(samples, ifeature, this->mapped_ifeature(ifeature), [&] (
                    const auto&, const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<4U>(data, mask, samples, [&] (auto it)
                    {
                        if (this->should_drop(ifeature))
                        {
                            storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                        else
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
                    });
                });
            }
            else
            {
                generator_t::select(samples, ifeature, storage);
            }
        }

        void select(indices_cmap_t samples, tensor_size_t ifeature, sclass_map_t storage) const override
        {
            if constexpr (tcomputer::generated_feature_type == feature_type::sclass)
            {
                this->iterate1(samples, ifeature, this->mapped_ifeature(ifeature), [&] (
                    const auto&, const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<4U>(data, mask, samples, [&] (auto it)
                    {
                        if (this->should_drop(ifeature))
                        {
                            storage.full(-1);
                        }
                        else
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
                    });
                });
            }
            else
            {
                generator_t::select(samples, ifeature, storage);
            }
        }

        void flatten(indices_cmap_t samples, tensor2d_map_t storage, tensor_size_t column) const override
        {
            for (tensor_size_t ifeature = 0, features = this->features(); ifeature < features; ++ ifeature)
            {
                this->iterate1(samples, ifeature, this->mapped_ifeature(ifeature), [&] (
                    const auto&, const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<4U>(data, mask, samples, [&] (auto it)
                    {
                        const auto component = this->mapped_component(ifeature);
                        if constexpr (tcomputer::generated_feature_type == feature_type::sclass)
                        {
                            const auto colsize = m_feature_classes(ifeature);
                            for (; it; ++ it)
                            {
                                if (const auto [index, given, values] = *it; given)
                                {
                                    auto segment = storage.array(index).segment(column, colsize);
                                    if (this->should_drop(ifeature))
                                    {
                                        segment.setConstant(+0.0);
                                    }
                                    else
                                    {
                                        segment.setConstant(-1.0);
                                        segment(this->make_value(values(component))) = +1.0;
                                    }
                                }
                                else
                                {
                                    auto segment = storage.array(index).segment(column, colsize);
                                    segment.setConstant(+0.0);
                                }
                            }

                            column += colsize;
                        }
                        else
                        {
                            for (; it; ++ it)
                            {
                                if (const auto [index, given, values] = *it; given)
                                {
                                    if (this->should_drop(ifeature))
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
                    });
                });
            }
        }

        // attributes
        indices_t       m_feature_classes;  ///<
    };

    ///
    /// \brief
    ///
    class slog1p_t : public elemwise_generator_t
    {
    public:

        static constexpr auto generated_feature_type = feature_type::float64;

        slog1p_t(const memory_dataset_t& dataset, feature_mapping_t feature_mapping) :
            elemwise_generator_t(dataset, std::move(feature_mapping))
        {
        }

        feature_t make_feature(const feature_t& feature, tensor_size_t component) const override
        {
            return feature_t{scat("slog1p(", feature.name(), "[", component, "])")}.scalar(feature_type::float64);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto make_value(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return (svalue < 0.0 ? -1.0 : +1.0) * std::log1p(std::fabs(svalue));
        }
    };

    ///
    /// \brief
    ///
    class sign_t : public elemwise_generator_t
    {
    public:

        static constexpr auto generated_feature_type = feature_type::float64;

        sign_t(const memory_dataset_t& dataset, feature_mapping_t feature_mapping) :
            elemwise_generator_t(dataset, std::move(feature_mapping))
        {
        }

        feature_t make_feature(const feature_t& feature, tensor_size_t component) const override
        {
            return feature_t{scat("sign(", feature.name(), "[", component, "])")}.scalar(feature_type::float64);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto make_value(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return (svalue < 0.0) ? -1.0 : +1.0;
        }
    };

    ///
    /// \brief
    ///
    class sign_class_t : public elemwise_generator_t
    {
    public:

        static constexpr auto generated_feature_type = feature_type::sclass;

        sign_class_t(const memory_dataset_t& dataset, feature_mapping_t feature_mapping) :
            elemwise_generator_t(dataset, std::move(feature_mapping))
        {
        }

        feature_t make_feature(const feature_t& feature, tensor_size_t component) const override
        {
            return feature_t{scat("sign_class(", feature.name(), "[", component, "])")}.sclass(strings_t{"negative", "positive"});
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto make_value(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return (svalue < 0.0) ? 0 : 1;
        }
    };

    ///
    /// \brief
    ///
    class percentile_class_t : public elemwise_generator_t
    {
    public:

        static constexpr auto generated_feature_type = feature_type::sclass;

        percentile_class_t(const memory_dataset_t& dataset, feature_mapping_t feature_mapping,
            std::vector<scalar_t> percentiles = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0}) :
            elemwise_generator_t(dataset, std::move(feature_mapping)),
            m_percentiles(std::move(percentiles))
        {
        }

        void fit(indices_cmap_t, execution) override
        {
            // TODO: compute the percentile thresholds for all original features
            // TODO:
        }

        feature_t make_feature(const feature_t& feature, tensor_size_t component) const override
        {
            return  feature_t{scat("percentile_class(", feature.name(), "[", component, "])")}.
                    sclass(m_percentiles.size());
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        static auto make_value(tscalar value)
        {
            // TODO
            const auto svalue = static_cast<scalar_t>(value);
            return (svalue < 0.0) ? 0 : 1;
        }

    private:

        // attributes
        std::vector<scalar_t>   m_percentiles;  ///<
    };
}
