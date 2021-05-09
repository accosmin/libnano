#pragma once

#include <nano/generator/util.h>
#include <nano/generator/pairwise.h>

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

    ///
    /// \brief
    ///
    template <typename tcomputer, std::enable_if_t<std::is_base_of_v<pairwise_generator_t, tcomputer>, bool> = true>
    class NANO_PUBLIC scalar_pairwise_generator_t : public tcomputer
    {
    public:

        scalar_pairwise_generator_t(
            const memory_dataset_t& dataset,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& feature_indices = indices_t{}) :
            tcomputer(dataset, select_scalar_components(dataset, s2s, feature_indices))
        {
        }

        void select(indices_cmap_t samples, tensor_size_t ifeature, scalar_map_t storage) const override
        {
            this->iterate2(samples, ifeature, this->mapped_ifeature1(ifeature), this->mapped_ifeature2(ifeature), [&] (
                const auto&, const auto& data1, const auto& mask1,
                const auto&, const auto& data2, const auto& mask2,
                indices_cmap_t samples)
            {
                loop_samples<4U>(data1, mask1, data2, mask2, samples, [&] (auto it)
                {
                    if (this->should_drop(ifeature))
                    {
                        storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
                    }
                    else
                    {
                        const auto component1 = this->mapped_component1(ifeature);
                        const auto component2 = this->mapped_component2(ifeature);
                        for (; it; ++ it)
                        {
                            if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                            {
                                storage(index) = this->make_value(values1(component1), values2(component2));
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

        void flatten(indices_cmap_t samples, tensor2d_map_t storage, tensor_size_t column) const override
        {
            for (tensor_size_t ifeature = 0, features = this->features(); ifeature < features; ++ ifeature, ++ column)
            {
                this->iterate2(samples, ifeature, this->mapped_ifeature1(ifeature), this->mapped_ifeature2(ifeature), [&] (
                    const auto&, const auto& data1, const auto& mask1,
                    const auto&, const auto& data2, const auto& mask2,
                    indices_cmap_t samples)
                {
                    loop_samples<4U>(data1, mask1, data2, mask2, samples, [&] (auto it)
                    {
                        const auto component1 = this->mapped_component1(ifeature);
                        const auto component2 = this->mapped_component2(ifeature);
                        for (; it; ++ it)
                        {
                            if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                            {
                                if (this->should_drop(ifeature))
                                {
                                    storage(index, column) = +0.0;
                                }
                                else
                                {
                                    storage(index, column) = this->make_value(values1(component1), values2(component2));
                                }
                            }
                            else
                            {
                                storage(index, column) = +0.0;
                            }
                        }
                    });
                });
            }
        }
    };

    ///
    /// \brief
    ///
    class product_t : public pairwise_generator_t
    {
    public:

        static constexpr auto generated_feature_type = feature_type::float64;

        product_t(const memory_dataset_t& dataset, feature_mapping_t feature_mapping) :
            pairwise_generator_t(dataset, std::move(feature_mapping))
        {
        }

        feature_t make_feature(
            const feature_t& feature1, tensor_size_t component1,
            const feature_t& feature2, tensor_size_t component2) const override
        {
            auto name = scat("product(", feature1.name(), "[", component1, "],", feature2.name(), "[", component2, "])");
            return feature_t{std::move(name)}.scalar(feature_type::float64);
        }

        template
        <
            typename tscalar1,
            typename tscalar2,
            std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true
        >
        static auto make_value(tscalar1 value1, tscalar2 value2)
        {
            return static_cast<scalar_t>(value1) * static_cast<scalar_t>(value2);
        }
    };
}
