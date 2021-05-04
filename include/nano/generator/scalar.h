#pragma once

#include <nano/generator/util.h>
#include <nano/generator/elemwise.h>
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
    template <typename tcomputer, std::enable_if_t<std::is_base_of_v<elemwise_generator_t, tcomputer>, bool> = true>
    class NANO_PUBLIC scalar_elemwise_generator_t : public tcomputer
    {
    public:

        scalar_elemwise_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& feature_indices = indices_t{}) :
            tcomputer(dataset, samples, select_scalar_components(dataset, s2s, feature_indices))
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

        void select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const override
        {
            do_select(ifeature, sample_range, storage);
        }
        void select(tensor_size_t ifeature, tensor_range_t sample_range, sclass_map_t storage) const override
        {
            do_select(ifeature, sample_range, storage);
        }
        void flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column_offset) const override
        {
            do_flatten(sample_range, storage, column_offset);
        }

    private:

        void do_select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const
        {
            if constexpr (tcomputer::generated_feature_type == feature_type::sclass)
            {
                generator_t::select(ifeature, sample_range, storage);
            }
            else
            {
                this->dataset().visit_inputs(this->mapped_ifeature(ifeature), [&] (const auto&, const auto& data, const auto& mask)
                {
                    loop_samples<4U>(data, mask, this->samples(ifeature, sample_range),
                    [&] (auto it)
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
                    },
                    [&] ()
                    {
                        generator_t::select(ifeature, sample_range, storage);
                    });
                });
            }
        }

        void do_select(tensor_size_t ifeature, tensor_range_t sample_range, sclass_map_t storage) const
        {
            if constexpr (tcomputer::generated_feature_type == feature_type::sclass)
            {
                this->dataset().visit_inputs(this->mapped_ifeature(ifeature), [&] (const auto&, const auto& data, const auto& mask)
                {
                    loop_samples<4U>(data, mask, this->samples(ifeature, sample_range),
                    [&] (auto it)
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
                    },
                    [&] ()
                    {
                        generator_t::select(ifeature, sample_range, storage);
                    });
                });
            }
            else
            {
                generator_t::select(ifeature, sample_range, storage);
            }
        }

        void do_flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column) const
        {
            for (tensor_size_t ifeature = 0, features = this->features(); ifeature < features; ++ ifeature)
            {
                this->dataset().visit_inputs(this->mapped_ifeature(ifeature), [&] (const auto&, const auto& data, const auto& mask)
                {
                    loop_samples<4U>(data, mask, this->samples(ifeature, sample_range),
                    [&] (auto it)
                    {
                        const auto component = this->mapped_component(ifeature);
                        if constexpr (tcomputer::generated_feature_type == feature_type::sclass)
                        {
                            const auto column_size = m_feature_classes(ifeature);
                            for (; it; ++ it)
                            {
                                if (const auto [index, given, values] = *it; given)
                                {
                                    auto segment = storage.array(index).segment(column, column_size);
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
                                    auto segment = storage.array(index).segment(column, column_size);
                                    segment.setConstant(+0.0);
                                }
                            }

                            column += column_size;
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
                    },
                    [&] ()
                    {
                        assert(false);
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

        slog1p_t(const memory_dataset_t& dataset, const indices_t& samples, feature_mapping_t feature_mapping) :
            elemwise_generator_t(dataset, samples, std::move(feature_mapping))
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

        sign_t(const memory_dataset_t& dataset, const indices_t& samples, feature_mapping_t feature_mapping) :
            elemwise_generator_t(dataset, samples, std::move(feature_mapping))
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

        sign_class_t(const memory_dataset_t& dataset, const indices_t& samples, feature_mapping_t feature_mapping) :
            elemwise_generator_t(dataset, samples, std::move(feature_mapping))
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

    ///
    /// \brief
    ///
    class NANO_PUBLIC scalar_pairwise_generator_t : public pairwise_generator_t
    {
    public:

        scalar_pairwise_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& feature_indices = indices_t{});
    };

    ///
    /// \brief
    ///
    template <typename toperator>
    class NANO_PUBLIC scalar2scalar_pairwise_generator_t : public scalar_pairwise_generator_t
    {
    public:

        scalar2scalar_pairwise_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& feature_indices = indices_t{}) :
            scalar_pairwise_generator_t(dataset, samples, s2s, feature_indices)
        {
        }

        void select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const override
        {
            do_select(ifeature, sample_range, storage);
        }
        void flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column_offset) const override
        {
            do_flatten(sample_range, storage, column_offset);
        }

    private:

        feature_t make_feature(
            const feature_t& feature1, tensor_size_t component1,
            const feature_t& feature2, tensor_size_t component2) const override
        {
            return  feature_t{scat(toperator::name(),
                    "(", feature1.name(), "[", component1, "]",
                    ",", feature2.name(), "[", component2, "])")}
                    .scalar(toperator::type(feature1, feature2));
        }

        void do_select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const
        {
            dataset().visit_inputs(mapped_ifeature1(ifeature), [&] (const auto&, const auto& data1, const auto& mask1)
            {
                dataset().visit_inputs(mapped_ifeature2(ifeature), [&] (const auto&, const auto& data2, const auto& mask2)
                {
                    loop_samples<4U>(data1, mask1, data2, mask2, samples(ifeature, sample_range),
                    [&] (auto it)
                    {
                        if (should_drop(ifeature))
                        {
                            storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                        else
                        {
                            const auto component1 = mapped_component1(ifeature);
                            const auto component2 = mapped_component2(ifeature);
                            for (; it; ++ it)
                            {
                                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                                {
                                    storage(index) = toperator::value(values1(component1), values2(component2));
                                }
                                else
                                {
                                    storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                                }
                            }
                        }
                    },
                    [&] ()
                    {
                        generator_t::select(ifeature, sample_range, storage);
                    });
                });
            });
        }

        void do_flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column) const
        {
            for (tensor_size_t ifeature = 0, features = this->features(); ifeature < features; ++ ifeature, ++ column)
            {
                dataset().visit_inputs(mapped_ifeature1(ifeature), [&] (const auto&, const auto& data1, const auto& mask1)
                {
                    dataset().visit_inputs(mapped_ifeature2(ifeature), [&] (const auto&, const auto& data2, const auto& mask2)
                    {
                        loop_samples<4U>(data1, mask1, data2, mask2, samples(ifeature, sample_range),
                        [&] (auto it)
                        {
                            const auto component1 = mapped_component1(ifeature);
                            const auto component2 = mapped_component2(ifeature);
                            for (; it; ++ it)
                            {
                                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                                {
                                    if (should_drop(ifeature))
                                    {
                                        storage(index, column) = +0.0;
                                    }
                                    else
                                    {
                                        storage(index, column) = toperator::value(values1(component1), values2(component2));
                                    }
                                }
                                else
                                {
                                    storage(index, column) = +0.0;
                                }
                            }
                        },
                        [&] ()
                        {
                            assert(false);
                        });
                    });
                });
            }
        }
    };

    //TODO: use CRTP to easily handle all transformations, including preprocessing of samples.

    /// \brief
    ///
    struct pairwise_product_t
    {
        static auto name()
        {
            return "product";
        }

        static auto type(const feature_t&, const feature_t&)
        {
            return feature_type::float64;
        }

        template
        <
            typename tscalar1,
            typename tscalar2,
            std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true
        >
        static auto value(tscalar1 value1, tscalar2 value2)
        {
            return static_cast<scalar_t>(value1) * static_cast<scalar_t>(value2);
        }
    };
}
