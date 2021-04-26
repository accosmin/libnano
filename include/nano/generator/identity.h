#pragma once

#include <nano/generator.h>

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
    NANO_PUBLIC std::vector<tensor_size_t> select_scalar_components(
        const memory_dataset_t&, struct2scalar, const indices_t& feature_indices);

    ///
    /// \brief
    ///
    class NANO_PUBLIC identity_generator_t : public generator_t
    {
    public:

        identity_generator_t(const memory_dataset_t& dataset, const indices_t& samples);

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;

        void select(tensor_size_t, tensor_range_t, sclass_map_t) const override;
        void select(tensor_size_t, tensor_range_t, mclass_map_t) const override;
        void select(tensor_size_t, tensor_range_t, scalar_map_t) const override;
        void select(tensor_size_t, tensor_range_t, struct_map_t) const override;
        void flatten(tensor_range_t, tensor2d_map_t, tensor_size_t) const override;
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC scalar_elemwise_generator_t : public generator_t
    {
    public:

        scalar_elemwise_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& feature_indices = indices_t{});

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;

    protected:

        auto mapped_ifeature(tensor_size_t ifeature) const { return m_mapping(ifeature, 0); }
        auto mapped_component(tensor_size_t ifeature) const { return m_mapping(ifeature, 1); }

        virtual feature_t make_feature(
            const feature_t&, tensor_size_t component) const = 0;

    private:

        using mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        mapping_t       m_mapping;                              ///<
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC scalar_pairwise_generator_t : public generator_t
    {
    public:

        scalar_pairwise_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& feature_indices = indices_t{});

        tensor_size_t features() const override;
        feature_t feature(tensor_size_t) const override;

    protected:

        auto mapped_ifeature1(tensor_size_t ifeature) const { return m_mapping(ifeature, 0); }
        auto mapped_ifeature2(tensor_size_t ifeature) const { return m_mapping(ifeature, 2); }
        auto mapped_component1(tensor_size_t ifeature) const { return m_mapping(ifeature, 1); }
        auto mapped_component2(tensor_size_t ifeature) const { return m_mapping(ifeature, 3); }

        virtual feature_t make_feature(
            const feature_t&, tensor_size_t component1,
            const feature_t&, tensor_size_t component2) const = 0;

    private:

        using mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        mapping_t       m_mapping;                              ///<
    };

    ///
    /// \brief
    ///
    template <typename toperator>
    class NANO_PUBLIC scalar2scalar_elemwise_generator_t : public scalar_elemwise_generator_t
    {
    public:

        scalar2scalar_elemwise_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            struct2scalar s2s = struct2scalar::off,
            const indices_t& feature_indices = indices_t{}) :
            scalar_elemwise_generator_t(dataset, samples, s2s, feature_indices)
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
            const feature_t& feature, tensor_size_t component) const override
        {
            return  feature_t{scat(toperator::name(), "(", feature.name(), "[", component, "])")}
                    .scalar(toperator::type(feature));
        }

        void do_select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const
        {
            dataset().visit_inputs(mapped_ifeature(ifeature), [&] (const auto&, const auto& data, const auto& mask)
            {
                loop_samples<4U>(data, mask, samples(ifeature, sample_range),
                [&] (auto it)
                {
                    if (should_drop(ifeature))
                    {
                        storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
                    }
                    else
                    {
                        const auto component = mapped_component(ifeature);
                        for (; it; ++ it)
                        {
                            if (const auto [index, given, values] = *it; given)
                            {
                                storage(index) = toperator::value(values(component));
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

        void do_flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column_offset) const
        {
            for (tensor_size_t ifeature = 0, column_size = 0, features = this->features();
                 ifeature < features; ++ ifeature, column_offset += column_size)
            {
                dataset().visit_inputs(mapped_ifeature(ifeature), [&] (const auto& feature, const auto& data, const auto& mask)
                {
                    loop_samples<4U>(data, mask, samples(ifeature, sample_range),
                    [&] (auto it)
                    {
                        column_size = size(feature.dims());
                        for (; it; ++ it)
                        {
                            if (const auto [index, given, values] = *it; given)
                            {
                                auto segment = storage.array(index).segment(column_offset, column_size);
                                if (should_drop(ifeature))
                                {
                                    segment.setConstant(+0.0);
                                }
                                else
                                {
                                    for (tensor_size_t component = 0; component < column_size; ++ component)
                                    {
                                        segment(component) = toperator::value(values(component));
                                    }
                                }
                            }
                            else
                            {
                                auto segment = storage.array(index).segment(column_offset, column_size);
                                segment.setConstant(+0.0);
                            }
                        }
                    },
                    [&] ()
                    {
                        assert(false);
                    });
                });
            }
        }
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

    struct sign_log_scaler_t
    {
        static auto name()
        {
            return "sign*log";
        }

        static auto type(const feature_t&)
        {
            return feature_type::float64;
        }

        template
        <
            typename tscalar,
            std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true
        >
        static auto value(tscalar value)
        {
            const auto svalue = static_cast<scalar_t>(value);
            return (value < 0.0 ? -1.0 : +1.0) * std::log1p(svalue * svalue);
        }
    };

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
