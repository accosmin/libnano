#pragma once

#include <nano/dataset/generator.h>

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
    template <typename toperator>
    class NANO_PUBLIC scalar_elemwise_generator_t : public generator_t
    {
    public:

        scalar_elemwise_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            string_t feature_name,
            struct2scalar s2s = struct2scalar::off,
            indices_t feature_indices = indices_t{}) :
            generator_t(dataset, samples),
            m_feature_name(std::move(feature_name)),
            m_struct2scalar(s2s)
        {
            make_mapping(feature_indices);
            allocate(this->features());
        }

        tensor_size_t features() const override { return get_features(); }
        feature_t feature(tensor_size_t ifeature) const override { return get_feature(ifeature); }

        void select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const override
        {
            do_select(ifeature, sample_range, storage);
        }
        void flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column_offset) const override
        {
            do_flatten(sample_range, storage, column_offset);
        }

    private:

        void make_mapping(const indices_t& feature_indices)
        {
            const auto mapping = select_scalar_components(dataset(), m_struct2scalar, feature_indices);

            m_mapping = map_tensor(mapping.data(), static_cast<tensor_size_t>(mapping.size()) / 2, 2);
        }

        tensor_size_t get_features() const
        {
            return m_mapping.size<0>();
        }

        feature_t get_feature(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < m_mapping.size<0>());

            const auto component = m_mapping(ifeature, 1);
            const auto& feature = dataset().feature(m_mapping(ifeature, 0));

            return  feature_t{scat(m_feature_name, "(", feature.name(), "[", component, "])")}
                    .scalar(toperator::type(feature), make_dims(1, 1, 1));
        }

        void do_select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const
        {
            dataset().visit_inputs(m_mapping(ifeature, 0), [&] (const auto&, const auto& data, const auto& mask)
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
                        const auto component = m_mapping(ifeature, 1);
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
                dataset().visit_inputs(m_mapping(ifeature, 0), [&] (const auto& feature, const auto& data, const auto& mask)
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

        using mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        mapping_t       m_mapping;                              ///<
        string_t        m_feature_name;                         ///<
        struct2scalar   m_struct2scalar{struct2scalar::off};    ///<
    };

    ///
    /// \brief
    ///
    template <typename toperator>
    class NANO_PUBLIC scalar_pairwise_generator_t : public generator_t
    {
    public:

        scalar_pairwise_generator_t(
            const memory_dataset_t& dataset, const indices_t& samples,
            string_t feature_name,
            struct2scalar s2s = struct2scalar::off,
            indices_t feature_indices = indices_t{}) :
            generator_t(dataset, samples),
            m_feature_name(std::move(feature_name)),
            m_struct2scalar(s2s)
        {
            make_mapping(feature_indices);
            allocate(this->features());
        }

        tensor_size_t features() const override { return get_features(); }
        feature_t feature(tensor_size_t ifeature) const override { return get_feature(ifeature); }

        void select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const override
        {
            do_select(ifeature, sample_range, storage);
        }
        void flatten(tensor_range_t sample_range, tensor2d_map_t storage, tensor_size_t column_offset) const override
        {
            do_flatten(sample_range, storage, column_offset);
        }

    private:

        void make_mapping(const indices_t& feature_indices)
        {
            const auto mapping = select_scalar_components(dataset(), m_struct2scalar, feature_indices);

            const auto size = static_cast<tensor_size_t>(mapping.size() / 2);

            m_mapping.resize(size * (size + 1) / 2, 4);
            for (tensor_size_t i = 0, k = 0; i < size; ++ i)
            {
                const auto feature1 = mapping[static_cast<size_t>(i) * 2 + 0];
                const auto component1 = mapping[static_cast<size_t>(i) * 2 + 1];

                for (tensor_size_t j = i; j < size; ++ j, ++ k)
                {
                    const auto feature2 = mapping[static_cast<size_t>(j) * 2 + 0];
                    const auto component2 = mapping[static_cast<size_t>(j) * 2 + 1];

                    m_mapping(k, 0) = feature1;
                    m_mapping(k, 1) = component1;
                    m_mapping(k, 2) = feature2;
                    m_mapping(k, 3) = component2;
                }
            }
        }

        tensor_size_t get_features() const
        {
            return m_mapping.size<0>();
        }

        feature_t get_feature(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < m_mapping.size<0>());

            const auto component1 = m_mapping(ifeature, 1);
            const auto component2 = m_mapping(ifeature, 3);

            const auto& feature1 = dataset().feature(m_mapping(ifeature, 0));
            const auto& feature2 = dataset().feature(m_mapping(ifeature, 2));

            return  feature_t{scat(m_feature_name,
                    "(", feature1.name(), "[", component1, "]",
                    ",", feature2.name(), "[", component2, "])")}
                    .scalar(toperator::type(feature1, feature2), make_dims(1, 1, 1));
        }

        void do_select(tensor_size_t ifeature, tensor_range_t sample_range, scalar_map_t storage) const
        {
            dataset().visit_inputs(m_mapping(ifeature, 0), [&] (const auto&, const auto& data1, const auto& mask1)
            {
                dataset().visit_inputs(m_mapping(ifeature, 2), [&] (const auto&, const auto& data2, const auto& mask2)
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
                            const auto component1 = m_mapping(ifeature, 1);
                            const auto component2 = m_mapping(ifeature, 3);
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
                dataset().visit_inputs(m_mapping(ifeature, 0), [&] (const auto&, const auto& data1, const auto& mask1)
                {
                    dataset().visit_inputs(m_mapping(ifeature, 2), [&] (const auto&, const auto& data2, const auto& mask2)
                    {
                        loop_samples<4U>(data1, mask1, data2, mask2, samples(ifeature, sample_range),
                        [&] (auto it)
                        {
                            const auto component1 = m_mapping(ifeature, 1);
                            const auto component2 = m_mapping(ifeature, 3);
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

        using mapping_t = tensor_mem_t<tensor_size_t, 2>;

        // attributes
        mapping_t       m_mapping;                              ///<
        string_t        m_feature_name;                         ///<
        struct2scalar   m_struct2scalar{struct2scalar::off};    ///<
    };

    struct sign_log_scaler_t
    {
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
