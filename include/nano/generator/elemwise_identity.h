#pragma once

#include <nano/generator/util.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief
    ///
    class sclass_identity_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 1U;
        static constexpr auto generated_type = generator_type::sclass;

        sclass_identity_t(const memory_dataset_t& dataset) :
            base_elemwise_generator_t(dataset)
        {
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            return dataset().feature(mapped_original(ifeature));
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_sclass(dataset(), sclass2binary::off);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t, sclass_map_t storage) const
        {
            for (; it; ++ it)
            {
                if (const auto [index, given, label] = *it; given)
                {
                    storage(index) = static_cast<int32_t>(label);
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
            const auto colsize = mapped_classes(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, label] = *it; given)
                {
                    auto segment = storage.array(index).segment(column, colsize);
                    if (should_drop)
                    {
                        segment.setConstant(+0.0);
                    }
                    else
                    {
                        segment.setConstant(-1.0);
                        segment(static_cast<tensor_size_t>(label)) = +1.0;
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
    };

    ///
    /// \brief
    ///
    class mclass_identity_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 2U;
        static constexpr auto generated_type = generator_type::mclass;

        mclass_identity_t(const memory_dataset_t& dataset) :
            base_elemwise_generator_t(dataset)
        {
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            return dataset().feature(mapped_original(ifeature));
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_mclass(dataset(), mclass2binary::off);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t, mclass_map_t storage) const
        {
            for (; it; ++ it)
            {
                if (const auto [index, given, hits] = *it; given)
                {
                    storage.vector(index) = hits.array().template cast<int8_t>();
                }
                else
                {
                    storage.vector(index).setConstant(-1);
                }
            }
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_flatten(dataset_iterator_t<tscalar, input_rank> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto should_drop = this->should_drop(ifeature);
            const auto colsize = mapped_classes(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, hits] = *it; given)
                {
                    auto segment = storage.array(index).segment(column, colsize);
                    if (should_drop)
                    {
                        segment.setConstant(+0.0);
                    }
                    else
                    {
                        segment.array() = 2.0 * hits.array().template cast<scalar_t>() - 1.0;
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
    };

    ///
    /// \brief
    ///
    class scalar_identity_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        scalar_identity_t(const memory_dataset_t& dataset) :
            base_elemwise_generator_t(dataset)
        {
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            return dataset().feature(mapped_original(ifeature));
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_scalar(dataset(), struct2scalar::off);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t, scalar_map_t storage) const
        {
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = static_cast<scalar_t>(values(0));
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
                        storage(index, column) = static_cast<scalar_t>(values(0));
                    }
                }
                else
                {
                    storage(index, column) = 0.0;
                }
            }
            ++ column;
        }
    };

    ///
    /// \brief
    ///
    class struct_identity_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::structured;

        struct_identity_t(const memory_dataset_t& dataset) :
            base_elemwise_generator_t(dataset)
        {
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            return dataset().feature(mapped_original(ifeature));
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_struct(dataset());
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t, struct_map_t storage) const
        {
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage.array(index) = values.array().template cast<scalar_t>();
                }
                else
                {
                    storage.tensor(index).full(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_flatten(dataset_iterator_t<tscalar, input_rank> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto should_drop = this->should_drop(ifeature);
            const auto colsize = size(mapped_dims(ifeature));
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    auto segment = storage.array(index).segment(column, colsize);
                    if (should_drop)
                    {
                        segment.setConstant(+0.0);
                    }
                    else
                    {
                        segment.array() = values.array().template cast<scalar_t>();
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
    };
}
