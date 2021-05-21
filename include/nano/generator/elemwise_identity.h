#pragma once

#include <nano/generator/util.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief
    ///
    class sclass_identity_t : public generator_t
    {
    public:

        static constexpr auto input_rank = 1U;
        static constexpr auto generated_type = generator_type::sclass;

        sclass_identity_t(const memory_dataset_t& dataset) :
            generator_t(dataset)
        {
        }

        void fit(indices_cmap_t, execution) override
        {
            m_feature_mapping = select_sclass(dataset(), sclass2binary::off);

            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            return dataset().feature(mapped_original(ifeature));
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor_size_t mapped_classes(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 2);
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

    private:

        // attributes
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };

    ///
    /// \brief
    ///
    class mclass_identity_t : public generator_t
    {
    public:

        static constexpr auto input_rank = 2U;
        static constexpr auto generated_type = generator_type::mclass;

        mclass_identity_t(const memory_dataset_t& dataset) :
            generator_t(dataset)
        {
        }

        void fit(indices_cmap_t, execution) override
        {
            m_feature_mapping = select_mclass(dataset(), mclass2binary::off);

            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            return dataset().feature(mapped_original(ifeature));
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor_size_t mapped_classes(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 2);
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

    private:

        // attributes
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };

    ///
    /// \brief
    ///
    class scalar_identity_t : public generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        scalar_identity_t(const memory_dataset_t& dataset) :
            generator_t(dataset)
        {
        }

        void fit(indices_cmap_t, execution) override
        {
            m_feature_mapping = select_scalar(dataset(), struct2scalar::off);

            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            return dataset().feature(mapped_original(ifeature));
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
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

    private:

        // attributes
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };

    ///
    /// \brief
    ///
    class struct_identity_t : public generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::structured;

        struct_identity_t(const memory_dataset_t& dataset) :
            generator_t(dataset)
        {
        }

        void fit(indices_cmap_t, execution) override
        {
            m_feature_mapping = select_struct(dataset());

            allocate(features());
        }

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            return dataset().feature(mapped_original(ifeature));
        }

        tensor_size_t mapped_original(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return m_feature_mapping(ifeature, 0);
        }

        tensor3d_dims_t mapped_dims(tensor_size_t ifeature) const
        {
            assert(ifeature >= 0 && ifeature < features());
            return make_dims(
                m_feature_mapping(ifeature, 3),
                m_feature_mapping(ifeature, 4),
                m_feature_mapping(ifeature, 5));
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

    private:

        // attributes
        feature_mapping_t   m_feature_mapping;          ///< (feature index, original feature index, ...)
    };
}
