#pragma once

#include <nano/generator/util.h>
#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief log-scale the feature values while preserving their signs.
    ///
    class slog1p_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        slog1p_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off) :
            base_elemwise_generator_t(dataset),
            m_s2s(s2s)
        {
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_scalar(dataset(), m_s2s);
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original = mapped_original(ifeature);
            const auto component = mapped_component(ifeature);

            const auto& feature = dataset().feature(original);
            return feature_t{scat("slog1p(", feature.name(), "[", component, "])")}.scalar(feature_type::float64);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, scalar_map_t storage) const
        {
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = make_value(values(component));
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
            const auto component = mapped_component(ifeature);
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
                        storage(index, column) = make_value(values(component));
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
    };

    ///
    /// \brief
    ///
    class sign_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        sign_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off) :
            base_elemwise_generator_t(dataset),
            m_s2s(s2s)
        {
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_scalar(dataset(), m_s2s);
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original = mapped_original(ifeature);
            const auto component = mapped_component(ifeature);

            const auto& feature = dataset().feature(original);
            return feature_t{scat("sign(", feature.name(), "[", component, "])")}.scalar(feature_type::float64);
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, scalar_map_t storage) const
        {
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = make_value(values(component));
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
            const auto component = mapped_component(ifeature);
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
                        storage(index, column) = make_value(values(component));
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
    };

    ///
    /// \brief
    ///
    class sign_class_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::sclass;

        sign_class_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off) :
            base_elemwise_generator_t(dataset),
            m_s2s(s2s)
        {
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            return select_scalar(dataset(), m_s2s);
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original = mapped_original(ifeature);
            const auto component = mapped_component(ifeature);

            const auto& feature = dataset().feature(original);
            return feature_t{scat("sign_class(", feature.name(), "[", component, "])")}.sclass(strings_t{"neg", "pos"});
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        void do_select(dataset_iterator_t<tscalar, input_rank> it, tensor_size_t ifeature, sclass_map_t storage) const
        {
            const auto component = mapped_component(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = make_value(values(component));
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
            const auto component = mapped_component(ifeature);
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
                        const auto label = make_value(values(component));
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
    };
}
