#pragma once

#include <nano/generator/util.h>
#include <nano/generator/pairwise.h>

namespace nano
{
    ///
    /// \brief
    ///
    class product_t : public base_pairwise_generator_t
    {
    public:

        static constexpr auto input_rank1 = 4U;
        static constexpr auto input_rank2 = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        product_t(const memory_dataset_t& dataset, struct2scalar s2s = struct2scalar::off) :
            base_pairwise_generator_t(dataset),
            m_s2s(s2s)
        {
        }

        feature_mapping_t do_fit(indices_cmap_t, execution) override
        {
            const auto mapping = select_scalar(dataset(), m_s2s);

            const auto size = mapping.size<0>();
            auto feature_mapping = feature_mapping_t{size * (size + 1) / 2, 12};

            for (tensor_size_t k = 0, i = 0; i < size; ++ i)
            {
                for (tensor_size_t j = i; j < size; ++ j, ++ k)
                {
                    feature_mapping.array(k).segment(0, 6) = mapping.array(i);
                    feature_mapping.array(k).segment(6, 6) = mapping.array(j);
                }
            }

            return feature_mapping;
        }

        feature_t feature(tensor_size_t ifeature) const override
        {
            assert(ifeature >= 0 && ifeature < features());
            const auto original1 = mapped_original1(ifeature);
            const auto original2 = mapped_original2(ifeature);
            const auto component1 = mapped_component1(ifeature);
            const auto component2 = mapped_component2(ifeature);

            const auto& feature1 = dataset().feature(original1);
            const auto& feature2 = dataset().feature(original2);

            auto name = scat("product(", feature1.name(), "[", component1, "],", feature2.name(), "[", component2, "])");
            return feature_t{std::move(name)}.scalar(feature_type::float64);
        }

        template
        <
            typename tscalar1, typename tscalar2,
            std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true
        >
        void do_select(
            dataset_pairwise_iterator_t<tscalar1, input_rank1, tscalar2, input_rank2> it,
            tensor_size_t ifeature, scalar_map_t storage) const
        {
            const auto component1 = mapped_component1(ifeature);
            const auto component2 = mapped_component2(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                {
                    storage(index) = make_value(values1(component1), values2(component2));
                }
                else
                {
                    storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                }
            }
        }

        template
        <
            typename tscalar1, typename tscalar2,
            std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true
        >
        void do_flatten(
            dataset_pairwise_iterator_t<tscalar1, input_rank1, tscalar2, input_rank2> it,
            tensor_size_t ifeature, tensor2d_map_t storage, tensor_size_t& column) const
        {
            const auto should_drop = this->should_drop(ifeature);
            const auto component1 = mapped_component1(ifeature);
            const auto component2 = mapped_component2(ifeature);
            for (; it; ++ it)
            {
                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                {
                    if (should_drop)
                    {
                        storage(index, column) = +0.0;
                    }
                    else
                    {
                        storage(index, column) = make_value(values1(component1), values2(component2));
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

        // attributes
        struct2scalar       m_s2s{struct2scalar::off};  ///<
    };
}
