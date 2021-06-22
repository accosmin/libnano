#pragma once

#include <nano/generator/pairwise_base.h>

namespace nano
{
    ///
    /// \brief interface for pair-wise feature generators.
    ///
    ///     new features are generated as a function of:
    ///         * original feature1,
    ///         * component index of the original feature1,
    ///         * original feature2,
    ///         * component index of the original feature2.
    ///
    template
    <
        typename tcomputer,
        std::enable_if_t<std::is_base_of_v<base_pairwise_generator_t, tcomputer>, bool> = true
    >
    class NANO_PUBLIC pairwise_generator_t : public tcomputer
    {
    public:

        template <typename... targs>
        pairwise_generator_t(const memory_dataset_t& dataset, targs... args) :
            tcomputer(dataset, args...)
        {
        }

        void select(indices_cmap_t samples, tensor_size_t ifeature, scalar_map_t storage) const override
        {
            if constexpr (tcomputer::generated_type == generator_type::scalar)
            {
                this->iterate2(samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature),
                    [&] (const auto& data1, const auto& mask1,
                         const auto& data2, const auto& mask2,
                         indices_cmap_t samples)
                {
                    loop_samples2<tcomputer::input_rank1, tcomputer::input_rank2>(
                        data1, mask1, data2, mask2, samples, [&] (auto it)
                    {
                        if (this->should_drop(ifeature))
                        {
                            storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                        else
                        {
                            [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
                            for (; it; ++ it)
                            {
                                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                                {
                                    storage(index) = op(values1, values2);
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
            if constexpr (tcomputer::generated_type == generator_type::sclass)
            {
                this->iterate2(samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature),
                    [&] (const auto& data1, const auto& mask1,
                         const auto& data2, const auto& mask2,
                         indices_cmap_t samples)
                {
                    loop_samples2<tcomputer::input_rank1, tcomputer::input_rank2>(
                        data1, mask1, data2, mask2, samples, [&] (auto it)
                    {
                        if (this->should_drop(ifeature))
                        {
                            storage.full(-1);
                        }
                        else
                        {
                            [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
                            for (; it; ++ it)
                            {
                                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                                {
                                    storage(index) = op(values1, values2);
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

        void select(indices_cmap_t samples, tensor_size_t ifeature, mclass_map_t storage) const override
        {
            if constexpr (tcomputer::generated_type == generator_type::mclass)
            {
                this->iterate2(samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature),
                    [&] (const auto& data1, const auto& mask1,
                         const auto& data2, const auto& mask2,
                         indices_cmap_t samples)
                {
                    loop_samples2<tcomputer::input_rank1, tcomputer::input_rank2>(
                        data1, mask1, data2, mask2, samples, [&] (auto it)
                    {
                        if (this->should_drop(ifeature))
                        {
                            storage.full(-1);
                        }
                        else
                        {
                            [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
                            for (; it; ++ it)
                            {
                                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                                {
                                    op(values1, values2, storage.vector(index));
                                }
                                else
                                {
                                    storage.vector(index).setConstant(-1);
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

        void select(indices_cmap_t samples, tensor_size_t ifeature, struct_map_t storage) const override
        {
            if constexpr (tcomputer::generated_type == generator_type::structured)
            {
                this->iterate2(samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature),
                    [&] (const auto& data1, const auto& mask1,
                         const auto& data2, const auto& mask2,
                         indices_cmap_t samples)
                {
                    loop_samples2<tcomputer::input_rank1, tcomputer::input_rank2>(
                        data1, mask1, data2, mask2, samples, [&] (auto it)
                    {
                        if (this->should_drop(ifeature))
                        {
                            storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                        else
                        {
                            [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
                            for (; it; ++ it)
                            {
                                if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                                {
                                    op(values1, values2, storage.vector(index));
                                }
                                else
                                {
                                    storage.tensor(index).full(std::numeric_limits<scalar_t>::quiet_NaN());
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
                this->iterate2(samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature),
                    [&] (const auto& data1, const auto& mask1,
                         const auto& data2, const auto& mask2,
                         indices_cmap_t samples)
                {
                    loop_samples2<tcomputer::input_rank1, tcomputer::input_rank2>(
                        data1, mask1, data2, mask2, samples, [&] (auto it)
                    {
                        const auto should_drop = this->should_drop(ifeature);
                        const auto [op, colsize] = this->process(ifeature);

                        for (; it; ++ it)
                        {
                            if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                            {
                                if constexpr (tcomputer::generated_type == generator_type::scalar)
                                {
                                    if (should_drop)
                                    {
                                        storage(index, column) = 0.0;
                                    }
                                    else
                                    {
                                        storage(index, column) = op(values1, values2);
                                    }
                                }
                                else
                                {
                                    auto segment = storage.vector(index).segment(column, colsize);
                                    if (should_drop)
                                    {
                                        segment.setConstant(+0.0);
                                    }
                                    else if constexpr (tcomputer::generated_type == generator_type::sclass)
                                    {
                                        segment.setConstant(-1.0);
                                        segment(op(values1, values2)) = +1.0;
                                    }
                                    else if constexpr (tcomputer::generated_type == generator_type::mclass)
                                    {
                                        op(values1, values2, segment);
                                        segment.array() = 2.0 * segment.array() - 1.0;
                                    }
                                    else
                                    {
                                        op(values1, values2, segment);
                                    }
                                }
                            }
                            else
                            {
                                if constexpr (tcomputer::generated_type == generator_type::scalar)
                                {
                                    storage(index, column) = 0.0;
                                }
                                else
                                {
                                    auto segment = storage.array(index).segment(column, colsize);
                                    segment.setConstant(+0.0);
                                }
                            }
                        }
                        column += colsize;
                    });
                });
            }
        }
    };
}
