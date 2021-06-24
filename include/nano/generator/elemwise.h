#pragma once

#include <nano/generator/elemwise_base.h>

namespace nano
{
    ///
    /// \brief interface for element-wise feature generators.
    ///
    ///     new features are generated as a function of:
    ///         * the original feature and
    ///         * the component index of the original feature.
    ///
    template
    <
        typename tcomputer,
        std::enable_if_t<std::is_base_of_v<base_elemwise_generator_t, tcomputer>, bool> = true
    >
    class NANO_PUBLIC elemwise_generator_t : public tcomputer
    {
    public:

        template <typename... targs>
        explicit elemwise_generator_t(const memory_dataset_t& dataset, targs... args) :
            tcomputer(dataset, args...)
        {
        }

        void select(indices_cmap_t samples, tensor_size_t ifeature, scalar_map_t storage) const override
        {
            if constexpr (tcomputer::generated_type == generator_type::scalar)
            {
                this->iterate1(samples, ifeature, this->mapped_original(ifeature),
                    [&] (const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<tcomputer::input_rank>(data, mask, samples, [&] (auto it)
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
                                if (const auto [index, given, values] = *it; given)
                                {
                                    storage(index) = op(values);
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
                this->iterate1(samples, ifeature, this->mapped_original(ifeature),
                    [&] (const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<tcomputer::input_rank>(data, mask, samples, [&] (auto it)
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
                                if (const auto [index, given, values] = *it; given)
                                {
                                    storage(index) = op(values);
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
                this->iterate1(samples, ifeature, this->mapped_original(ifeature),
                    [&] (const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<tcomputer::input_rank>(data, mask, samples, [&] (auto it)
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
                                if (const auto [index, given, values] = *it; given)
                                {
                                    op(values, storage.vector(index));
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
                this->iterate1(samples, ifeature, this->mapped_original(ifeature),
                    [&] (const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<tcomputer::input_rank>(data, mask, samples, [&] (auto it)
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
                                if (const auto [index, given, values] = *it; given)
                                {
                                    op(values, storage.vector(index));
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
                this->iterate1(samples, ifeature, this->mapped_original(ifeature),
                    [&] (const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<tcomputer::input_rank>(data, mask, samples, [&] (auto it)
                    {
                        const auto should_drop = this->should_drop(ifeature);
                        const auto [op, colsize] = this->process(ifeature);

                        for (; it; ++ it)
                        {
                            if (const auto [index, given, values] = *it; given)
                            {
                                if constexpr (tcomputer::generated_type == generator_type::scalar)
                                {
                                    if (should_drop)
                                    {
                                        storage(index, column) = 0.0;
                                    }
                                    else
                                    {
                                        storage(index, column) = op(values);
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
                                    {   // NOLINT(bugprone-branch-clone)
                                        segment.setConstant(-1.0);
                                        segment(op(values)) = +1.0;
                                    }
                                    else if constexpr (tcomputer::generated_type == generator_type::mclass)
                                    {
                                        op(values, segment);
                                        segment.array() = 2.0 * segment.array() - 1.0;
                                    }
                                    else
                                    {
                                        op(values, segment);
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
