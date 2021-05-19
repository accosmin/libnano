#pragma once

#include <nano/generator/util.h>

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
    template <typename tcomputer, std::enable_if_t<std::is_base_of_v<generator_t, tcomputer>, bool> = true>
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
                    [&] (const auto&, const auto& data1, const auto& mask1,
                         const auto&, const auto& data2, const auto& mask2,
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
                            this->do_select(it, ifeature, storage);
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
                    [&] (const auto&, const auto& data1, const auto& mask1,
                         const auto&, const auto& data2, const auto& mask2,
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
                            this->do_select(it, ifeature, storage);
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
                    [&] (const auto&, const auto& data1, const auto& mask1,
                         const auto&, const auto& data2, const auto& mask2,
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
                            this->do_select(it, ifeature, storage);
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
                    [&] (const auto&, const auto& data1, const auto& mask1,
                         const auto&, const auto& data2, const auto& mask2,
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
                            this->do_select(it, ifeature, storage);
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
                    [&] (const auto&, const auto& data1, const auto& mask1,
                         const auto&, const auto& data2, const auto& mask2,
                         indices_cmap_t samples)
                {
                    loop_samples2<tcomputer::input_rank1, tcomputer::input_rank2>(
                        data1, mask1, data2, mask2, samples, [&] (auto it)
                    {
                        this->do_flatten(it, ifeature, storage, column);
                    });
                });
            }
        }
    };
}
