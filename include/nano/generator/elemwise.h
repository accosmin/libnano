#pragma once

#include <nano/generator.h>

namespace nano
{
    ///
    /// \brief interface for element-wise feature generators.
    ///
    ///     new features are generated as a function of:
    ///         * original feature,
    ///         * component index of the original feature.
    ///
    template <typename tcomputer, std::enable_if_t<std::is_base_of_v<generator_t, tcomputer>, bool> = true>
    class NANO_PUBLIC elemwise_generator_t : public tcomputer
    {
    public:

        template <typename... targs>
        elemwise_generator_t(const memory_dataset_t& dataset, targs... args) :
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
                this->iterate1(samples, ifeature, this->mapped_original(ifeature),
                    [&] (const auto& data, const auto& mask, indices_cmap_t samples)
                {
                    loop_samples<tcomputer::input_rank>(data, mask, samples, [&] (auto it)
                    {
                        this->do_flatten(it, ifeature, storage, column);
                    });
                });
            }
        }
    };
}
