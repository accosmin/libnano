#pragma once

#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief
    ///
    class NANO_PUBLIC sclass_identity_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 1U;
        static constexpr auto generated_type = generator_type::sclass;

        explicit sclass_identity_t(const memory_dataset_t& dataset);

        feature_t feature(tensor_size_t ifeature) const override;
        feature_mapping_t do_fit(indices_cmap_t, execution) override;

        auto process(tensor_size_t ifeature) const
        {
            const auto colsize = mapped_classes(ifeature);
            const auto process = [=] (const auto& label)
            {
                return static_cast<int32_t>(label);
            };

            return std::make_tuple(process, colsize);
        }
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC mclass_identity_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 2U;
        static constexpr auto generated_type = generator_type::mclass;

        explicit mclass_identity_t(const memory_dataset_t& dataset);

        feature_t feature(tensor_size_t ifeature) const override;
        feature_mapping_t do_fit(indices_cmap_t, execution) override;

        auto process(tensor_size_t ifeature) const
        {
            const auto colsize = mapped_classes(ifeature);
            const auto process = [=] (const auto& hits, auto&& storage)
            {
                this->copy(hits, storage);
            };

            return std::make_tuple(process, colsize);
        }

    private:

        template <typename thits, typename tstorage>
        static void copy(const thits& hits, tstorage& storage)
        {
            storage = hits.array().template cast<typename tstorage::Scalar>();
        }
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC scalar_identity_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::scalar;

        explicit scalar_identity_t(const memory_dataset_t& dataset);

        feature_t feature(tensor_size_t ifeature) const override;
        feature_mapping_t do_fit(indices_cmap_t, execution) override;

        auto process(tensor_size_t) const
        {
            const auto colsize = tensor_size_t{1};
            const auto process = [=] (const auto& values)
            {
                return static_cast<scalar_t>(values(0));
            };

            return std::make_tuple(process, colsize);
        }
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC struct_identity_t : public base_elemwise_generator_t
    {
    public:

        static constexpr auto input_rank = 4U;
        static constexpr auto generated_type = generator_type::structured;

        explicit struct_identity_t(const memory_dataset_t& dataset);

        feature_t feature(tensor_size_t ifeature) const override;
        feature_mapping_t do_fit(indices_cmap_t, execution) override;

        auto process(tensor_size_t ifeature) const
        {
            const auto colsize = size(mapped_dims(ifeature));
            const auto process = [=] (const auto& values, auto&& storage)
            {
                storage = values.array().template cast<scalar_t>();
            };

            return std::make_tuple(process, colsize);
        }
    };
}
