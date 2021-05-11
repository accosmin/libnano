#pragma once

#include <nano/generator.h>

namespace nano
{
    using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

    ///
    /// \brief
    ///
    NANO_PUBLIC feature_mapping_t select_scalar_components(
        const memory_dataset_t&, struct2scalar, const indices_t& feature_indices);

    ///
    /// \brief utilities to filter the given dataset's (or generator's) features by their type.
    ///
    ///     the given operator is called for matching features and their components if applicable like:
    ///     op(feature_t, feature index, component index or -1).
    ///
    template <typename tdataset, typename toperator>
    void call_scalar(const tdataset& dataset, struct2scalar s2s, tensor_size_t ifeature, const toperator& op)
    {
        const auto& feature = dataset.feature(ifeature);
        if (feature.type() != feature_type::mclass &&
            feature.type() != feature_type::sclass)
        {
            const auto components = size(feature.dims());
            if (components == 1)
            {
                op(feature, ifeature, -1);
            }
            else if (s2s == struct2scalar::on)
            {
                for (tensor_size_t icomponent = 0; icomponent < components; ++ icomponent)
                {
                    op(feature, ifeature, icomponent);
                }
            }
        }
    }

    template <typename tdataset, typename toperator>
    void call_struct(const tdataset& dataset, tensor_size_t ifeature, const toperator& op)
    {
        const auto& feature = dataset.feature(ifeature);
        if (feature.type() != feature_type::mclass &&
            feature.type() != feature_type::sclass)
        {
            const auto components = size(feature.dims());
            if (components > 1)
            {
                op(feature, ifeature, -1);
            }
        }
    }

    template <typename tdataset, typename toperator>
    void call_sclass(const tdataset& dataset, sclass2binary s2b, tensor_size_t ifeature, const toperator& op)
    {
        const auto& feature = dataset.feature(ifeature);
        if (feature.type() == feature_type::sclass)
        {
            const auto components = feature.classes();
            if (s2b == sclass2binary::on)
            {
                for (tensor_size_t icomponent = 0; icomponent < components; ++ icomponent)
                {
                    op(feature, ifeature, icomponent);
                }
            }
            else
            {
                op(feature, ifeature, -1);
            }
        }
    }

    template <typename tdataset, typename toperator>
    void call_mclass(const tdataset& dataset, mclass2binary m2b, tensor_size_t ifeature, const toperator& op)
    {
        const auto& feature = dataset.feature(ifeature);
        if (feature.type() == feature_type::mclass)
        {
            const auto components = feature.classes();
            if (m2b == mclass2binary::on)
            {
                for (tensor_size_t icomponent = 0; icomponent < components; ++ icomponent)
                {
                    op(feature, ifeature, icomponent);
                }
            }
            else
            {
                op(feature, ifeature, -1);
            }
        }
    }

    template <typename tdataset, typename toperator>
    void for_each_scalar(const tdataset& dataset, struct2scalar s2s, const toperator& op)
    {
        for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
        {
            call_scalar(dataset, s2s, ifeature, op);
        }
    }

    template <typename tdataset, typename toperator>
    void for_each_struct(const tdataset& dataset, const toperator& op)
    {
        for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
        {
            call_struct(dataset, ifeature, op);
        }
    }

    template <typename tdataset, typename toperator>
    void for_each_sclass(const tdataset& dataset, sclass2binary s2b, const toperator& op)
    {
        for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
        {
            call_sclass(dataset, s2b, ifeature, op);
        }
    }

    template <typename tdataset, typename toperator>
    void for_each_mclass(const tdataset& dataset, mclass2binary m2b, const toperator& op)
    {
        for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
        {
            call_mclass(dataset, m2b, ifeature, op);
        }
    }
}
