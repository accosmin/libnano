#pragma once

#include <nano/generator.h>

namespace nano
{
    // (original feature index, feature component, ...)
    using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

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

    namespace detail
    {
        template <typename tdataset, typename toperator>
        feature_mapping_t select(const tdataset& dataset, const toperator& callback)
        {
            tensor_size_t count = 0;
            for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
            {
                callback(dataset, ifeature, [&] (const auto&, auto, auto) { ++ count; });
            }

            feature_mapping_t mapping(count, 3);
            for (tensor_size_t k = 0, ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
            {
                callback(dataset, ifeature, [&] (const auto&, tensor_size_t original, tensor_size_t component)
                {
                    mapping(k, 0) = original;
                    mapping(k, 1) = std::max(component, tensor_size_t{0});
                    mapping(k, 2) = component;
                    ++ k;
                });
            }
            return mapping;
        }
    }

    template <typename tdataset>
    feature_mapping_t select_sclass(const tdataset& dataset, sclass2binary s2b)
    {
        return detail::select(dataset, [&] (const auto&, tensor_size_t ifeature, const auto& op)
        {
            call_sclass(dataset, s2b, ifeature, op);
        });
    }

    template <typename tdataset>
    feature_mapping_t select_mclass(const tdataset& dataset, mclass2binary m2b)
    {
        return detail::select(dataset, [&] (const auto&, tensor_size_t ifeature, const auto& op)
        {
            call_mclass(dataset, m2b, ifeature, op);
        });
    }

    template <typename tdataset>
    feature_mapping_t select_scalar(const tdataset& dataset, struct2scalar s2s)
    {
        return detail::select(dataset, [&] (const auto&, tensor_size_t ifeature, const auto& op)
        {
            call_scalar(dataset, s2s, ifeature, op);
        });
    }

    template <typename tdataset>
    feature_mapping_t select_struct(const tdataset& dataset)
    {
        return detail::select(dataset, [&] (const auto&, tensor_size_t ifeature, const auto& op)
        {
            call_struct(dataset, ifeature, op);
        });
    }
}
