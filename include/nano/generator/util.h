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
        feature_mapping_t select(const tdataset& dataset, const indices_t& feature_indices, const toperator& callback)
        {
            tensor_size_t count = 0;
            if (feature_indices.size() > 0)
            {
                for (const auto ifeature : feature_indices)
                {
                    callback(dataset, ifeature, [&] (const auto&, auto, auto) { ++ count; });
                }
            }
            else
            {
                for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
                {
                    callback(dataset, ifeature, [&] (const auto&, auto, auto) { ++ count; });
                }
            }

            feature_mapping_t mapping(count, 6);

            tensor_size_t k = 0;
            const auto update_mapping = [&] (tensor_size_t ifeature)
            {
                callback(dataset, ifeature, [&] (const auto& feature, tensor_size_t original, tensor_size_t component)
                {
                    mapping(k, 0) = original;
                    mapping(k, 1) = component;
                    mapping(k, 2) = feature.classes();
                    mapping(k, 3) = std::get<0>(feature.dims());
                    mapping(k, 4) = std::get<1>(feature.dims());
                    mapping(k, 5) = std::get<2>(feature.dims());
                    ++ k;
                });
            };

            if (feature_indices.size() > 0)
            {
                for (const auto ifeature : feature_indices)
                {
                    update_mapping(ifeature);
                }
            }
            else
            {
                for (tensor_size_t ifeature = 0, features = dataset.features(); ifeature < features; ++ ifeature)
                {
                    update_mapping(ifeature);
                }
            }
            return mapping;
        }
    }

    template <typename tdataset>
    feature_mapping_t select_sclass(
        const tdataset& dataset, sclass2binary s2b, const indices_t& feature_indices = indices_t{})
    {
        return detail::select(dataset, feature_indices, [&] (const auto&, tensor_size_t ifeature, const auto& op)
        {
            call_sclass(dataset, s2b, ifeature, op);
        });
    }

    template <typename tdataset>
    feature_mapping_t select_mclass(
        const tdataset& dataset, mclass2binary m2b, const indices_t& feature_indices = indices_t{})
    {
        return detail::select(dataset, feature_indices, [&] (const auto&, tensor_size_t ifeature, const auto& op)
        {
            call_mclass(dataset, m2b, ifeature, op);
        });
    }

    template <typename tdataset>
    feature_mapping_t select_scalar(
        const tdataset& dataset, struct2scalar s2s, const indices_t& feature_indices = indices_t{})
    {
        return detail::select(dataset, feature_indices, [&] (const auto&, tensor_size_t ifeature, const auto& op)
        {
            call_scalar(dataset, s2s, ifeature, op);
        });
    }

    template <typename tdataset>
    feature_mapping_t select_struct(
        const tdataset& dataset, const indices_t& feature_indices = indices_t{})
    {
        return detail::select(dataset, feature_indices, [&] (const auto&, tensor_size_t ifeature, const auto& op)
        {
            call_struct(dataset, ifeature, op);
        });
    }
}
