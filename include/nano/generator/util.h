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

    ///
    /// \brief utility to write generated single-label features.
    ///
    /// NB: missing or dropped feature values are written as -1.
    ///
    template
    <
        template <typename, size_t> class titerator, typename tscalar, size_t trank,
        typename tgenerator
    >
    void select(titerator<tscalar, trank> it, bool should_drop, sclass_map_t storage, const tgenerator& generator)
    {
        if (should_drop)
        {
            storage.full(-1);
        }
        else
        {
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = generator(it.sample(), values);
                }
                else
                {
                    storage(index) = -1;
                }
            }
        }
    }

    ///
    /// \brief utility to write generated multi-label features.
    ///
    /// NB: missing or dropped feature values are written as -1.
    ///
    template
    <
        template <typename, size_t> class titerator, typename tscalar, size_t trank,
        typename tgenerator
    >
    void select(titerator<tscalar, trank> it, bool should_drop, mclass_map_t storage, const tgenerator& generator)
    {
        if (should_drop)
        {
            storage.full(-1);
        }
        else
        {
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    generator(it.sample(), values, storage.array(index));
                }
                else
                {
                    storage.vector(index) = -1;
                }
            }
        }
    }

    ///
    /// \brief utility to write generated scalar features.
    ///
    /// NB: missing or dropped feature values are written as NaNs.
    ///
    template
    <
        template <typename, size_t> class titerator, typename tscalar, size_t trank,
        typename tgenerator
    >
    void select(titerator<tscalar, trank> it, bool should_drop, scalar_map_t storage, const tgenerator& generator)
    {
        if (should_drop)
        {
            storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
        }
        else
        {
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage(index) = generator(it.sample(), values);
                }
                else
                {
                    storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                }
            }
        }
    }

    ///
    /// \brief utility to write generated structured features.
    ///
    /// NB: missing or dropped feature values are written as NaNs.
    ///
    template
    <
        template <typename, size_t> class titerator, typename tscalar, size_t trank,
        typename tgenerator
    >
    void select(titerator<tscalar, trank> it, bool should_drop, struct_map_t storage, const tgenerator& generator)
    {
        if (should_drop)
        {
            storage.full(std::numeric_limits<scalar_t>::quiet_NaN());
        }
        else
        {
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    generator(it.sample(), values, storage.tensor(index));
                }
                else
                {
                    storage.tensor(index).full(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
        }
    }
}
