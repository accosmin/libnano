#include <mutex>
#include <nano/tpool.h>
#include <nano/dataset/iterator.h>
#include <nano/dataset/generators.h>

using namespace nano;

template <typename tscalar, size_t trank, typename... tindices>
static auto resize_and_map(tensor_mem_t<tscalar, trank>& buffer, tindices... dims)
{
    if (buffer.size() < ::nano::size(make_dims(dims...)))
    {
        buffer.resize(dims...);
    }
    return map_tensor(buffer.data(), dims...);
}

generator_factory_t& generator_t::all()
{
    static generator_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        //manager.add<identity_generator_t>(
        //    "identity",
        //    "original input features");
    });

    return manager;
}

generator_t::generator_t(const memory_dataset_t& dataset, const indices_t& samples) :
    m_dataset(dataset),
    m_samples(samples)
{
}

void generator_t::preprocess(execution)
{
}

void generator_t::select(tensor_size_t feature, tensor_range_t, sclass_map_t) const
{
    critical0("generator_t: unhandled single-label feature <", feature, ":", this->feature(feature), ">!");
}

void generator_t::select(tensor_size_t feature, tensor_range_t, mclass_map_t) const
{
    critical0("generator_t: unhandled multi-label feature <", feature, ":", this->feature(feature), ">!");
}

void generator_t::select(tensor_size_t feature, tensor_range_t, scalar_map_t) const
{
    critical0("generator_t: unhandled scalar feature <", feature, ":", this->feature(feature), ">!");
}

void generator_t::select(tensor_size_t feature, tensor_range_t, struct_map_t) const
{
    critical0("generator_t: unhandled structured feature <", feature, ":", this->feature(feature), ">!");
}

dataset_generator_t::dataset_generator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
}

void dataset_generator_t::update()
{
    tensor_size_t columns = 0, features = 0;
    for (const auto& generator : m_generators)
    {
        columns += generator->columns();
        features += generator->features();
    }

    m_column_mapping.resize(columns, 2);
    m_feature_mapping.resize(features, 5);

    tensor_size_t offset_features = 0, offset_columns = 0, index = 0;
    for (const auto& generator : m_generators)
    {
        const auto columns = generator->columns();
        for (tensor_size_t column = 0; column < columns; ++ column)
        {
            m_column_mapping(offset_columns + column, 0) = index;
            m_column_mapping(offset_columns + column, 1) = column;
        }

        const auto features = generator->features();
        for (tensor_size_t feature = 0; feature < features; ++ feature)
        {
            m_feature_mapping(offset_features + feature, 0) = index;
            m_feature_mapping(offset_features + feature, 1) = feature;

            tensor_size_t dim1 = 1, dim2 = 1, dim3 = 1;
            switch (const auto feature_ = generator->feature(feature); feature_.type())
            {
            case feature_type::sclass:
                break;
            case feature_type::mclass:
                dim1 = feature_.classes();
                break;
            default:
                dim1 = feature_.dims()[0];
                dim2 = feature_.dims()[1];
                dim3 = feature_.dims()[2];
                break;
            }
            m_feature_mapping(offset_features + feature, 2) = dim1;
            m_feature_mapping(offset_features + feature, 3) = dim2;
            m_feature_mapping(offset_features + feature, 4) = dim3;
        }

        ++ index;
        offset_columns += columns;
        offset_features += features;
    }
}

tensor_size_t dataset_generator_t::features() const
{
    return m_feature_mapping.size<0>();
}

feature_t dataset_generator_t::feature(tensor_size_t feature) const
{
    return byfeature(feature)->feature(m_feature_mapping(feature, 1));
}

tensor_size_t dataset_generator_t::columns() const
{
    return m_column_mapping.size<0>();
}

tensor_size_t dataset_generator_t::column2feature(tensor_size_t column) const
{
    return bycolumn(column)->column2feature(m_column_mapping(column, 1));
}

sclass_cmap_t dataset_generator_t::select(tensor_size_t feature, sclass_mem_t& buffer) const
{
    return select(feature, make_range(0, m_samples.size()), buffer);
}

mclass_cmap_t dataset_generator_t::select(tensor_size_t feature, mclass_mem_t& buffer) const
{
    return select(feature, make_range(0, m_samples.size()), buffer);
}

scalar_cmap_t dataset_generator_t::select(tensor_size_t feature, scalar_mem_t& buffer) const
{
    return select(feature, make_range(0, m_samples.size()), buffer);
}

struct_cmap_t dataset_generator_t::select(tensor_size_t feature, struct_mem_t& buffer) const
{
    return select(feature, make_range(0, m_samples.size()), buffer);
}

sclass_cmap_t dataset_generator_t::select(tensor_size_t feature, tensor_range_t sample_range, sclass_mem_t& buffer) const
{
    assert(sample_range.begin() >= 0 && sample_range.end() <= m_samples.size());

    auto storage = resize_and_map(buffer, sample_range.size());
    byfeature(feature)->select(m_feature_mapping(feature, 1), sample_range, storage);
    return storage;
}

mclass_cmap_t dataset_generator_t::select(tensor_size_t feature, tensor_range_t sample_range, mclass_mem_t& buffer) const
{
    assert(sample_range.begin() >= 0 && sample_range.end() <= m_samples.size());

    auto storage = resize_and_map(buffer, sample_range.size(), m_feature_mapping(feature, 2));
    byfeature(feature)->select(m_feature_mapping(feature, 1), sample_range, storage);
    return storage;
}

scalar_cmap_t dataset_generator_t::select(tensor_size_t feature, tensor_range_t sample_range, scalar_mem_t& buffer) const
{
    assert(sample_range.begin() >= 0 && sample_range.end() <= m_samples.size());

    auto storage = resize_and_map(buffer, sample_range.size());
    byfeature(feature)->select(m_feature_mapping(feature, 1), sample_range, storage);
    return storage;
}

struct_cmap_t dataset_generator_t::select(tensor_size_t feature, tensor_range_t sample_range, struct_mem_t& buffer) const
{
    assert(sample_range.begin() >= 0 && sample_range.end() <= m_samples.size());

    auto storage = resize_and_map(buffer, sample_range.size(), m_feature_mapping(feature, 2), m_feature_mapping(feature, 3), m_feature_mapping(feature, 4));
    byfeature(feature)->select(m_feature_mapping(feature, 1), sample_range, storage);
    return storage;
}

tensor2d_cmap_t dataset_generator_t::flatten(tensor_range_t sample_range, tensor2d_t& buffer) const
{
    const auto storage = resize_and_map(buffer, sample_range.size(), columns());

    tensor_size_t offset = 0;
    for (const auto& generator : m_generators)
    {
        generator->flatten(sample_range, storage, offset);
        offset += generator->columns();
    }
    return storage;
}

feature_t dataset_generator_t::target() const
{
    switch (m_dataset.type())
    {
    case task_type::unsupervised:
        return feature_t{};

    default:
        return m_dataset.visit_target([] (const feature_t& feature, const auto&, const auto&)
        {
            return feature;
        });
    }
}

tensor3d_dims_t dataset_generator_t::target_dims() const
{
    switch (m_dataset.type())
    {
    case task_type::unsupervised:
        return make_dims(0, 0, 0);

    default:
        return m_dataset.visit_target([] (const feature_t& feature, const auto&, const auto&)
        {
            switch (feature.type())
            {
            case feature_type::sclass:  return make_dims(feature.classes(), 1, 1);
            case feature_type::mclass:  return make_dims(feature.classes(), 1, 1);
            default:                    return feature.dims();
            }
        });
    }
}

tensor4d_cmap_t dataset_generator_t::targets(tensor_range_t sample_range, tensor4d_t& buffer) const
{
    if (m_dataset.type() == task_type::unsupervised)
    {
        critical0("dataset_generator_t: targets are not available for unsupervised datasets!");
    }

    const auto samples = m_samples.slice(sample_range);

    return m_dataset.visit_target([&] (const feature_t& feature, const auto& tensor, const auto& mask)
    {
        if constexpr (tensor.rank() == 1)
        {
            const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
            for (auto it = make_iterator(tensor, mask, samples); it; ++ it)
            {
                if (const auto [index, given, label] = *it; given)
                {
                    storage.array(index).setConstant(-1.0);
                    storage.array(index)(label) = +1.0;
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_cmap_t{storage};
        }
        else if constexpr (tensor.rank() == 2)
        {
            const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
            for (auto it = make_iterator(tensor, mask, samples); it; ++ it)
            {
                if (const auto [index, given, hits] = *it; given)
                {
                    storage.array(index) = hits.array().template cast<scalar_t>() * 2.0 - 1.0;
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_cmap_t{storage};
        }
        else
        {
            const auto [dim1, dim2, dim3] = feature.dims();
            const auto storage = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
            for (auto it = make_iterator(tensor, mask, samples); it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage.array(index) = values.array().template cast<scalar_t>();
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_cmap_t{storage};
        }
    });
}

select_stats_t dataset_generator_t::select_stats(execution) const
{
    std::vector<tensor_size_t> sclasss, mclasss, scalars, structs;
    for (tensor_size_t i = 0, size = features(); i < size; ++ i)
    {
        switch (const auto& feature = this->feature(i); feature.type())
        {
        case feature_type::sclass:
            sclasss.push_back(i);
            break;

        case feature_type::mclass:
            mclasss.push_back(i);
            break;

        default:
            (::nano::size(feature.dims()) > 1 ? structs : scalars).push_back(i);
            break;
        }
    }

    select_stats_t stats;
    stats.m_sclass_features = map_tensor(sclasss.data(), make_dims(static_cast<tensor_size_t>(sclasss.size())));
    stats.m_mclass_features = map_tensor(mclasss.data(), make_dims(static_cast<tensor_size_t>(mclasss.size())));
    stats.m_scalar_features = map_tensor(scalars.data(), make_dims(static_cast<tensor_size_t>(scalars.size())));
    stats.m_struct_features = map_tensor(structs.data(), make_dims(static_cast<tensor_size_t>(structs.size())));
    return stats;
}

flatten_stats_t dataset_generator_t::flatten_stats(execution ex, tensor_size_t batch) const
{
    switch (ex)
    {
    case execution::par:
        {
            std::vector<tensor2d_t> buffers(tpool_t::size());
            std::vector<flatten_stats_t> stats(tpool_t::size(), flatten_stats_t{columns()});

            loopr(m_samples.size(), batch, [&] (tensor_size_t begin, tensor_size_t end, size_t tnum)
            {
                const auto data = flatten(make_range(begin, end), buffers[tnum]);
                for (tensor_size_t i = begin; i < end; ++ i)
                {
                    stats[tnum] += data.array(i - begin);
                }
            });

            for (size_t i = 1; i < stats.size(); ++ i)
            {
                stats[0] += stats[i];
            }
            return stats[0].done();
        }

    default:
        {
            tensor2d_t buffer;
            flatten_stats_t stats{columns()};

            for (tensor_size_t begin = 0, size = m_samples.size(); begin < size; begin += batch)
            {
                const auto end = std::min(begin + batch, size);
                const auto data = flatten(make_range(begin, end), buffer);
                for (tensor_size_t i = begin; i < end; ++ i)
                {
                    stats += data.array(i - begin);
                }
            }

            return stats.done();
        }
    }
}

targets_stats_t dataset_generator_t::targets_stats(execution, tensor_size_t) const
{
    if (m_dataset.type() == task_type::unsupervised)
    {
        critical0("dataset_generator_t: target statistics are not available for unsupervised datasets!");
    }

    return m_dataset.visit_target([&] (const feature_t& feature, const auto& tensor, const auto& mask)
    {
        if constexpr (tensor.rank() == 1)
        {
            sclass_stats_t stats{feature.classes()};
            for (auto it = make_iterator(tensor, mask, m_samples); it; ++ it)
            {
                if (const auto [index, given, label] = *it; given)
                {
                    stats += label;
                }
            }
            return targets_stats_t{stats};
        }
        else if constexpr (tensor.rank() == 2)
        {
            sclass_stats_t stats{feature.classes()};
            for (auto it = make_iterator(tensor, mask, m_samples); it; ++ it)
            {
                if (const auto [index, given, hits] = *it; given)
                {
                    stats += hits;
                }
            }
            return targets_stats_t{stats};
        }
        else
        {
            scalar_stats_t stats{nano::size(feature.dims())};
            for (auto it = make_iterator(tensor, mask, m_samples); it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    stats += values.array().template cast<scalar_t>();
                }
            }
            return targets_stats_t{stats.done()};
        }
    });
}

tensor1d_t dataset_generator_t::sample_weights(const targets_stats_t& targets_stats) const
{
    if (m_dataset.type() == task_type::unsupervised)
    {
        critical0("dataset_generator_t: sample weights are not available for unsupervised datasets!");
    }

    tensor1d_t weights(m_samples.size());
    weights.constant(1.0);

    return m_dataset.visit_target([&] (const feature_t& feature, const auto& tensor, const auto& mask)
    {
        if constexpr (tensor.rank() == 1)
        {
            const auto* pstats = std::get_if<sclass_stats_t>(&targets_stats);
            critical(
                pstats == nullptr ||
                pstats->m_class_counts.size() != feature.classes(),
                "dataset_generator_t: mis-matching targets class statistics, expecting ",
                feature.classes(), " classes, got ",
                pstats == nullptr ? tensor_size_t(0) : pstats->m_class_counts.size(), " instead!");

            const vector_t class_weights =
                static_cast<scalar_t>(m_samples.size()) /
                static_cast<scalar_t>(feature.classes()) /
                pstats->m_class_counts.array().cast<scalar_t>().max(1.0);

            for (auto it = make_iterator(tensor, mask, m_samples); it; ++ it)
            {
                if (const auto [index, given, label] = *it; given)
                {
                    weights(index) = class_weights(label);
                }
            }
            return weights;
        }
        else if constexpr (tensor.rank() == 2)
        {
            const auto* pstats = std::get_if<sclass_stats_t>(&targets_stats);
            critical(
                pstats == nullptr ||
                pstats->m_class_counts.size() != feature.classes(),
                "dataset_generator_t: mis-matching targets class statistics, expecting ",
                feature.classes(), " classes, got ",
                pstats == nullptr ? tensor_size_t(0) : pstats->m_class_counts.size(), " instead!");

            // TODO: is it possible to weight samples similarly to the single-label multi-class case?!
            return weights;
        }
        else
        {
            return weights;
        }
    });
}

void dataset_generator_t::undrop() const
{
    for (const auto& generator : m_generators)
    {
        generator->undrop();
    }
}

void dataset_generator_t::unshuffle() const
{
    for (const auto& generator : m_generators)
    {
        generator->unshuffle();
    }
}

void dataset_generator_t::drop(tensor_size_t feature) const
{
    byfeature(feature)->drop(m_feature_mapping(feature, 1));
}

indices_t dataset_generator_t::shuffle(tensor_size_t feature) const
{
    return byfeature(feature)->shuffle(m_feature_mapping(feature, 1));
}
