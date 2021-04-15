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

void generator_t::allocate(tensor_size_t features)
{
    m_feature_infos.resize(features);
    m_feature_infos.zero();
}

void generator_t::undrop()
{
    m_feature_infos.array() = 0;
}

void generator_t::unshuffle()
{
    m_feature_infos.array() = 0;
}

void generator_t::drop(tensor_size_t feature)
{
    m_feature_infos(feature) = 1;
}

indices_t generator_t::shuffle(tensor_size_t feature)
{
    m_feature_infos(feature) = 2;

    indices_t indices = samples();
    std::shuffle(indices.begin(), indices.end(), make_rng());
    m_shuffle_indices[feature] = indices;
    return indices;
}

dataset_generator_t::dataset_generator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
}

void dataset_generator_t::update()
{
    tensor_size_t columns = 0, features = 0, generators = 0;
    for (const auto& generator : m_generators)
    {
        for (tensor_size_t ifeature = 0; ifeature < generator->features(); ++ ifeature, ++ features)
        {
            switch (const auto feature = generator->feature(ifeature); feature.type())
            {
            case feature_type::sclass:  columns += feature.classes(); break;
            case feature_type::mclass:  columns += feature.classes(); break;
            default:                    columns += size(feature.dims()); break;
            }
        }
        ++ generators;
    }

    m_column_mapping.resize(columns, 3);
    m_feature_mapping.resize(features, 5);
    m_generator_mapping.resize(generators, 1);

    tensor_size_t offset_features = 0, offset_columns = 0, index = 0;
    for (const auto& generator : m_generators)
    {
        const auto old_offset_columns = offset_columns;

        for (tensor_size_t ifeature = 0; ifeature < generator->features(); ++ ifeature, ++ offset_features)
        {
            m_feature_mapping(offset_features, 0) = index;
            m_feature_mapping(offset_features, 1) = ifeature;

            tensor_size_t dim1 = 1, dim2 = 1, dim3 = 1, columns = 0;
            switch (const auto feature = generator->feature(ifeature); feature.type())
            {
            case feature_type::sclass:
                columns = feature.classes();
                break;
            case feature_type::mclass:
                dim1 = feature.classes();
                columns = feature.classes();
                break;
            default:
                dim1 = feature.dims()[0];
                dim2 = feature.dims()[1];
                dim3 = feature.dims()[2];
                columns = size(feature.dims());
                break;
            }
            m_feature_mapping(offset_features, 2) = dim1;
            m_feature_mapping(offset_features, 3) = dim2;
            m_feature_mapping(offset_features, 4) = dim3;

            for (tensor_size_t icolumn = 0; icolumn < columns; ++ icolumn, ++ offset_columns)
            {
                m_column_mapping(offset_columns, 0) = index;
                m_column_mapping(offset_columns, 1) = icolumn;
                m_column_mapping(offset_columns, 2) = offset_features;
            }
        }

        m_generator_mapping(index ++, 0) = offset_columns - old_offset_columns;
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
    return m_column_mapping(column, 2);
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

    auto storage = resize_and_map(buffer, sample_range.size(),
        m_feature_mapping(feature, 2), m_feature_mapping(feature, 3), m_feature_mapping(feature, 4));
    byfeature(feature)->select(m_feature_mapping(feature, 1), sample_range, storage);
    return storage;
}

tensor2d_cmap_t dataset_generator_t::flatten(tensor_range_t sample_range, tensor2d_t& buffer) const
{
    const auto storage = resize_and_map(buffer, sample_range.size(), columns());

    tensor_size_t offset = 0, index = 0;
    for (const auto& generator : m_generators)
    {
        generator->flatten(sample_range, storage, offset);
        offset += m_generator_mapping(index ++, 0);
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

    return m_dataset.visit_target([&] (const feature_t& feature, const auto& data, const auto& mask)
    {
        return loop_samples(data, mask, samples,
            [&] (auto it)
            {
                const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
                for (; it; ++ it)
                {
                    if (const auto [index, given, label] = *it; given)
                    {
                        storage.array(index).setConstant(-1.0);
                        storage.array(index)(static_cast<tensor_size_t>(label)) = +1.0;
                    }
                    else
                    {
                        storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                    }
                }
                return tensor4d_cmap_t{storage};
            },
            [&] (auto it)
            {
                const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
                for (; it; ++ it)
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
            },
            [&] (auto it)
            {
                const auto [dim1, dim2, dim3] = feature.dims();
                const auto storage = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
                for (; it; ++ it)
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
            });
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
        return targets_stats_t{};
    }
    else
    {
        return m_dataset.visit_target([&] (const feature_t& feature, const auto& data, const auto& mask)
        {
            return loop_samples(data, mask, m_samples,
                [&] (auto it) -> targets_stats_t { return sclass_stats_t::make(feature, it); },
                [&] (auto it) -> targets_stats_t { return mclass_stats_t::make(feature, it); },
                [&] (auto it) -> targets_stats_t { return scalar_stats_t::make(feature, it); });
        });
    }
}

tensor1d_t dataset_generator_t::sample_weights(const targets_stats_t& targets_stats) const
{
    if (m_dataset.type() == task_type::unsupervised)
    {
        tensor1d_t weights(m_samples.size());
        weights.full(1.0);
        return weights;
    }
    else
    {
        return m_dataset.visit_target([&] (const feature_t& feature, const auto& data, const auto& mask)
        {
            return loop_samples(data, mask, m_samples,
                [&] (auto it)
                {
                    const auto* pstats = std::get_if<sclass_stats_t>(&targets_stats);
                    critical(
                        pstats == nullptr ||
                        pstats->classes() != feature.classes(),
                        "dataset_generator_t: mis-matching single-label targets statistics, expecting ",
                        feature.classes(), " classes, got ",
                        pstats == nullptr ? tensor_size_t(0) : pstats->classes(), " instead!");

                    return pstats->sample_weights(feature, it);
                },
                [&] (auto it)
                {
                    const auto* pstats = std::get_if<mclass_stats_t>(&targets_stats);
                    critical(
                        pstats == nullptr ||
                        pstats->classes() != feature.classes(),
                        "dataset_generator_t: mis-matching multi-label targets statistics, expecting ",
                        feature.classes(), " classes, got ",
                        pstats == nullptr ? tensor_size_t(0) : pstats->classes(), " instead!");

                    return pstats->sample_weights(feature, it);
                },
                [&] (auto)
                {
                    tensor1d_t weights(m_samples.size());
                    weights.full(1.0);
                    return weights;
                });
        });
    }
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
