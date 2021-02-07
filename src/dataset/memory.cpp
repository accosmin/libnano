#include <nano/dataset/memory.h>

using namespace nano;

memory_dataset_t::memory_dataset_t() = default;

task_type memory_dataset_t::type() const
{
    return static_cast<task_type>(m_target.feature());
}

tensor_size_t memory_dataset_t::samples() const
{
    if (m_inputs.empty())
    {
        return 0;
    }
    else
    {
        return m_inputs.begin()->samples();
    }
}

const feature_storage_t& memory_dataset_t::istorage(tensor_size_t feature) const
{
    critical(
        feature < 0 || feature >= static_cast<tensor_size_t>(m_inputs.size()),
        "failed to access input feature: index ", feature, " not in [0, ", m_inputs.size());

    return m_inputs[static_cast<size_t>(feature)];
}

void memory_dataset_t::resize(tensor_size_t samples, const features_t& features, size_t target)
{
    m_inputs.clear();
    m_inputs.shrink_to_fit();
    m_target = feature_storage_t{};

    for (size_t i = 0; i < features.size(); ++ i)
    {
        if (i == target)
        {
            m_target = feature_storage_t{features[i], samples};
        }
        else
        {
            m_inputs.emplace_back(features[i], samples);
        }
    }
}

rfeature_dataset_iterator_t memory_dataset_t::feature_iterator(indices_t samples) const
{
    return std::make_unique<memory_feature_dataset_iterator_t>(*this, std::move(samples));
}

rflatten_dataset_iterator_t memory_dataset_t::flatten_iterator(indices_t samples) const
{
    return std::make_unique<memory_flatten_dataset_iterator_t>(*this, std::move(samples));
}

memory_feature_dataset_iterator_t::memory_feature_dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
}

const indices_t& memory_feature_dataset_iterator_t::samples() const
{
    return m_samples;
}

bool memory_feature_dataset_iterator_t::cache_inputs(int64_t, execution)
{
    // TODO
    return false;
}

bool memory_feature_dataset_iterator_t::cache_targets(int64_t, execution)
{
    // TODO
    return false;
}

bool memory_feature_dataset_iterator_t::cache_inputs(int64_t, indices_cmap_t, execution)
{
    // TODO
    return false;
}

feature_t memory_feature_dataset_iterator_t::target() const
{
    return m_dataset.tstorage().feature();
}

tensor3d_dims_t memory_feature_dataset_iterator_t::target_dims() const\
{
    return m_dataset.tstorage().feature().dims();
}

tensor4d_cmap_t memory_feature_dataset_iterator_t::targets(tensor_range_t samples, tensor4d_t& buffer) const
{
    m_dataset.tstorage().get(m_samples.slice(samples), buffer);
    return buffer.tensor();
}

tensor_size_t memory_feature_dataset_iterator_t::features() const
{
    return static_cast<tensor_size_t>(m_dataset.istorage().size());
}

namespace {

    template <typename top>
    indices_t filter(const std::vector<feature_storage_t>& istorage, const top& op)
    {
        const auto count = std::count_if(istorage.begin(), istorage.end(), op);

        indices_t indices(count);

        tensor_size_t feature = 0, index = 0;
        for (const auto& fs : istorage)
        {
            if (op(fs))
            {
                indices(index ++) = feature;
            }
            ++ feature;
        }

        return indices;
    }
}

indices_t memory_feature_dataset_iterator_t::scalar_features() const
{
    const auto op = [] (const feature_storage_t& fs)
    {
        return  fs.feature().type() != feature_type::sclass &&
                fs.feature().type() != feature_type::mclass;
    };

    return ::filter(m_dataset.istorage(), op);
}

indices_t memory_feature_dataset_iterator_t::sclass_features() const
{
    const auto op = [] (const feature_storage_t& fs)
    {
        return  fs.feature().type() == feature_type::sclass;
    };

    return ::filter(m_dataset.istorage(), op);
}

indices_t memory_feature_dataset_iterator_t::mclass_features() const
{
    const auto op = [] (const feature_storage_t& fs)
    {
        return  fs.feature().type() == feature_type::mclass;
    };

    return ::filter(m_dataset.istorage(), op);
}

feature_t memory_feature_dataset_iterator_t::feature(tensor_size_t feature) const
{
    const auto& fs = m_dataset.istorage(feature);
    return fs.feature();
}

sindices_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t feature, sindices_t& buffer) const
{
    const auto& fs = m_dataset.istorage(feature);
    fs.get(m_samples, buffer);
    return buffer.tensor();
}

mindices_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t feature, mindices_t& buffer) const
{
    const auto& fs = m_dataset.istorage(feature);
    fs.get(m_samples, buffer);
    return buffer.tensor();
}

tensor1d_cmap_t memory_feature_dataset_iterator_t::input(tensor_size_t feature, tensor1d_t& buffer) const
{
    const auto& fs = m_dataset.istorage(feature);
    // FIXME: have feature_storage_t return 1D tensors as well
    tensor4d_t buffer2;
    fs.get(m_samples, buffer2);
    buffer = buffer2.reshape(-1);
    return buffer.tensor();
}

memory_flatten_dataset_iterator_t::memory_flatten_dataset_iterator_t(const memory_dataset_t& dataset, indices_t samples) :
    m_dataset(dataset),
    m_samples(std::move(samples))
{
}

const indices_t& memory_flatten_dataset_iterator_t::samples() const
{
    return m_samples;
}

bool memory_flatten_dataset_iterator_t::cache_inputs(int64_t, execution)
{
    // TODO
    return false;
}

bool memory_flatten_dataset_iterator_t::cache_targets(int64_t, execution)
{
    // TODO
    return false;
}

feature_t memory_flatten_dataset_iterator_t::target() const
{
    return m_dataset.tstorage().feature();
}

tensor3d_dims_t memory_flatten_dataset_iterator_t::target_dims() const\
{
    return m_dataset.tstorage().feature().dims();
}

tensor4d_cmap_t memory_flatten_dataset_iterator_t::targets(tensor_range_t samples, tensor4d_t& buffer) const
{
    m_dataset.tstorage().get(m_samples.slice(samples), buffer);
    return buffer.tensor();
}

