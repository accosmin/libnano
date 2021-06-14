#include <nano/generator/select.h>
#include <nano/generator/elemwise_histogram.h>

using namespace nano;

histogram_medians_t::histogram_medians_t(const memory_dataset_t& dataset, struct2scalar s2s) :
    base_elemwise_generator_t(dataset),
    m_s2s(s2s)
{
}

feature_mapping_t histogram_medians_t::do_fit(indices_cmap_t samples, execution)
{
    const auto mapping = select_scalar(dataset(), m_s2s);

    m_histograms.clear();

    for (tensor_size_t ifeature = 0; ifeature < mapping.size<0>(); ++ ifeature)
    {
        const auto original = mapping(ifeature, 0);
        const auto component = std::max(mapping(ifeature, 1), tensor_size_t{0});

        std::vector<scalar_t> allvalues;
        dataset().visit_inputs(original, [&] (const auto&, const auto& data, const auto& mask)
        {
            loop_samples<input_rank>(data, mask, samples, [&] (auto it)
            {
                for (; it; ++ it)
                {
                    if (const auto [index, given, values] = *it; given)
                    {
                        allvalues.push_back(static_cast<scalar_t>(values(component)));
                    }
                }
            });
        });

        m_histograms.push_back(make_histogram(allvalues));
    }

    return mapping;
}

feature_t histogram_medians_t::feature(tensor_size_t ifeature) const
{
    const auto original = mapped_original(ifeature);
    const auto component = mapped_component(ifeature);

    const auto& feature = dataset().feature(original);
    return feature_t{scat(suffix(), "(", feature.name(), "[", component, "])")}.scalar(feature_type::float64);
}
