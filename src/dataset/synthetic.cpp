#include <nano/dataset/synthetic.h>
#include <nano/generator/elemwise_identity.h>

using namespace nano;

synthetic_affine_dataset_t::synthetic_affine_dataset_t() = default;

void synthetic_affine_dataset_t::load()
{
    // generate random input features & target
    features_t features;
    for (tensor_size_t ifeature = 0; ifeature < m_features; ++ ifeature)
    {
        feature_t feature;
        switch (ifeature % 4)
        {
        case 0:     feature = feature_t{scat("scalar", ifeature)}.scalar(); break;
        case 1:     feature = feature_t{scat("sclass", ifeature)}.sclass(3U); break;
        case 2:     feature = feature_t{scat("mclass", ifeature)}.mclass(4U); break;
        default:    feature = feature_t{scat("struct", ifeature)}.scalar(feature_type::float64, make_dims(2, 1, 3)); break;
        }
        features.push_back(feature);
    }
    features.push_back(feature_t{"Wx+b+eps"}.scalar(feature_type::float64, make_dims(m_targets, 1, 1)));

    const auto itarget = features.size() - 1U;
    resize(m_samples, features, itarget);

    // populate dataset
    for (tensor_size_t ifeature = 0; ifeature < m_features; ++ ifeature)
    {
        switch (ifeature % 4)
        {
        case 0:
            {
                tensor_mem_t<scalar_t, 1> values(m_samples);
                values.random(-1.0, +1.0);
                for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
                {
                    if ((sample + ifeature) > 0 && (sample + ifeature) % m_modulo == 0)
                    {
                        set(sample, ifeature, values(sample));
                    }
                }
            }
            break;

        case 1:
            {
                tensor_mem_t<int32_t, 1> values(m_samples);
                values.random(0, 2);
                for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
                {
                    if ((sample + ifeature) > 0 && (sample + ifeature) % m_modulo == 0)
                    {
                        set(sample, ifeature, values(sample));
                    }
                }
            }
            break;

        case 2:
            {
                tensor_mem_t<int32_t, 2> values(m_samples, 4);
                values.random(0, 1);
                for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
                {
                    if ((sample + ifeature) > 0 && (sample + ifeature) % m_modulo == 0)
                    {
                        set(sample, ifeature, values.tensor(sample));
                    }
                }
            }
            break;

        default:
            {
                tensor_mem_t<scalar_t, 4> values(m_samples, 2, 1, 3);
                values.random(-1.0, +1.0);
                for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
                {
                    if ((sample + ifeature) > 0 && (sample + ifeature) % m_modulo == 0)
                    {
                        set(sample, ifeature, values.tensor(sample));
                    }
                }
            }
            break;
        }
    }

    // create samples: target = weights * input + bias + noise
    auto generator = dataset_generator_t{*this};
    generator.add<elemwise_generator_t<sclass_identity_t>>();
    generator.add<elemwise_generator_t<mclass_identity_t>>();
    generator.add<elemwise_generator_t<scalar_identity_t>>();
    generator.add<elemwise_generator_t<struct_identity_t>>();
    generator.fit(arange(0, m_samples), execution::par);

    m_bias.resize(m_targets);
    m_weights.resize(generator.columns(), m_targets);

    m_bias.random();
    m_weights.random();

    tensor2d_t inputs;
    generator.flatten(arange(0, m_samples), inputs);

    tensor1d_t target(m_targets);
    for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
    {
        target.vector() = m_weights.matrix().transpose() * inputs.vector(sample) + m_bias.vector();
        target.vector() += m_noise * vector_t::Random(m_bias.size());
        set(sample, static_cast<tensor_size_t>(itarget), target);
    }
}
