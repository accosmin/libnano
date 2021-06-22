#include <utest/utest.h>
#include <nano/core/numeric.h>
#include <nano/dataset/synthetic.h>
#include <nano/generator/elemwise_identity.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_synthetic)

UTEST_CASE(affine)
{
    const auto targets = tensor_size_t{3};
    const auto samples = tensor_size_t{100};
    const auto features = tensor_size_t{4};

    auto dataset = synthetic_affine_dataset_t{};

    dataset.noise(0);
    dataset.modulo(31);
    dataset.samples(samples);
    dataset.targets(targets);
    dataset.features(features);

    UTEST_REQUIRE_NOTHROW(dataset.load());

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<sclass_identity_t>>();
    generator.add<elemwise_generator_t<mclass_identity_t>>();
    generator.add<elemwise_generator_t<scalar_identity_t>>();
    generator.add<elemwise_generator_t<struct_identity_t>>();
    generator.fit(arange(0, samples), execution::par);

    UTEST_CHECK_EQUAL(generator.target(), feature_t{"Wx+b+eps"}.scalar(feature_type::float64, make_dims(targets, 1, 1)));

    const auto bias = dataset.bias().vector();
    UTEST_REQUIRE_EQUAL(bias.size(), targets);

    const auto weights = dataset.weights().matrix();
    UTEST_REQUIRE_EQUAL(weights.rows(), 14 * features / 4);
    UTEST_REQUIRE_EQUAL(weights.cols(), targets);

    UTEST_CHECK_EQUAL(dataset.features(), features);
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    UTEST_CHECK_EQUAL(dataset.test_samples(), arange(0, 0));
    UTEST_CHECK_EQUAL(dataset.train_samples(), arange(0, samples));

    tensor2d_t inputs;
    tensor4d_t output;
    generator.flatten(arange(0, samples), inputs);
    generator.targets(arange(0, samples), output);

    for (tensor_size_t sample = 0; sample < samples; ++ sample)
    {
        UTEST_CHECK_EIGEN_CLOSE(
            output.vector(sample),
            weights.transpose() * inputs.vector(sample) + bias,
            epsilon1<scalar_t>());
    }
}

UTEST_END_MODULE()
