#include <utest/utest.h>
#include <nano/core/numeric.h>
#include <nano/dataset/synth_affine.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_synthetic)

UTEST_CASE(affine)
{
    auto dataset = synthetic_affine_dataset_t{};

    dataset.noise(0);
    dataset.modulo(2);
    dataset.samples(100);
    dataset.idims(make_dims(7, 1, 1));
    dataset.tdims(make_dims(3, 1, 1));

    UTEST_REQUIRE_NOTHROW(dataset.load());

    const auto tfeature = dataset.target();
    UTEST_CHECK(!tfeature.discrete());
    UTEST_CHECK(!tfeature.optional());

    const auto& bias = dataset.bias();
    UTEST_REQUIRE_EQUAL(bias.size(), 3);

    const auto& weights = dataset.weights();
    UTEST_REQUIRE_EQUAL(weights.rows(), 7);
    UTEST_REQUIRE_EQUAL(weights.cols(), 3);
    for (tensor_size_t row = 0; row < weights.rows(); ++ row)
    {
        if ((row % dataset.modulo()) > 0)
        {
            UTEST_CHECK_EIGEN_CLOSE(weights.row(row).transpose(), vector_t::Zero(weights.cols()), epsilon1<scalar_t>());
        }
    }

    UTEST_CHECK_EQUAL(dataset.samples(), 100);
    UTEST_CHECK_EQUAL(dataset.test_samples(), arange(0, 0));
    UTEST_CHECK_EQUAL(dataset.train_samples(), arange(0, 100));

    const auto inputs = dataset.inputs(arange(0, 100));
    const auto targets = dataset.targets(arange(0, 100));

    UTEST_CHECK_EQUAL(inputs.dims(), nano::make_dims(100, 7, 1, 1));
    UTEST_CHECK_EQUAL(targets.dims(), nano::make_dims(100, 3, 1, 1));

    for (tensor_size_t s = 0; s < 100; ++ s)
    {
        UTEST_CHECK_EIGEN_CLOSE(targets.vector(s), weights.transpose() * inputs.vector(s) + bias, epsilon1<scalar_t>());
    }
}

UTEST_END_MODULE()
