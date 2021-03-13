#include <utest/utest.h>
#include "fixture/memfixed.h"

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_memfixed)

UTEST_CASE(check_samples)
{
    auto dataset = fixture_dataset_t{};

    dataset.resize(nano::make_dims(100, 3, 10, 10), nano::make_dims(100, 10, 1, 1));
    UTEST_REQUIRE_NOTHROW(dataset.load());

    UTEST_CHECK_EQUAL(dataset.samples(), 100);
    {
        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 0);

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 100);
        UTEST_CHECK_EQUAL(train_samples, ::nano::arange(0, 100));
    }
    {
        dataset.testing(make_range(0, 10));
        dataset.testing(make_range(20, 50));

        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 40);
        UTEST_CHECK_EQUAL(test_samples.slice(0, 10), ::nano::arange(0, 10));
        UTEST_CHECK_EQUAL(test_samples.slice(10, 40), ::nano::arange(20, 50));

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 60);
        UTEST_CHECK(train_samples.slice(0, 10) == ::nano::arange(10, 20));
        UTEST_CHECK(train_samples.slice(10, 60) == ::nano::arange(50, 100));
    }
    {
        dataset.no_testing();

        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 0);

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 100);
        UTEST_CHECK_EQUAL(train_samples, ::nano::arange(0, 100));
    }
}

UTEST_CASE(check_inputs_targets)
{
    auto dataset = fixture_dataset_t{};

    dataset.resize(nano::make_dims(100, 3, 10, 10), nano::make_dims(100, 10, 1, 1));
    UTEST_REQUIRE_NOTHROW(dataset.load());

    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"feature_0_0_0"});
    UTEST_CHECK_EQUAL(dataset.feature(31), feature_t{"feature_0_3_1"});
    UTEST_CHECK_EQUAL(dataset.feature(257), feature_t{"feature_2_5_7"});
    UTEST_CHECK_EQUAL(dataset.target(), feature_t{"fixture"});

    const auto inputs = dataset.inputs(::nano::arange(10, 70));
    const auto inputs0 = dataset.inputs(::nano::arange(10, 70), 13);
    const auto inputsX = dataset.inputs(::nano::arange(10, 70), make_tensor<tensor_size_t>(make_dims(3), 13, 17, 201));
    const auto targets = dataset.targets(::nano::arange(10, 70));

    UTEST_CHECK_EQUAL(inputs.dims(), nano::make_dims(60, 3, 10, 10));
    UTEST_CHECK_EQUAL(inputs0.dims(), nano::make_dims(60));
    UTEST_CHECK_EQUAL(inputsX.dims(), nano::make_dims(60, 3));
    UTEST_CHECK_EQUAL(targets.dims(), nano::make_dims(60, 10, 1, 1));

    for (tensor_size_t s = 0; s < 60; ++ s)
    {
        const auto row = s + 10;

        const auto imatrix = inputs.reshape(inputs.size<0>(), -1);
        for (tensor_size_t f = 0; f < 300; ++ f)
        {
            UTEST_CHECK_EQUAL(imatrix(s, f), fixture_dataset_t::value(row, f));
        }

        UTEST_CHECK_EQUAL(inputs0(s), fixture_dataset_t::value(row, 13));
        UTEST_CHECK_EQUAL(inputsX(s, 0), fixture_dataset_t::value(row, 13));
        UTEST_CHECK_EQUAL(inputsX(s, 1), fixture_dataset_t::value(row, 17));
        UTEST_CHECK_EQUAL(inputsX(s, 2), fixture_dataset_t::value(row, 201));

        UTEST_CHECK_CLOSE(targets.vector(s).minCoeff(), static_cast<scalar_t>(-row), 1e-8);
        UTEST_CHECK_CLOSE(targets.vector(s).maxCoeff(), static_cast<scalar_t>(-row), 1e-8);
    }
}

UTEST_CASE(stats)
{
    auto dataset = fixture_dataset_t{};

    dataset.resize(nano::make_dims(100, 1, 2, 3), nano::make_dims(100, 10, 1, 1));
    UTEST_REQUIRE_NOTHROW(dataset.load());

    const auto batch = 11;
    const auto istats = dataset.istats(::nano::arange(0, 60), batch);

    UTEST_CHECK_EQUAL(istats.mean().template size<0>(), 1);
    UTEST_CHECK_EQUAL(istats.mean().template size<1>(), 2);
    UTEST_CHECK_EQUAL(istats.mean().template size<2>(), 3);

    UTEST_CHECK_EQUAL(istats.stdev().template size<0>(), 1);
    UTEST_CHECK_EQUAL(istats.stdev().template size<1>(), 2);
    UTEST_CHECK_EQUAL(istats.stdev().template size<2>(), 3);

    UTEST_CHECK_CLOSE(istats.min()(0), 0.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(1), 1.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(2), 2.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(3), 3.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(4), 4.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(5), 5.0, 1e-8);

    UTEST_CHECK_CLOSE(istats.max()(0), 59.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(1), 60.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(2), 61.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(3), 62.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(4), 63.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(5), 64.0, 1e-8);

    UTEST_CHECK_CLOSE(istats.mean()(0), 29.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(1), 30.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(2), 31.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(3), 32.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(4), 33.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(5), 34.5, 1e-8);

    UTEST_CHECK_CLOSE(istats.stdev().array().minCoeff(), 17.46425, 1e-6);
    UTEST_CHECK_CLOSE(istats.stdev().array().maxCoeff(), 17.46425, 1e-6);
}

UTEST_END_MODULE()
