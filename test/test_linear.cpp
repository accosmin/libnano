#include "fixture/utils.h"
#include <nano/numeric.h>
#include <nano/linear/util.h>
#include <nano/linear/model.h>
#include <nano/linear/function.h>
#include <nano/dataset/synth_affine.h>

using namespace nano;

static auto make_samples()
{
    return arange(0, 100);
}

static auto make_dataset(const tensor_size_t isize = 5, const tensor_size_t tsize = 3)
{
    auto dataset = synthetic_affine_dataset_t{};
    dataset.noise(epsilon1<scalar_t>());
    dataset.idims(make_dims(isize, 1, 1));
    dataset.tdims(make_dims(tsize, 1, 1));
    dataset.modulo(1);
    dataset.samples(100);
    UTEST_CHECK_NOTHROW(dataset.load());
    return dataset;
}

UTEST_BEGIN_MODULE(test_linear)

UTEST_CASE(predict)
{
    tensor1d_t bias(3); bias.random();
    tensor2d_t weights(5, 3); weights.random();
    tensor4d_t inputs(11, 5, 1, 1); inputs.random();

    tensor4d_t outputs;
    linear::predict(inputs, weights, bias, outputs);

    for (tensor_size_t sample = 0; sample < inputs.size<0>(); ++ sample)
    {
        UTEST_CHECK_EIGEN_CLOSE(
            outputs.vector(sample),
            weights.matrix().transpose() * inputs.vector(sample) + bias.vector(),
            epsilon1<scalar_t>());
    }
}

UTEST_CASE(evaluate)
{
    const auto loss = make_loss();
    const auto dataset = make_dataset();
    const auto samples = make_samples();

    auto function = linear_function_t{*loss, dataset, samples};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_EQUAL(function.isize(), 5);
    UTEST_REQUIRE_EQUAL(function.tsize(), 3);
    UTEST_REQUIRE_NOTHROW(function.l1reg(0));
    UTEST_REQUIRE_NOTHROW(function.l2reg(0));
    UTEST_REQUIRE_NOTHROW(function.vAreg(0));

    const auto inputs = dataset.inputs(make_samples());
    const auto targets = dataset.targets(make_samples());

    tensor4d_t outputs;
    const vector_t x = vector_t::Random(function.size());
    linear::predict(inputs, function.weights(x), function.bias(x), outputs);

    tensor1d_t values;
    loss->value(targets, outputs, values);

    for (tensor_size_t batch = 1; batch <= 16; ++ batch)
    {
        UTEST_REQUIRE_NOTHROW(function.batch(batch));
        UTEST_CHECK_LESS(std::fabs(function.vgrad(x) - values.vector().mean()), epsilon1<scalar_t>());
    }
}

UTEST_CASE(gradient)
{
    const auto loss = make_loss();
    const auto dataset = make_dataset();
    const auto samples = make_samples();

    auto function = linear_function_t{*loss, dataset, samples};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_EQUAL(function.isize(), 5);
    UTEST_REQUIRE_EQUAL(function.tsize(), 3);
    UTEST_REQUIRE_THROW(function.l1reg(-1e+0), std::runtime_error);
    UTEST_REQUIRE_THROW(function.l1reg(+1e+9), std::runtime_error);
    UTEST_REQUIRE_THROW(function.l2reg(-1e+0), std::runtime_error);
    UTEST_REQUIRE_THROW(function.l2reg(+1e+9), std::runtime_error);
    UTEST_REQUIRE_THROW(function.vAreg(-1e+0), std::runtime_error);
    UTEST_REQUIRE_THROW(function.vAreg(+1e+9), std::runtime_error);
    UTEST_REQUIRE_NOTHROW(function.l1reg(1e-1));
    UTEST_REQUIRE_NOTHROW(function.l2reg(1e+1));
    UTEST_REQUIRE_NOTHROW(function.vAreg(5e-1));

    const vector_t x = vector_t::Random(function.size());

    for (const auto normalization : enum_values<::nano::normalization>())
    {
        UTEST_REQUIRE_NOTHROW(function.normalization(normalization));
        UTEST_CHECK_LESS(function.grad_accuracy(x), 10 * epsilon2<scalar_t>());
    }
}

UTEST_CASE(minimize)
{
    const auto loss = make_loss();
    const auto solver = make_solver("cgd", epsilon3<scalar_t>());
    const auto dataset = make_dataset(3, 2);
    const auto samples = make_samples();

    auto function = linear_function_t{*loss, dataset, samples};
    UTEST_REQUIRE_EQUAL(function.size(), 3 * 2 + 2);
    UTEST_REQUIRE_NOTHROW(function.l1reg(0.0));
    UTEST_REQUIRE_NOTHROW(function.l2reg(0.0));
    UTEST_REQUIRE_NOTHROW(function.vAreg(0.0));

    const auto state = solver->minimize(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state);
    UTEST_CHECK(state.converged(solver->epsilon()));

    UTEST_CHECK_EIGEN_CLOSE(function.bias(state.x).vector(), dataset.bias(), 1e+1 * solver->epsilon());
    UTEST_CHECK_EIGEN_CLOSE(function.weights(state.x).matrix(), dataset.weights(), 1e+1 * solver->epsilon());
}

UTEST_CASE(train)
{
    const auto loss = make_loss();
    const auto solver = make_solver("cgd", epsilon3<scalar_t>());
    const auto dataset = make_dataset(3, 2);
    const auto samples = make_samples();

    for (const auto normalization : enum_values<::nano::normalization>())
    {
        auto model = linear_model_t{};
        UTEST_REQUIRE_NOTHROW(model.batch(16));
        if (normalization == ::nano::normalization::mean)
        {
            UTEST_REQUIRE_NOTHROW(model.l1reg(1e-6));
        }
        if (normalization == ::nano::normalization::minmax)
        {
            UTEST_REQUIRE_NOTHROW(model.l2reg(1e-6));
        }
        if (normalization == ::nano::normalization::standard)
        {
            UTEST_REQUIRE_NOTHROW(model.vAreg(1e-6));
        }
        UTEST_REQUIRE_NOTHROW(model.normalization(normalization));

        tensor4d_t outputs;
        UTEST_REQUIRE_NOTHROW(model.fit(*loss, dataset, samples, *solver));
        UTEST_REQUIRE_NOTHROW(outputs = model.predict(dataset, samples));

        UTEST_CHECK_EIGEN_CLOSE(model.bias().vector(), dataset.bias(), 1e+2 * solver->epsilon());
        UTEST_CHECK_EIGEN_CLOSE(model.weights().matrix(), dataset.weights(), 1e+2 * solver->epsilon());

        const auto targets = dataset.targets(samples);
        UTEST_CHECK_EQUAL(outputs.dims(), targets.dims());
        UTEST_CHECK_EIGEN_CLOSE(outputs.vector(), targets.vector(), 1e+1 * solver->epsilon());

        UTEST_REQUIRE_NOTHROW(outputs = model.predict(dataset, samples));
        UTEST_CHECK_EQUAL(outputs.dims(), targets.dims());
        UTEST_CHECK_EIGEN_CLOSE(outputs.vector(), targets.vector(), 1e+1 * solver->epsilon());

        outputs.random(-1, +2);
        UTEST_REQUIRE_NOTHROW(outputs = model.predict(dataset, samples));
        UTEST_CHECK_EIGEN_CLOSE(targets.vector(), outputs.vector(), 1e+1 * solver->epsilon());

        string_t str;
        {
            std::ostringstream stream;
            UTEST_REQUIRE_NOTHROW(model.write(stream));
            str = stream.str();
        }
        {
            auto new_model = linear_model_t{};
            std::istringstream stream(str);
            UTEST_REQUIRE_NOTHROW(new_model.read(stream));
            UTEST_CHECK_EIGEN_CLOSE(new_model.bias().vector(), model.bias().vector(), epsilon0<scalar_t>());
            UTEST_CHECK_EIGEN_CLOSE(new_model.weights().matrix(), model.weights().matrix(), epsilon0<scalar_t>());
        }
    }
}

UTEST_END_MODULE()
