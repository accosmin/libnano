#include <utest/utest.h>
#include <nano/numeric.h>
#include <nano/linear/util.h>
#include <nano/linear/model.h>
#include <nano/linear/function.h>
#include <nano/dataset/synth_affine.h>

using namespace nano;

static auto make_fold()
{
    return fold_t{0, protocol::train};
}

static auto make_loss()
{
    auto loss = loss_t::all().get("squared");
    UTEST_REQUIRE(loss);
    return loss;
}

static auto make_solver(const char* name = "lbfgs", const scalar_t epsilon = epsilon3<scalar_t>())
{
    auto solver = solver_t::all().get(name);
    UTEST_REQUIRE(solver);
    solver->epsilon(epsilon);
    solver->max_iterations(100);
    return solver;
}

static auto make_dataset(const tensor_size_t isize = 5, const tensor_size_t tsize = 3)
{
    auto dataset = synthetic_affine_dataset_t{};
    dataset.folds(1);
    dataset.noise(epsilon1<scalar_t>());
    dataset.idim(make_dims(isize, 1, 1));
    dataset.tdim(make_dims(tsize, 1, 1));
    dataset.modulo(1);
    dataset.samples(100);
    dataset.train_percentage(80);
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
    auto loss = make_loss();
    auto dataset = make_dataset();

    auto function = linear_function_t{*loss, dataset, make_fold()};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_EQUAL(function.isize(), 5);
    UTEST_REQUIRE_EQUAL(function.tsize(), 3);
    UTEST_REQUIRE_NOTHROW(function.l1reg(0));
    UTEST_REQUIRE_NOTHROW(function.l2reg(0));
    UTEST_REQUIRE_NOTHROW(function.vAreg(0));

    const auto inputs = dataset.inputs(make_fold());
    const auto targets = dataset.targets(make_fold());

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
    auto loss = make_loss();
    auto dataset = make_dataset();

    auto function = linear_function_t{*loss, dataset, make_fold()};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_EQUAL(function.isize(), 5);
    UTEST_REQUIRE_EQUAL(function.tsize(), 3);
    UTEST_REQUIRE_THROW(function.l1reg(-1e+0), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.l1reg(+1e+9), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.l2reg(-1e+0), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.l2reg(+1e+9), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.vAreg(-1e+0), std::invalid_argument);
    UTEST_REQUIRE_THROW(function.vAreg(+1e+9), std::invalid_argument);
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
    auto loss = make_loss();
    auto solver = make_solver("cgd");
    auto dataset = make_dataset(3, 2);

    auto function = linear_function_t{*loss, dataset, make_fold()};
    UTEST_REQUIRE_EQUAL(function.size(), 3 * 2 + 2);
    UTEST_REQUIRE_NOTHROW(function.l1reg(0.0));
    UTEST_REQUIRE_NOTHROW(function.l2reg(0.0));
    UTEST_REQUIRE_NOTHROW(function.vAreg(0.0));

    solver->logger([] (const solver_state_t& state)
    {
        std::cout << state << ".\n";
        return true;
    });

    const auto state = solver->minimize(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state);
    UTEST_CHECK(state.converged(solver->epsilon()));

    UTEST_CHECK_EIGEN_CLOSE(function.bias(state.x).vector(), dataset.bias(), 1e+1 * solver->epsilon());
    UTEST_CHECK_EIGEN_CLOSE(function.weights(state.x).matrix(), dataset.weights(), 1e+1 * solver->epsilon());
}

UTEST_CASE(train)
{
    auto loss = make_loss();
    auto solver = make_solver();
    auto dataset = make_dataset(3, 2);

    auto model = linear_model_t{};
    for (const auto normalization : enum_values<::nano::normalization>())
    {
        for (const auto regularization : enum_values<::nano::regularization>())
        {
            train_result_t training;
            UTEST_REQUIRE_NOTHROW(model.batch(16));
            UTEST_REQUIRE_NOTHROW(model.tune_steps(1));
            UTEST_REQUIRE_NOTHROW(model.tune_trials(4));
            UTEST_REQUIRE_NOTHROW(model.normalization(normalization));
            UTEST_REQUIRE_NOTHROW(model.regularization(regularization));
            UTEST_REQUIRE_NOTHROW(training = model.train(*loss, dataset, *solver));

            UTEST_CHECK_EIGEN_CLOSE(model.bias().vector(), dataset.bias(), 1e+2 * solver->epsilon());
            UTEST_CHECK_EIGEN_CLOSE(model.weights().matrix(), dataset.weights(), 1e+2 * solver->epsilon());

            UTEST_CHECK_EQUAL(training.size(), dataset.folds());
            for (const auto& train_fold : training)
            {
                UTEST_CHECK_GREATER_EQUAL(train_fold.tr_error(), scalar_t(0));
                UTEST_CHECK_GREATER_EQUAL(train_fold.vd_error(), scalar_t(0));
                UTEST_CHECK_GREATER_EQUAL(train_fold.te_error(), scalar_t(0));

                UTEST_CHECK_LESS_EQUAL(train_fold.tr_error(), 1e+2 * solver->epsilon());
                UTEST_CHECK_LESS_EQUAL(train_fold.vd_error(), 1e+2 * solver->epsilon());
                UTEST_CHECK_LESS_EQUAL(train_fold.te_error(), 1e+2 * solver->epsilon());
            }

            const auto targets = dataset.targets(make_fold());

            tensor4d_t outputs;
            model.predict(dataset, make_fold(), outputs);
            UTEST_CHECK_EIGEN_CLOSE(targets.vector(), outputs.vector(), 1e+1 * solver->epsilon());

            outputs.random(-1, +2);
            model.predict(dataset, make_fold(), outputs.tensor());
            UTEST_CHECK_EIGEN_CLOSE(targets.vector(), outputs.vector(), 1e+1 * solver->epsilon());

            const auto filepath = string_t("test_linear.model");

            UTEST_REQUIRE_NOTHROW(model.save(filepath));
            {
                auto new_model = linear_model_t{};
                UTEST_REQUIRE_NOTHROW(new_model.load(filepath));
                UTEST_CHECK_EIGEN_CLOSE(new_model.bias().vector(), model.bias().vector(), epsilon0<scalar_t>());
                UTEST_CHECK_EIGEN_CLOSE(new_model.weights().matrix(), model.weights().matrix(), epsilon0<scalar_t>());
            }
        }
    }

    UTEST_CHECK_NOTHROW(model.regularization(static_cast<regularization>(-1)));
    UTEST_CHECK_THROW(model.train(*loss, dataset, *solver), std::runtime_error);
}

UTEST_END_MODULE()