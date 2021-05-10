#include <nano/loss.h>
#include <utest/utest.h>
#include <nano/function.h>
#include <nano/core/random.h>
#include <nano/core/numeric.h>
#include <nano/mlearn/class.h>

using namespace nano;

struct loss_function_t final : public function_t
{
    loss_function_t(const rloss_t& loss, const tensor_size_t xmaps) :
        function_t("loss", 3 * xmaps, convexity::no),
        m_loss(loss), m_target(3, xmaps, 1, 1), m_values(3)
    {
        m_target.tensor(0) = class_target(xmaps, 11 % xmaps);
        m_target.tensor(1) = class_target(xmaps, 12 % xmaps);
        m_target.tensor(2) = class_target(xmaps, 13 % xmaps);
    }

    scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
    {
        UTEST_REQUIRE_EQUAL(x.size(), m_target.size());
        const auto output = map_tensor(x.data(), m_target.dims());

        if (gx != nullptr)
        {
            gx->resize(m_target.size());

            m_loss->vgrad(m_target, output, map_tensor(gx->data(), m_target.dims()));
            UTEST_REQUIRE(gx->array().isFinite().all());
        }

        m_loss->value(m_target, output, m_values.tensor());
        UTEST_REQUIRE(m_values.array().isFinite().all());
        return m_values.array().sum();
    }

    const rloss_t&      m_loss;
    tensor4d_t          m_target;
    mutable tensor1d_t  m_values;
};

UTEST_BEGIN_MODULE(test_loss)

UTEST_CASE(gradient)
{
    const tensor_size_t cmd_min_dims = 2;
    const tensor_size_t cmd_max_dims = 5;

    // evaluate the analytical gradient vs. the finite difference approximation
    for (const auto& loss_id : loss_t::all().ids())
    {
        std::cout << "evaluating loss <" << loss_id << ">...\n";
        for (tensor_size_t cmd_dims = cmd_min_dims; cmd_dims <= cmd_max_dims; ++ cmd_dims)
        {
            const auto loss = loss_t::all().get(loss_id);
            const auto function = loss_function_t(loss, cmd_dims);

            vector_t x(function.size());

            const auto max_power = (loss_id == "m-exponential" || loss_id == "s-exponential") ? 5 : 20;
            for (int power = 0; power <= max_power; ++ power)
            {
                x.setRandom();
                x.array() *= std::pow(std::exp(1.0), power);

                const auto f = function.vgrad(x);
                UTEST_CHECK_EQUAL(std::isfinite(f), true);
                UTEST_CHECK_GREATER_EQUAL(f, scalar_t(0));
                UTEST_CHECK_LESS(function.grad_accuracy(x), 2 * epsilon2<scalar_t>());
            }
        }
    }
}

UTEST_CASE(single_class)
{
    for (const auto& loss_id : loss_t::all().ids(std::regex("s-.+")))
    {
        std::cout << "evaluating loss <" << loss_id << ">...\n";
        const auto loss = loss_t::all().get(loss_id);
        UTEST_REQUIRE(loss);

        const auto n_classes = 1;

        tensor4d_t targets(4, n_classes, 1, 1);
        targets.tensor(0) = class_target(n_classes);
        targets.tensor(1) = class_target(n_classes, 0);
        targets.tensor(2) = class_target(n_classes);
        targets.tensor(3) = class_target(n_classes, 0);

        tensor4d_t outputs(4, n_classes, 1, 1);
        outputs.tensor(0) = class_target(n_classes);
        outputs.tensor(1) = class_target(n_classes, 0);
        outputs.tensor(2) = class_target(n_classes, 0);
        outputs.tensor(3) = class_target(n_classes);

        tensor1d_t errors;
        loss->error(targets, outputs, errors);

        UTEST_CHECK_CLOSE(errors(0), scalar_t(0), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(1), scalar_t(0), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(2), scalar_t(1), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(3), scalar_t(1), epsilon0<scalar_t>());
    }
}

UTEST_CASE(single_label_multi_class)
{
    for (const auto& loss_id : loss_t::all().ids(std::regex("s-.+")))
    {
        std::cout << "evaluating loss <" << loss_id << ">...\n";
        const auto loss = loss_t::all().get(loss_id);
        UTEST_REQUIRE(loss);

        const auto n_classes = 13;

        tensor4d_t targets(4, n_classes, 1, 1);
        targets.tensor(0) = class_target(n_classes, 11);
        targets.tensor(1) = class_target(n_classes, 11);
        targets.tensor(2) = class_target(n_classes, 11);
        targets.tensor(3) = class_target(n_classes, 11);

        tensor4d_t outputs(4, n_classes, 1, 1);
        outputs.tensor(0) = class_target(n_classes, 11);
        outputs.tensor(1) = class_target(n_classes, 12);
        outputs.tensor(2) = class_target(n_classes, 11); outputs.vector(2)(7) = pos_target() + 1;
        outputs.tensor(3) = class_target(n_classes);

        tensor1d_t errors;
        loss->error(targets, outputs, errors);

        UTEST_CHECK_CLOSE(errors(0), scalar_t(0), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(1), scalar_t(1), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(2), scalar_t(1), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(3), scalar_t(1), epsilon0<scalar_t>());
    }
}

UTEST_CASE(multi_label_multi_class)
{
    for (const auto& loss_id : loss_t::all().ids(std::regex("m-.+")))
    {
        std::cout << "evaluating loss <" << loss_id << ">...\n";
        const auto loss = loss_t::all().get(loss_id);
        UTEST_REQUIRE(loss);

        const auto n_classes = 13;

        tensor4d_t targets(6, n_classes, 1, 1);
        targets.tensor(0) = class_target(n_classes, 7, 9);
        targets.tensor(1) = class_target(n_classes, 7, 9);
        targets.tensor(2) = class_target(n_classes, 7, 9);
        targets.tensor(3) = class_target(n_classes, 7, 9);
        targets.tensor(4) = class_target(n_classes, 7, 9);
        targets.tensor(5) = class_target(n_classes, 7, 9);

        tensor4d_t outputs(6, n_classes, 1, 1);
        outputs.tensor(0) = class_target(n_classes, 7, 9);
        outputs.tensor(1) = class_target(n_classes);
        outputs.tensor(2) = class_target(n_classes, 5);
        outputs.tensor(3) = class_target(n_classes, 7);
        outputs.tensor(4) = class_target(n_classes, 5, 9);
        outputs.tensor(5) = class_target(n_classes, 7, 9, 11);

        tensor1d_t errors;
        loss->error(targets, outputs, errors);

        UTEST_CHECK_CLOSE(errors(0), scalar_t(0), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(1), scalar_t(2), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(2), scalar_t(3), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(3), scalar_t(1), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(4), scalar_t(2), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(5), scalar_t(1), epsilon0<scalar_t>());
    }
}

UTEST_CASE(regression)
{
    for (const auto& loss_id : {"absolute", "squared", "cauchy"})
    {
        std::cout << "evaluating loss <" << loss_id << ">...\n";
        const auto loss = loss_t::all().get(loss_id);
        UTEST_REQUIRE(loss);

        tensor4d_t target(3, 4, 1, 1);
        target.random();

        tensor4d_t output = target;

        tensor1d_t errors(3);
        loss->error(target, output, errors.tensor());
        UTEST_CHECK_LESS(errors.array().abs().maxCoeff(), epsilon0<scalar_t>());
    }
}

UTEST_END_MODULE()
