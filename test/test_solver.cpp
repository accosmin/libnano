#include <iomanip>
#include <utest/utest.h>
#include <nano/core/numeric.h>
#include <nano/solver/quasi.h>
#include <nano/function/sphere.h>

using namespace nano;

static void setup_logger(solver_t& solver, std::stringstream& stream, tensor_size_t& iterations)
{
    // log the optimization steps
    solver.logger([&] (const solver_state_t& state)
    {
        ++ iterations;
        stream << "\tdescent: " << state << ".\n";
        return true;
    });

    // log the line-search steps
    solver.lsearch0_logger([&] (const solver_state_t& state0, const scalar_t t)
    {
        stream
            << "\t\tlsearch(0): t=" << state0.t << ",f=" << state0.f << ",g=" << state0.convergence_criterion()
            << ",t=" << t << ".\n";
    });

    solver.lsearchk_logger([&] (const solver_state_t& state0, const solver_state_t& state)
    {
        stream
            << "\t\tlsearch(t):t=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << ",armijo=" << state.has_armijo(state0, solver.c1())
            << ",wolfe=" << state.has_wolfe(state0, solver.c2())
            << ",swolfe=" << state.has_strong_wolfe(state0, solver.c2()) << ".\n";
    });
}

static void test(solver_t& solver, const string_t& solver_id, const function_t& function, const vector_t& x0)
{
    const auto old_n_failures = utest_n_failures.load();
    const auto state0 = solver_state_t{function, x0};

    std::stringstream stream;
    stream
        << std::fixed << std::setprecision(16)
        << function.name() << " " << solver_id << "[" << solver.lsearch0_id() << "," << solver.lsearchk_id() << "]\n"
        << ":x0=[" << state0.x.transpose() << "],f0=" << state0.f<< ",g0=" << state0.convergence_criterion() << "\n";

    tensor_size_t iterations = 0;
    setup_logger(solver, stream, iterations);

    // minimize
    solver.epsilon(1e-5);
    solver.max_iterations(5000);
    const auto state = solver.minimize(function, x0);
    UTEST_CHECK(state);

    // check function value decrease
    UTEST_CHECK_LESS_EQUAL(state.f, state0.f + epsilon1<scalar_t>());

    // check convergence
    UTEST_CHECK_LESS(state.convergence_criterion(), solver.epsilon());
    UTEST_CHECK_EQUAL(state.m_status, solver_state_t::status::converged);
    UTEST_CHECK_EQUAL(iterations, state.m_iterations);

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }
}

UTEST_BEGIN_MODULE(test_solver_lsearch)

UTEST_CASE(state_str)
{
    for (const auto status : enum_values<solver_state_t::status>())
    {
        std::stringstream stream;
        stream << status;
        UTEST_CHECK_EQUAL(stream.str(), scat(status));
    }
}

UTEST_CASE(state_valid)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    UTEST_CHECK(state);
}

UTEST_CASE(state_invalid_tINF)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.t = INFINITY;
    UTEST_CHECK(!state);
}

UTEST_CASE(state_invalid_fNAN)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.f = NAN;
    UTEST_CHECK(!state);
}

UTEST_CASE(state_has_descent)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.d = -state.g;
    UTEST_CHECK(state.has_descent());
}

UTEST_CASE(state_has_no_descent0)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.d.setZero();
    UTEST_CHECK(!state.has_descent());
}

UTEST_CASE(state_has_no_descent1)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()));
    state.d = state.g;
    UTEST_CHECK(!state.has_descent());
}

UTEST_CASE(state_convergence0)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state.converged(epsilon2<scalar_t>()));
    UTEST_CHECK_GREATER_EQUAL(state.convergence_criterion(), 0);
    UTEST_CHECK_LESS(state.convergence_criterion(), epsilon0<scalar_t>());
}

UTEST_CASE(state_convergence1)
{
    const function_sphere_t function(7);
    solver_state_t state(function, vector_t::Random(function.size()) * epsilon1<scalar_t>());
    UTEST_CHECK(state.converged(epsilon2<scalar_t>()));
    UTEST_CHECK_GREATER_EQUAL(state.convergence_criterion(), 0);
    UTEST_CHECK_LESS(state.convergence_criterion(), epsilon2<scalar_t>());
}

UTEST_CASE(config_solvers)
{
    for (const auto& solver_id : solver_t::all().ids())
    {
        const auto solver = solver_t::all().get(solver_id);
        UTEST_REQUIRE(solver);

        // NB: 0 < c1 < c2 < 1
        UTEST_CHECK_NOTHROW(solver->tolerance(1e-4, 1e-1));
        UTEST_CHECK_EQUAL(solver->c1(), 1e-4);
        UTEST_CHECK_EQUAL(solver->c2(), 1e-1);

        UTEST_CHECK_THROW(solver->tolerance(2e-1, 1e-1), std::runtime_error);
        UTEST_CHECK_THROW(solver->tolerance(1e-1, 1e-4), std::runtime_error);
        UTEST_CHECK_THROW(solver->tolerance(1e-1, +1.1), std::runtime_error);
        UTEST_CHECK_THROW(solver->tolerance(1e-1, -0.1), std::runtime_error);
        UTEST_CHECK_THROW(solver->tolerance(-0.1, +1.1), std::runtime_error);
        UTEST_CHECK_EQUAL(solver->c1(), 1e-4);
        UTEST_CHECK_EQUAL(solver->c2(), 1e-1);

        UTEST_CHECK_NOTHROW(solver->tolerance(1e-1, 9e-1));
        UTEST_CHECK_EQUAL(solver->c1(), 1e-1);
        UTEST_CHECK_EQUAL(solver->c2(), 9e-1);

        UTEST_CHECK_THROW(solver->lsearch0("invalid-lsearch0-id"), std::runtime_error);
        UTEST_CHECK_THROW(solver->lsearch0("constant", rlsearch0_t()), std::runtime_error);

        UTEST_CHECK_NOTHROW(solver->lsearch0("constant"));
        UTEST_CHECK_NOTHROW(solver->lsearch0("constant", lsearch0_t::all().get("constant")));

        UTEST_CHECK_THROW(solver->lsearchk("invalid-lsearchk-id"), std::runtime_error);
        UTEST_CHECK_THROW(solver->lsearchk("backtrack", rlsearchk_t()), std::runtime_error);

        UTEST_CHECK_NOTHROW(solver->lsearchk("backtrack"));
        UTEST_CHECK_NOTHROW(solver->lsearchk("backtrack", lsearchk_t::all().get("backtrack")));
    }
}

const auto all_functions = get_functions(4, 4, convexity::unknown);
const auto convex_functions = get_functions(4, 4, convexity::yes);

const auto all_solver_ids = solver_t::all().ids();
const auto best_solver_ids = solver_t::all().ids(std::regex("cgd|lbfgs|bfgs"));
const auto all_lsearch0_ids = lsearch0_t::all().ids();
const auto all_lsearchk_ids = lsearchk_t::all().ids();

UTEST_CASE(all_default_solvers)
{
    for (const auto& function : convex_functions)
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : all_solver_ids)
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            test(*solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_CASE(best_solvers_with_lsearches)
{
    for (const auto& function : all_functions)
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : best_solver_ids)
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            for (const auto& lsearch0_id : all_lsearch0_ids)
            {
                for (const auto& lsearchk_id : all_lsearchk_ids)
                {
                    UTEST_REQUIRE_NOTHROW(solver->lsearch0(lsearch0_id));
                    UTEST_REQUIRE_NOTHROW(solver->lsearchk(lsearchk_id));

                    test(*solver, solver_id, *function, vector_t::Random(function->size()));
                }
            }
        }
    }
}

UTEST_CASE(best_solvers_with_tolerances)
{
    for (const auto& function : all_functions)
    {
        UTEST_REQUIRE(function);

        for (const auto& solver_id : best_solver_ids)
        {
            const auto solver = solver_t::all().get(solver_id);
            UTEST_REQUIRE(solver);

            UTEST_REQUIRE_NOTHROW(solver->tolerance(1e-4, 1e-1));
            test(*solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver->tolerance(1e-4, 9e-1));
            test(*solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver->tolerance(1e-1, 9e-1));
            test(*solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_CASE(quasi_with_initializations)
{
    for (const auto& function : convex_functions)
    {
        UTEST_REQUIRE(function);

        {
            const auto *const solver_id = "bfgs";
            auto solver = solver_quasi_bfgs_t{};

            UTEST_REQUIRE_NOTHROW(solver.init(solver_quasi_t::initialization::identity));
            test(solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver.init(solver_quasi_t::initialization::scaled));
            test(solver, solver_id, *function, vector_t::Random(function->size()));
        }
        {
            const auto *const solver_id = "fletcher";
            auto solver = solver_quasi_fletcher_t{};

            UTEST_REQUIRE_NOTHROW(solver.init(solver_quasi_t::initialization::identity));
            test(solver, solver_id, *function, vector_t::Random(function->size()));

            UTEST_REQUIRE_NOTHROW(solver.init(solver_quasi_t::initialization::scaled));
            test(solver, solver_id, *function, vector_t::Random(function->size()));
        }
    }
}

UTEST_END_MODULE()
