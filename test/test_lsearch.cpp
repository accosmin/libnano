#include <iomanip>
#include <utest/utest.h>
#include <nano/solver.h>
#include <nano/core/numeric.h>
#include <nano/lsearchk/fletcher.h>
#include <nano/lsearchk/backtrack.h>
#include <nano/lsearchk/cgdescent.h>
#include <nano/lsearchk/lemarechal.h>
#include <nano/lsearchk/morethuente.h>

using namespace nano;

static void config_lsearch(lsearchk_t& lsearch, const scalar_t c1 = 1e-4, const scalar_t c2 = 9e-1)
{
    UTEST_REQUIRE_NOTHROW(lsearch.tolerance(c1, c2));
    UTEST_REQUIRE_NOTHROW(lsearch.max_iterations(100));
}

static auto get_lsearch(const string_t& id, const scalar_t c1 = 1e-4, const scalar_t c2 = 9e-1)
{
    auto lsearch = lsearchk_t::all().get(id);
    UTEST_REQUIRE(lsearch);
    config_lsearch(*lsearch, c1, c2);
    return lsearch;
}

static auto get_lsearch_cgdescent(const lsearchk_cgdescent_t::criterion criterion)
{
    auto lsearch = lsearchk_cgdescent_t{};
    config_lsearch(lsearch);
    UTEST_REQUIRE_NOTHROW(lsearch.crit(criterion));
    return lsearch;
}

enum class lsearch_type
{
    fletcher,
    backtrack,
    lemarechal,
    morethuente,
    cgdescent_wolfe,
    cgdescent_approx_wolfe,
    cgdescent_wolfe_approx_wolfe,
};

static void setup_logger(lsearchk_t& lsearch, std::stringstream& stream)
{
    // log the line-search trials
    lsearch.logger([&] (const solver_state_t& state0, const solver_state_t& state)
    {
        stream
            << "\tt=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << ",armijo=" << state.has_armijo(state0, lsearch.c1())
            << ",wolfe=" << state.has_wolfe(state0, lsearch.c2())
            << ",swolfe=" << state.has_strong_wolfe(state0, lsearch.c2())
            << ",awolfe=" << state.has_approx_wolfe(state0, lsearch.c1(), lsearch.c2()) << ".\n";
    });
}

static void test(
    lsearchk_t& lsearch, const string_t& lsearch_id, const function_t& function,
    const lsearch_type type, const vector_t& x0, const scalar_t t0)
{
    const auto old_n_failures = utest_n_failures.load();

    auto state0 = solver_state_t{function, x0};
    UTEST_CHECK(state0);
    state0.d = -state0.g;
    const auto epsilon = 1e-6;// todo: get the updated value of epsilon for CGDESCENT!!!

    std::stringstream stream;
    stream
        << std::fixed << std::setprecision(16) << function.name() << " " << lsearch_id
        << ": x0=[" << state0.x.transpose() << "],t0=" << t0 << ",f0=" << state0.f << "\n";
    setup_logger(lsearch, stream);

    // check the Armijo and the Wolfe-like conditions are valid after line-search
    auto state = state0;
    UTEST_CHECK(lsearch.get(state, t0));
    UTEST_CHECK(state);

    switch (type)
    {
    case lsearch_type::backtrack:
        UTEST_CHECK(state.has_armijo(state0, lsearch.c1()));
        break;

    case lsearch_type::lemarechal:
        UTEST_CHECK(state.has_armijo(state0, lsearch.c1()));
        UTEST_CHECK(state.has_wolfe(state0, lsearch.c2()));
        break;

    case lsearch_type::morethuente:
        UTEST_CHECK(state.has_armijo(state0, lsearch.c1()));
        UTEST_CHECK(state.has_strong_wolfe(state0, lsearch.c2()));
        break;

    case lsearch_type::fletcher:
        UTEST_CHECK(state.has_armijo(state0, lsearch.c1()));
        UTEST_CHECK(state.has_strong_wolfe(state0, lsearch.c2()));
        break;

    case lsearch_type::cgdescent_wolfe:
        UTEST_CHECK(state.has_armijo(state0, lsearch.c1()));
        UTEST_CHECK(state.has_wolfe(state0, lsearch.c2()));
        break;

    case lsearch_type::cgdescent_approx_wolfe:
        UTEST_CHECK(state.has_approx_armijo(state0, epsilon));
        UTEST_CHECK(state.has_approx_wolfe(state0, lsearch.c1(), lsearch.c2()));
        break;

    case lsearch_type::cgdescent_wolfe_approx_wolfe:
        UTEST_CHECK(
            (state.has_armijo(state0, lsearch.c1()) && state.has_wolfe(state0, lsearch.c2())) ||
            (state.has_approx_armijo(state0, epsilon) && state.has_approx_wolfe(state0, lsearch.c1(), lsearch.c2())));
        break;

    default:
        break;
    }

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }
}

static void test(
    lsearchk_t& lsearch, const string_t& lsearch_id, const function_t& function, const lsearch_type type)
{
    for (const auto& c12 : std::vector<std::pair<scalar_t, scalar_t>>{{1e-4, 1e-1}, {1e-4, 9e-1}, {1e-1, 9e-1}})
    {
        UTEST_REQUIRE_NOTHROW(lsearch.tolerance(c12.first, c12.second));

        test(lsearch, lsearch_id, function, type, vector_t::Random(function.size()), 1e-1);
        test(lsearch, lsearch_id, function, type, vector_t::Random(function.size()), 3e-1);
        test(lsearch, lsearch_id, function, type, vector_t::Random(function.size()), 1e+0);
        test(lsearch, lsearch_id, function, type, vector_t::Random(function.size()), 3e+1);
    }
}

UTEST_BEGIN_MODULE(test_lsearch)

UTEST_CASE(backtrack)
{
    const auto *const lsearch_id = "backtrack";
    const auto lsearch = get_lsearch(lsearch_id);

    const auto functions = get_functions(1, 16, convexity::unknown, std::regex(".+"));
    for (const auto& function : functions)
    {
        test(*lsearch, lsearch_id, *function, lsearch_type::backtrack);
    }
}

UTEST_CASE(lemarechal)
{
    const auto *const lsearch_id = "lemarechal";
    const auto lsearch = get_lsearch(lsearch_id);

    const auto functions = get_functions(1, 16, convexity::unknown, std::regex(".+"));
    for (const auto& function : functions)
    {
        test(*lsearch, lsearch_id, *function, lsearch_type::lemarechal);
    }
}

UTEST_CASE(morethuente)
{
    const auto *const lsearch_id = "morethuente";
    const auto lsearch = get_lsearch(lsearch_id);

    const auto functions = get_functions(1, 16, convexity::unknown, std::regex(".+"));
    for (const auto& function : functions)
    {
        test(*lsearch, lsearch_id, *function, lsearch_type::morethuente);
    }
}

UTEST_CASE(fletcher)
{
    const auto *const lsearch_id = "fletcher";
    const auto lsearch = get_lsearch(lsearch_id);

    const auto functions = get_functions(1, 16, convexity::unknown, std::regex(".+"));
    for (const auto& function : functions)
    {
        test(*lsearch, lsearch_id, *function, lsearch_type::fletcher);
    }
}

UTEST_CASE(cgdescent_wolfe)
{
    const auto *const lsearch_id = "cgdscent";
    auto lsearch = get_lsearch_cgdescent(lsearchk_cgdescent_t::criterion::wolfe);

    const auto functions = get_functions(1, 16, convexity::unknown, std::regex(".+"));
    for (const auto& function : functions)
    {
        test(lsearch, lsearch_id, *function, lsearch_type::cgdescent_wolfe);
    }
}

UTEST_CASE(cgdescent_approx_wolfe)
{
    const auto *const lsearch_id = "cgdscent";
    auto lsearch = get_lsearch_cgdescent(lsearchk_cgdescent_t::criterion::approx_wolfe);

    const auto functions = get_functions(1, 16, convexity::unknown, std::regex(".+"));
    for (const auto& function : functions)
    {
        test(lsearch, lsearch_id, *function, lsearch_type::cgdescent_approx_wolfe);
    }
}

UTEST_CASE(cgdescent_wolfe_approx_wolfe)
{
    const auto *const lsearch_id = "cgdscent";
    auto lsearch = get_lsearch_cgdescent(lsearchk_cgdescent_t::criterion::wolfe_approx_wolfe);

    const auto functions = get_functions(1, 16, convexity::unknown, std::regex(".+"));
    for (const auto& function : functions)
    {
        test(lsearch, lsearch_id, *function, lsearch_type::cgdescent_wolfe_approx_wolfe);
    }
}

UTEST_END_MODULE()
