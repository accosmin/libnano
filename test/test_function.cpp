#include <utest/utest.h>
#include <nano/function.h>
#include <nano/core/numeric.h>
#include <nano/function/geometric.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_function)

UTEST_CASE(convex)
{
    for (const auto& rfunction : get_functions(1, 4, convexity::unknown, std::regex(".+")))
    {
        const auto& function = *rfunction;
        std::cout << function.name() << std::endl;

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 1);

        auto is_convex = true;
        for (auto t = 0; t < 100; ++ t)
        {
            const vector_t x0 = vector_t::Random(dims);
            const vector_t x1 = vector_t::Random(dims);

            is_convex = is_convex && function.is_convex(x0, x1, 20);
        }
        UTEST_CHECK((function.convex() != convexity::yes) || is_convex);
    }
}

UTEST_CASE(vgrad)
{
    for (const auto& rfunction : get_functions(1, 4, convexity::unknown, std::regex(".+")))
    {
        const auto& function = *rfunction;
        std::cout << function.name() << std::endl;

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 1);

        for (auto t = 0; t < 100; ++ t)
        {
            const vector_t x0 = vector_t::Random(dims);
            UTEST_CHECK_LESS(function.grad_accuracy(x0), 10 * epsilon2<scalar_t>());
        }
    }
}

UTEST_END_MODULE()
