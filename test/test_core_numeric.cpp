#include <utest/utest.h>
#include <nano/core/numeric.h>

#include <iomanip>

using namespace nano;

UTEST_BEGIN_MODULE(test_core_numeric)

UTEST_CASE(square)
{
    UTEST_CHECK_EQUAL(square(1), 1);
    UTEST_CHECK_EQUAL(square(7), 49);
}

UTEST_CASE(cube)
{
    UTEST_CHECK_EQUAL(cube(1), 1);
    UTEST_CHECK_EQUAL(cube(7), 343);
}

UTEST_CASE(quartic)
{
    UTEST_CHECK_EQUAL(quartic(1), 1);
    UTEST_CHECK_EQUAL(quartic(3), 81);
}

UTEST_CASE(idiv)
{
    UTEST_CHECK_EQUAL(idiv(1, 3), 0);
    UTEST_CHECK_EQUAL(idiv(2, 3), 1);
    UTEST_CHECK_EQUAL(idiv(3, 3), 1);
    UTEST_CHECK_EQUAL(idiv(4, 3), 1);
    UTEST_CHECK_EQUAL(idiv(5, 3), 2);
    UTEST_CHECK_EQUAL(idiv(6, 3), 2);
    UTEST_CHECK_EQUAL(idiv(7, 3), 2);

    UTEST_CHECK_EQUAL(idiv(1, 4), 0);
    UTEST_CHECK_EQUAL(idiv(2, 4), 1);
    UTEST_CHECK_EQUAL(idiv(3, 4), 1);
    UTEST_CHECK_EQUAL(idiv(4, 4), 1);
    UTEST_CHECK_EQUAL(idiv(5, 4), 1);
    UTEST_CHECK_EQUAL(idiv(6, 4), 2);
    UTEST_CHECK_EQUAL(idiv(7, 4), 2);
    UTEST_CHECK_EQUAL(idiv(8, 4), 2);
    UTEST_CHECK_EQUAL(idiv(8, 4), 2);
    UTEST_CHECK_EQUAL(idiv(9, 4), 2);
    UTEST_CHECK_EQUAL(idiv(10, 4), 3);
    UTEST_CHECK_EQUAL(idiv(11, 4), 3);
    UTEST_CHECK_EQUAL(idiv(12, 4), 3);
}

UTEST_CASE(iround)
{
    UTEST_CHECK_EQUAL(iround(1, 4), 0);
    UTEST_CHECK_EQUAL(iround(2, 4), 4);
    UTEST_CHECK_EQUAL(iround(4, 4), 4);
    UTEST_CHECK_EQUAL(iround(7, 4), 8);
    UTEST_CHECK_EQUAL(iround(11, 4), 12);

}

UTEST_CASE(close)
{
    UTEST_CHECK(close(1.01 * 1e-3, 1.02 * 1e-3, 1e-3));
    UTEST_CHECK(close(1.01 * 1e-3, 1.02 * 1e-3, 1e-4));

    UTEST_CHECK(close(1.1, 1.2, 1e-1));
    UTEST_CHECK(!close(1.1, 1.2, 1e-2));
    UTEST_CHECK(!close(1e-3, 1e-4, 1e-5));
}

UTEST_CASE(roundpow10)
{
    UTEST_CHECK_CLOSE(roundpow10(1e+3), 1e+3, 1e-12);
    UTEST_CHECK_CLOSE(roundpow10(1e-6), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(roundpow10(1.12), 1e+0, 1e-12);
    UTEST_CHECK_CLOSE(roundpow10(0.97 * 1e-3), 1e-3, 1e-12);
    UTEST_CHECK_CLOSE(roundpow10(1.12 * 1e-3), 1e-3, 1e-12);
    UTEST_CHECK_CLOSE(roundpow10(0.82 * 1e+4), 1e+4, 1e-12);
    UTEST_CHECK_CLOSE(roundpow10(1.42 * 1e+4), 1e+4, 1e-12);
}

UTEST_CASE(epsilon)
{
    UTEST_CHECK_CLOSE(epsilon0<double>(), 1e-15, 1e-16);
    UTEST_CHECK_CLOSE(epsilon1<double>(), 1e-10, 1e-11);
    UTEST_CHECK_CLOSE(epsilon2<double>(), 1e-8, 1e-9);
    UTEST_CHECK_CLOSE(epsilon3<double>(), 1e-5, 1e-6);

    UTEST_CHECK_CLOSE(epsilon0<float>(), 1e-6, 1e-7);
    UTEST_CHECK_CLOSE(epsilon1<float>(), 1e-5, 1e-6);
    UTEST_CHECK_CLOSE(epsilon2<float>(), 1e-3, 1e-4);
    UTEST_CHECK_CLOSE(epsilon3<float>(), 1e-2, 1e-3);
}

UTEST_END_MODULE()
