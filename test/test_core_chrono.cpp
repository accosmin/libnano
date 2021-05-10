#include <utest/utest.h>
#include <nano/core/chrono.h>

UTEST_BEGIN_MODULE(test_chrono)

UTEST_CASE(gflops)
{
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::seconds_t(1)), 0);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::milliseconds_t(1)), 0);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::microseconds_t(1)), 0);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(100)), 0);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(10)), 4);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(1)), 42);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(100)), 420);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(10)), 4200);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(1)), 42000);
}

UTEST_CASE(probe)
{
    const auto *const basename = "base";
    const auto *const fullname = "full";
    const auto flops = 2048;

    nano::probe_t probe(basename, fullname, flops);

    UTEST_CHECK_EQUAL(probe.basename(), basename);
    UTEST_CHECK_EQUAL(probe.fullname(), fullname);
    UTEST_CHECK_EQUAL(probe.flops(), flops);
    UTEST_CHECK_EQUAL(probe.kflops(), flops / 1024);
    UTEST_CHECK(!probe);

    probe.measure([] () {});
    probe.measure([] () {});
    probe.measure([] () {});
    probe.measure([] () {});

    UTEST_CHECK(probe);
    UTEST_CHECK_EQUAL(probe.flops(), flops);
    UTEST_CHECK_EQUAL(probe.kflops(), flops / 1024);
    UTEST_CHECK_EQUAL(probe.gflops(), nano::gflops(flops, nano::nanoseconds_t(static_cast<int64_t>(probe.timings().min()))));
}

UTEST_END_MODULE()
