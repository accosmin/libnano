#include <utest/utest.h>
#include <nano/tensor/tensor.h>
#include <nano/core/percentile.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_core_percentile)

UTEST_CASE(percentile10)
{
    auto data = make_tensor<int>(make_dims(11), 0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

    const auto value0 = percentile(begin(data), end(data), 0);
    const auto value10 = percentile(begin(data), end(data), 10);
    const auto value20 = percentile(begin(data), end(data), 20);
    const auto value30 = percentile(begin(data), end(data), 30);
    const auto value40 = percentile(begin(data), end(data), 40);
    const auto value50 = percentile(begin(data), end(data), 50);
    const auto value60 = percentile(begin(data), end(data), 60);
    const auto value70 = percentile(begin(data), end(data), 70);
    const auto value80 = percentile(begin(data), end(data), 80);
    const auto value90 = percentile(begin(data), end(data), 90);
    const auto value100 = percentile(begin(data), end(data), 100);

    UTEST_CHECK_CLOSE(value0, 0.0, 1e-12);
    UTEST_CHECK_CLOSE(value10, 1.0, 1e-12);
    UTEST_CHECK_CLOSE(value20, 2.0, 1e-12);
    UTEST_CHECK_CLOSE(value30, 3.0, 1e-12);
    UTEST_CHECK_CLOSE(value40, 4.0, 1e-12);
    UTEST_CHECK_CLOSE(value50, 5.0, 1e-12);
    UTEST_CHECK_CLOSE(value60, 6.0, 1e-12);
    UTEST_CHECK_CLOSE(value70, 7.0, 1e-12);
    UTEST_CHECK_CLOSE(value80, 8.0, 1e-12);
    UTEST_CHECK_CLOSE(value90, 9.0, 1e-12);
    UTEST_CHECK_CLOSE(value100, 10.0, 1e-12);
}

UTEST_CASE(percentile13)
{
    auto data = make_tensor<int>(make_dims(13), 8, 1, 1, 2, 2, 4, 5, 2, 1, 2, 2, 3, 7);

    const auto value0 = percentile(begin(data), end(data), 0);
    const auto value10 = percentile(begin(data), end(data), 10);
    const auto value20 = percentile(begin(data), end(data), 20);
    const auto value30 = percentile(begin(data), end(data), 30);
    const auto value40 = percentile(begin(data), end(data), 40);
    const auto value50 = percentile(begin(data), end(data), 50);
    const auto value60 = percentile(begin(data), end(data), 60);
    const auto value70 = percentile(begin(data), end(data), 70);
    const auto value80 = percentile(begin(data), end(data), 80);
    const auto value90 = percentile(begin(data), end(data), 90);
    const auto value100 = percentile(begin(data), end(data), 100);

    UTEST_CHECK_CLOSE(value0, 1.0, 1e-12);
    UTEST_CHECK_CLOSE(value10, 1.0, 1e-12);
    UTEST_CHECK_CLOSE(value20, 1.5, 1e-12);
    UTEST_CHECK_CLOSE(value30, 2.0, 1e-12);
    UTEST_CHECK_CLOSE(value40, 2.0, 1e-12);
    UTEST_CHECK_CLOSE(value50, 2.0, 1e-12);
    UTEST_CHECK_CLOSE(value60, 2.5, 1e-12);
    UTEST_CHECK_CLOSE(value70, 3.5, 1e-12);
    UTEST_CHECK_CLOSE(value80, 4.5, 1e-12);
    UTEST_CHECK_CLOSE(value90, 6.0, 1e-12);
    UTEST_CHECK_CLOSE(value100, 8.0, 1e-12);
}

UTEST_CASE(median4)
{
    auto data = make_tensor<int>(make_dims(4), 1, 1, 2, 2);

    const auto value50 = median(begin(data), end(data));
    UTEST_CHECK_CLOSE(value50, 1.5, 1e-12);
}

UTEST_CASE(median5)
{
    auto data = make_tensor<int>(make_dims(5), 4, 1, 1, 2, 1);

    const auto value50 = median(begin(data), end(data));
    UTEST_CHECK_CLOSE(value50, 1.0, 1e-12);
}

UTEST_END_MODULE()
