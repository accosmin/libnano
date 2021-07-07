#include <utest/utest.h>
#include <nano/string.h>
#include <nano/core/stats.h>
#include <nano/core/random.h>
#include <nano/tensor/tensor.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_core_stats)

UTEST_CASE(empty)
{
    using tensor3d_t = nano::tensor_mem_t<int16_t, 3>;

    tensor3d_t tensor;

    UTEST_CHECK_CLOSE(tensor.stdev(), 0.0, 1e-16);
    UTEST_CHECK_CLOSE(tensor.variance(), 0.0, 1e-16);
}

UTEST_CASE(tensor)
{
    auto tensor = make_tensor<int16_t>(make_dims(4, 2, 1), 2, 4, 4, 4, 5, 5, 7, 9);

    UTEST_CHECK_EQUAL(tensor.min(), 2.0);
    UTEST_CHECK_EQUAL(tensor.max(), 9.0);
    UTEST_CHECK_CLOSE(tensor.sum(), 40.0, 1e-16);
    UTEST_CHECK_CLOSE(tensor.mean(), 5.0, 1e-16);
    UTEST_CHECK_CLOSE(tensor.variance(), 4.0, 1e-16);
    UTEST_CHECK_CLOSE(tensor.stdev(), std::sqrt(4.0/7.0), 1e-16);
    UTEST_CHECK_CLOSE(median(begin(tensor), end(tensor)), 4.5, 1e-16);
    UTEST_CHECK_CLOSE(percentile(begin(tensor), end(tensor), 10), 3.0, 1e-16);
    UTEST_CHECK_CLOSE(percentile(begin(tensor), end(tensor), 90), 8.0, 1e-16);
}

UTEST_CASE(stats)
{
    stats_t stats;
    stats(2, 4, 4, 4, 5, 5, 7, 9);

    UTEST_CHECK_EQUAL(stats.count(), size_t(8));
    UTEST_CHECK_EQUAL(stats.min(), 2.0);
    UTEST_CHECK_EQUAL(stats.max(), 9.0);
    UTEST_CHECK_CLOSE(stats.sum1(), 40.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.sum2(), 232.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.var(), 4.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.stdev(), 2.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.median(), 4.5, 1e-16);
    UTEST_CHECK_CLOSE(stats.percentile(10), 3.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.percentile(90), 8.0, 1e-16);
    UTEST_CHECK_EQUAL(scat(stats), "5+/-2[2,9]");

    stats.clear();
    auto tensor = make_tensor<int16_t>(make_dims(4, 2, 1), 2, 4, 4, 4, 5, 5, 7, 9);
    stats(begin(tensor), end(tensor));

    UTEST_CHECK_EQUAL(stats.count(), size_t(8));
    UTEST_CHECK_EQUAL(stats.min(), 2.0);
    UTEST_CHECK_EQUAL(stats.max(), 9.0);
    UTEST_CHECK_CLOSE(stats.sum1(), 40.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.sum2(), 232.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.var(), 4.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.stdev(), 2.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.median(), 4.5, 1e-16);
    UTEST_CHECK_CLOSE(stats.percentile(10), 3.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.percentile(90), 8.0, 1e-16);
}

UTEST_CASE(stats_iter)
{
    auto tensor = make_tensor<int16_t>(make_dims(4, 2, 1), 2, 4, 4, 4, 5, 5, 7, 9);
    stats_t stats(begin(tensor), end(tensor));

    UTEST_CHECK_EQUAL(stats.count(), size_t(8));
    UTEST_CHECK_EQUAL(stats.min(), 2.0);
    UTEST_CHECK_EQUAL(stats.max(), 9.0);
    UTEST_CHECK_CLOSE(stats.sum1(), 40.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.sum2(), 232.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.var(), 4.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.stdev(), 2.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.median(), 4.5, 1e-16);
    UTEST_CHECK_CLOSE(stats.percentile(10), 3.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.percentile(90), 8.0, 1e-16);
}

UTEST_CASE(stats_merge)
{
    stats_t stats1;
    stats1(2, 4, 4);

    stats_t stats2;
    stats2(4, 5, 5, 7, 9);

    stats_t stats;
    stats(stats1);
    stats(stats2);

    UTEST_CHECK_EQUAL(stats.count(), size_t(8));
    UTEST_CHECK_EQUAL(stats.min(), 2.0);
    UTEST_CHECK_EQUAL(stats.max(), 9.0);
    UTEST_CHECK_CLOSE(stats.sum1(), 40.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.sum2(), 232.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.var(), 4.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.stdev(), 2.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.median(), 4.5, 1e-16);
    UTEST_CHECK_CLOSE(stats.percentile(10), 3.0, 1e-16);
    UTEST_CHECK_CLOSE(stats.percentile(90), 8.0, 1e-16);
}

UTEST_CASE(random)
{
    const auto avg = -4.2;
    const auto var = 0.47;
    const auto count = size_t(37);

    auto rng = make_rng();
    auto udist = make_udist<double>(-var, +var);

    // generate random values
    std::vector<double> values;
    for (size_t i = 0; i < count; ++ i)
    {
        values.push_back(avg + udist(rng));
    }

    const auto min = *std::min_element(values.begin(), values.end());
    const auto max = *std::max_element(values.begin(), values.end());

    const auto sum1 = std::accumulate(values.begin(), values.end(), 0.0);
    const auto sum2 = std::accumulate(values.begin(), values.end(), 0.0,
        [] (const auto acc, const auto val) { return acc + val * val; });

    stats_t stats;
    stats(values.begin(), values.end());

    UTEST_CHECK_EQUAL(stats.count(), count);
    UTEST_CHECK_CLOSE(stats.min(), min, 1e-16);
    UTEST_CHECK_CLOSE(stats.max(), max, 1e-16);
    UTEST_CHECK_CLOSE(stats.sum1(), sum1, 1e-12);
    UTEST_CHECK_CLOSE(stats.sum2(), sum2, 1e-12);

    UTEST_CHECK_LESS_EQUAL(stats.max(), avg + var);
    UTEST_CHECK_GREATER_EQUAL(stats.min(), avg - var);

    UTEST_CHECK_CLOSE(stats.avg(), sum1 / count, 1e-12);
    UTEST_CHECK_LESS_EQUAL(stats.avg(), avg + var);
    UTEST_CHECK_GREATER_EQUAL(stats.avg(), avg - var);

    UTEST_CHECK_GREATER_EQUAL(stats.var(), 0.0);
    UTEST_CHECK_CLOSE(stats.var(), (sum2 - sum1 * sum1 / count) / count, 1e-12);
}

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

    const auto value50s = median_sorted(begin(data), end(data));
    UTEST_CHECK_CLOSE(value50s, 1.5, 1e-12);
}

UTEST_CASE(median5)
{
    auto data = make_tensor<int>(make_dims(5), 4, 1, 1, 2, 1);

    const auto value50 = median(begin(data), end(data));
    UTEST_CHECK_CLOSE(value50, 1.0, 1e-12);
}

UTEST_END_MODULE()
