#include <utest/utest.h>
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

UTEST_CASE(fixed)
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
}

UTEST_CASE(merge)
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

UTEST_END_MODULE()
