#include <utest/utest.h>
#include <nano/core/histogram.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_core_histogram)

UTEST_CASE(make_equidistant_ratios)
{
    {
        const auto ratios = make_equidistant_ratios(2);
        const auto expected_ratios = make_tensor<scalar_t>(make_dims(1), 0.50);
        UTEST_CHECK_TENSOR_CLOSE(ratios, expected_ratios, 1e-12);
    }
    {
        const auto ratios = make_equidistant_ratios(3);
        const auto expected_ratios = make_tensor<scalar_t>(make_dims(2), 1.0/3.0, 2.0/3.0);
        UTEST_CHECK_TENSOR_CLOSE(ratios, expected_ratios, 1e-12);
    }
    {
        const auto ratios = make_equidistant_ratios(4);
        const auto expected_ratios = make_tensor<scalar_t>(make_dims(3), 0.25, 0.50, 0.75);
        UTEST_CHECK_TENSOR_CLOSE(ratios, expected_ratios, 1e-12);
    }
    {
        const auto ratios = make_equidistant_ratios(5);
        const auto expected_ratios = make_tensor<scalar_t>(make_dims(4), 0.20, 0.40, 0.60, 0.80);
        UTEST_CHECK_TENSOR_CLOSE(ratios, expected_ratios, 1e-12);
    }
}

UTEST_CASE(make_equidistant_percentiles)
{
    {
        const auto percentiles = make_equidistant_percentiles(2);
        const auto expected_percentiles = make_tensor<scalar_t>(make_dims(1), 50.0);
        UTEST_CHECK_TENSOR_CLOSE(percentiles, expected_percentiles, 1e-12);
    }
    {
        const auto percentiles = make_equidistant_percentiles(3);
        const auto expected_percentiles = make_tensor<scalar_t>(make_dims(2), 1.0*100.0/3.0, 2.0*100.0/3.0);
        UTEST_CHECK_TENSOR_CLOSE(percentiles, expected_percentiles, 1e-12);
    }
    {
        const auto percentiles = make_equidistant_percentiles(4);
        const auto expected_percentiles = make_tensor<scalar_t>(make_dims(3), 25.0, 50.0, 75.0);
        UTEST_CHECK_TENSOR_CLOSE(percentiles, expected_percentiles, 1e-12);
    }
    {
        const auto percentiles = make_equidistant_percentiles(5);
        const auto expected_percentiles = make_tensor<scalar_t>(make_dims(4), 20.0, 40.0, 60.0, 80.0);
        UTEST_CHECK_TENSOR_CLOSE(percentiles, expected_percentiles, 1e-12);
    }
}

UTEST_CASE(default_histogram)
{
    const auto histogram = histogram_t{};
    UTEST_CHECK_EQUAL(histogram.bins(), 0);
}

UTEST_CASE(histogram_from_thresholds)
{
    {
        auto data = make_tensor<scalar_t>(make_dims(10), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        const auto thresholds = make_tensor<scalar_t>(make_dims(2), 2.5, 6.4);
        const auto histogram = histogram_t::make_from_thresholds(begin(data), end(data), thresholds);

        const auto expected_means = make_tensor<scalar_t>(make_dims(3), 1.0, 4.5, 8.0);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(3), 3, 4, 3);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(3), 1.0, 4.5, 8.0);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(2), 2.5, 6.4);

        UTEST_CHECK_EQUAL(histogram.bins(), 3);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-12);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-12);

        UTEST_CHECK_CLOSE(histogram.mean(0), 1.0, 1e-12);
        UTEST_CHECK_EQUAL(histogram.count(0), 3);
        UTEST_CHECK_CLOSE(histogram.median(0), 1.0, 1e-12);

        UTEST_CHECK_EQUAL(histogram.bin(-1), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+0), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+2), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+3), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+4), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+6), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+7), 2);
        UTEST_CHECK_EQUAL(histogram.bin(+8), 2);
        UTEST_CHECK_EQUAL(histogram.bin(+9), 2);
        UTEST_CHECK_EQUAL(histogram.bin(+10), 2);
    }
}

UTEST_END_MODULE()
