#include <utest/utest.h>
#include <nano/core/histogram.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_core_histogram)

UTEST_CASE(make_equidistant_ratios)
{
    {
        const auto ratios = make_equidistant_ratios(2);
        const auto expected_ratios = make_tensor<scalar_t>(make_dims(1), 0.50);
        UTEST_CHECK_TENSOR_CLOSE(ratios, expected_ratios, 1e-15);
    }
    {
        const auto ratios = make_equidistant_ratios(3);
        const auto expected_ratios = make_tensor<scalar_t>(make_dims(2), 1.0/3.0, 2.0/3.0);
        UTEST_CHECK_TENSOR_CLOSE(ratios, expected_ratios, 1e-15);
    }
    {
        const auto ratios = make_equidistant_ratios(4);
        const auto expected_ratios = make_tensor<scalar_t>(make_dims(3), 0.25, 0.50, 0.75);
        UTEST_CHECK_TENSOR_CLOSE(ratios, expected_ratios, 1e-15);
    }
    {
        const auto ratios = make_equidistant_ratios(5);
        const auto expected_ratios = make_tensor<scalar_t>(make_dims(4), 0.20, 0.40, 0.60, 0.80);
        UTEST_CHECK_TENSOR_CLOSE(ratios, expected_ratios, 1e-15);
    }
}

UTEST_CASE(make_equidistant_percentiles)
{
    {
        const auto percentiles = make_equidistant_percentiles(2);
        const auto expected_percentiles = make_tensor<scalar_t>(make_dims(1), 50.0);
        UTEST_CHECK_TENSOR_CLOSE(percentiles, expected_percentiles, 1e-15);
    }
    {
        const auto percentiles = make_equidistant_percentiles(3);
        const auto expected_percentiles = make_tensor<scalar_t>(make_dims(2), 1.0*100.0/3.0, 2.0*100.0/3.0);
        UTEST_CHECK_TENSOR_CLOSE(percentiles, expected_percentiles, 1e-15);
    }
    {
        const auto percentiles = make_equidistant_percentiles(4);
        const auto expected_percentiles = make_tensor<scalar_t>(make_dims(3), 25.0, 50.0, 75.0);
        UTEST_CHECK_TENSOR_CLOSE(percentiles, expected_percentiles, 1e-15);
    }
    {
        const auto percentiles = make_equidistant_percentiles(5);
        const auto expected_percentiles = make_tensor<scalar_t>(make_dims(4), 20.0, 40.0, 60.0, 80.0);
        UTEST_CHECK_TENSOR_CLOSE(percentiles, expected_percentiles, 1e-15);
    }
}

UTEST_CASE(default_histogram)
{
    const auto histogram = histogram_t{};
    UTEST_CHECK_EQUAL(histogram.bins(), 0);
}

UTEST_CASE(histogram_from_ratios)
{
    {
        auto data = make_tensor<scalar_t>(make_dims(11), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        const auto ratios = make_tensor<scalar_t>(make_dims(3), 0.15, 0.55, 0.85);
        const auto histogram = histogram_t::make_from_ratios(begin(data), end(data), ratios);

        const auto expected_means = make_tensor<scalar_t>(make_dims(4), 0.5, 3.5, 7.0, 9.5);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(4), 2, 4, 3, 2);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(4), 0.5, 3.5, 7.0, 9.5);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(3), 1.5, 5.5, 8.5);

        UTEST_CHECK_EQUAL(histogram.bins(), 4);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-15);
    }
    {

        auto data = make_tensor<scalar_t>(make_dims(11), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        const auto ratios = 4;
        const auto histogram = histogram_t::make_from_ratios(begin(data), end(data), ratios);

        const auto expected_means = make_tensor<scalar_t>(make_dims(4), 1.0, 3.5, 6.0, 9.0);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(4), 3, 2, 3, 3);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(4), 1.0, 3.5, 6.0, 9.0);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(3), 2.5, 5.0, 7.5);

        UTEST_CHECK_EQUAL(histogram.bins(), 4);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-15);

        UTEST_CHECK_EQUAL(histogram.bin(-1), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+0), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+1), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+2), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+3), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+4), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+5), 2);
        UTEST_CHECK_EQUAL(histogram.bin(+6), 2);
        UTEST_CHECK_EQUAL(histogram.bin(+7), 2);
        UTEST_CHECK_EQUAL(histogram.bin(+8), 3);
        UTEST_CHECK_EQUAL(histogram.bin(+9), 3);
        UTEST_CHECK_EQUAL(histogram.bin(+10), 3);
        UTEST_CHECK_EQUAL(histogram.bin(+11), 3);
    }
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
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-15);

        UTEST_CHECK_CLOSE(histogram.mean(0), 1.0, 1e-15);
        UTEST_CHECK_EQUAL(histogram.count(0), 3);
        UTEST_CHECK_CLOSE(histogram.median(0), 1.0, 1e-15);

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
    {

        auto data = make_tensor<scalar_t>(make_dims(10), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        const auto thresholds = make_tensor<scalar_t>(make_dims(1), 5.3);
        const auto histogram = histogram_t::make_from_thresholds(begin(data), end(data), thresholds);

        const auto expected_means = make_tensor<scalar_t>(make_dims(2), 2.5, 7.5);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(2), 6, 4);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(2), 2.5, 7.5);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(1), 5.3);

        UTEST_CHECK_EQUAL(histogram.bins(), 2);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-15);

        UTEST_CHECK_CLOSE(histogram.mean(0), 2.5, 1e-15);
        UTEST_CHECK_EQUAL(histogram.count(0), 6);
        UTEST_CHECK_CLOSE(histogram.median(0), 2.5, 1e-15);

        UTEST_CHECK_EQUAL(histogram.bin(-1), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+0), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+2), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+3), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+4), 0);
        UTEST_CHECK_EQUAL(histogram.bin(+6), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+7), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+8), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+9), 1);
        UTEST_CHECK_EQUAL(histogram.bin(+10), 1);
    }
}

UTEST_CASE(histogram_from_percentiles)
{
    {
        auto data = make_tensor<scalar_t>(make_dims(11), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        const auto percentiles = make_tensor<scalar_t>(make_dims(3), 15.0, 55.0, 85.0);
        const auto histogram = histogram_t::make_from_percentiles(begin(data), end(data), percentiles);

        const auto expected_means = make_tensor<scalar_t>(make_dims(4), 0.5, 3.5, 7.0, 9.5);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(4), 2, 4, 3, 2);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(4), 0.5, 3.5, 7.0, 9.5);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(3), 1.5, 5.5, 8.5);

        UTEST_CHECK_EQUAL(histogram.bins(), 4);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-15);
    }
    {

        auto data = make_tensor<scalar_t>(make_dims(11), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        const auto percentiles = 4;
        const auto histogram = histogram_t::make_from_percentiles(begin(data), end(data), percentiles);

        const auto expected_means = make_tensor<scalar_t>(make_dims(4), 1.0, 3.5, 6.0, 9.0);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(4), 3, 2, 3, 3);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(4), 1.0, 3.5, 6.0, 9.0);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(3), 2.5, 5.0, 7.5);

        UTEST_CHECK_EQUAL(histogram.bins(), 4);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-15);
    }
}

UTEST_CASE(histogram_from_exponents)
{
    static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();
    {
        auto data = make_tensor<scalar_t>(make_dims(12),
            2e-32, 3e-16, 2e-6, 3e-6, 5e-6, 2e-5, 3e-5, 1e+0, 2e+0, 3e+0, 3e+1, 4e+2);
        const auto base = 10.0;
        const auto epsilon = 1e-16;
        const auto histogram = histogram_t::make_from_exponents(begin(data), end(data), base, epsilon);

        const auto expected_means = make_tensor<scalar_t>(make_dims(7),
            2e-32, 3e-16, (10.0/3.0)*1e-6, 2.5e-5, 2e+0, 3e+1, 4e+2);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(7),
            1, 1, 3, 2, 3, 1, 1);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(7),
            2e-32, 3e-16, 3e-6, 2.5e-5, 2e+0, 3e+1, 4e+2);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(6),
            1e-16, 1e-6, 1e-5, 1e+0, 1e+1, 1e+2);

        UTEST_CHECK_EQUAL(histogram.bins(), 7);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-16);
    }
    {
        auto data = make_tensor<scalar_t>(make_dims(14),
            1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 4e-3, 5e-3, 1e+0, 2e+0);
        const auto base = 10.0;
        const auto epsilon = 1e-16;
        const auto histogram = histogram_t::make_from_exponents(begin(data), end(data), base, epsilon);

        const auto expected_means = make_tensor<scalar_t>(make_dims(6),
            NaN, 2e-6, 2e-5, 4e-4, (13.0/4.0)*1e-3, 1.5e+0);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(6),
            0, 2, 2, 4, 4, 2);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(6),
            NaN, 2e-6, 2e-5, 4e-4, 3.5*1e-3, 1.5e+0);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(5),
            1e-6, 1e-5, 1e-4, 1e-3, 1e+0);

        UTEST_CHECK_EQUAL(histogram.bins(), 6);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-15);
    }
    {
        auto data = make_tensor<scalar_t>(make_dims(16),
            -1e-1, -3e-2, -3e-2, -1e-2, -1e-18, -0.0, +0.0, 1e-18, 1e-1, 3e-1, 1e+0, 2e+0, 5e+0, 1e+1, 3e+1, 1e+2);
        const auto base = 10.0;
        const auto epsilon = 1e-16;
        const auto histogram = histogram_t::make_from_exponents(begin(data), end(data), base, epsilon);

        const auto expected_means = make_tensor<scalar_t>(make_dims(9),
            NaN, -16.0/3.0*1e-2, -1e-2, +0.0, NaN, 2e-1, 8.0/3.0*1e+0, 2e+1, 1e+2);
        const auto expected_counts = make_tensor<tensor_size_t>(make_dims(9),
            0, 3, 1, 4, 0, 2, 3, 2, 1);
        const auto expected_medians = make_tensor<scalar_t>(make_dims(9),
            NaN, -3e-2, -1e-2, +0.0, NaN, 2e-1, 2e+0, 2e+1, 1e+2);
        const auto expected_thresholds = make_tensor<scalar_t>(make_dims(8),
            -1e-1, -1e-2, -1e-16, +1e-16, +1e-1, +1e+0, +1e+1, +1e+2);

        UTEST_CHECK_EQUAL(histogram.bins(), 9);
        UTEST_CHECK_TENSOR_CLOSE(histogram.means(), expected_means, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.counts(), expected_counts, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.medians(), expected_medians, 1e-15);
        UTEST_CHECK_TENSOR_CLOSE(histogram.thresholds(), expected_thresholds, 1e-15);
    }
}

UTEST_END_MODULE()
