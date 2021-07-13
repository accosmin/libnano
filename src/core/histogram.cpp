#include <nano/core/histogram.h>

using namespace nano;

tensor_mem_t<scalar_t, 1> nano::make_equidistant_percentiles(tensor_size_t bins)
{
    assert(bins > 1);

    const auto delta = 100.0 / static_cast<scalar_t>(bins);

    tensor_mem_t<scalar_t, 1> percentiles(bins - 1);
    percentiles.lin_spaced(delta, 100.0 - delta);
    return percentiles;
} // LCOV_EXCL_LINE

tensor_mem_t<scalar_t, 1> nano::make_equidistant_ratios(tensor_size_t bins)
{
    assert(bins > 1);

    const auto delta = 1.0 / static_cast<scalar_t>(bins);

    tensor_mem_t<scalar_t, 1> ratios(bins - 1);
    ratios.lin_spaced(delta, 1.0 - delta);
    return ratios;
} // LCOV_EXCL_LINE
