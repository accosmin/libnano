#include <utest/utest.h>
#include <nano/mlearn/elemwise.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_mlearn_elemwise)

UTEST_CASE(scale)
{
    const std::array<scalar_t, 3> min = {{-1, +1, +2}};
    const std::array<scalar_t, 3> max = {{+1, +3, +7}};
    const std::array<scalar_t, 3> mean = {{0.0, 1.1, 5.1}};
    const std::array<scalar_t, 3> stdev = {{0.1, 0.2, 0.8}};

    elemwise_stats_t stats;
    stats.set(
        map_tensor(min.data(), 3, 1, 1),
        map_tensor(max.data(), 3, 1, 1),
        map_tensor(mean.data(), 3, 1, 1),
        map_tensor(stdev.data(), 3, 1, 1));

    const std::array<scalar_t, 15> original =
    {{
        -1.0, +1.0, +2.0,
        -0.5, +1.5, +3.0,
        -0.0, +2.0, +4.0,
        +0.5, +2.5, +5.0,
        +1.0, +3.0, +6.0
    }};

    const std::array<scalar_t, 15> normed_none =
    {{
        -1.0, +1.0, +2.0,
        -0.5, +1.5, +3.0,
        -0.0, +2.0, +4.0,
        +0.5, +2.5, +5.0,
        +1.0, +3.0, +6.0
    }};

    const std::array<scalar_t, 15> normed_mean =
    {{
        -0.50, -0.05, -3.1/5.0,
        -0.25, +0.20, -2.1/5.0,
        -0.00, +0.45, -1.1/5.0,
        +0.25, +0.70, -0.1/5.0,
        +0.50, +0.95, +0.9/5.0
    }};

    const std::array<scalar_t, 15> normed_minmax =
    {{
        +0.00, +0.00, +0.00,
        +0.25, +0.25, +0.20,
        +0.50, +0.50, +0.40,
        +0.75, +0.75, +0.60,
        +1.00, +1.00, +0.80
    }};

    const std::array<scalar_t, 15> normed_standard =
    {{
        -10.0, -0.50, -3.1/0.8,
         -5.0, +2.00, -2.1/0.8,
         +0.0, +4.50, -1.1/0.8,
         +5.0, +7.00, -0.1/0.8,
        +10.0, +9.50, +0.9/0.8
    }};

    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(feature_scaling::none, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_none.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(feature_scaling::mean, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_mean.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(feature_scaling::minmax, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_minmax.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(feature_scaling::standard, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_standard.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_THROW(stats.scale(static_cast<feature_scaling>(-1), inputs), std::runtime_error);
    }
}

UTEST_END_MODULE()
