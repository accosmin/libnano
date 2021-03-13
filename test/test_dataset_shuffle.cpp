#include <utest/utest.h>
#include "fixture/memfixed.h"
#include <nano/dataset/shuffle.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_shuffle)

UTEST_CASE(shuffle)
{
    auto source = fixture_dataset_t{};
    source.resize(nano::make_dims(100, 1, 8, 8), nano::make_dims(100, 3, 1, 1));
    UTEST_REQUIRE_NOTHROW(source.load());

    const auto dataset = shuffle_dataset_t{source, 13};

    UTEST_CHECK_EQUAL(dataset.samples(), 100);
    UTEST_CHECK_EQUAL(dataset.features(), source.features());
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"feature_0_0_0"});
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_t{"feature_0_0_1"});
    UTEST_CHECK_EQUAL(dataset.feature(31), feature_t{"feature_0_3_7"});
    UTEST_CHECK_EQUAL(dataset.feature(62), feature_t{"feature_0_7_6"});
    UTEST_CHECK_EQUAL(dataset.feature(63), feature_t{"feature_0_7_7"});
    UTEST_CHECK_EQUAL(dataset.target(), source.target());

    const auto check_inputs = [&] (const auto& inputs, tensor_range_t range, const indices_t& features)
    {
        const auto imatrix = inputs.reshape(range.size(), -1);
        UTEST_REQUIRE_EQUAL(imatrix.cols(), features.size());
        for (tensor_size_t s = range.begin(); s < range.end(); ++ s)
        {
            const auto row = s - range.begin();
            for (tensor_size_t f = 0; f < features.size(); ++ f)
            {
                if (features(f) != 13)
                {
                    UTEST_CHECK_EQUAL(imatrix(row, f), fixture_dataset_t::value(s, features(f)));
                }
            }
        }

        const auto* const it = std::find(begin(features), end(features), 13);
        if (it != features.end())
        {
            const auto f = std::distance(begin(features), it);

            tensor1d_t original(range.size());
            tensor1d_t permuted(range.size());
            for (tensor_size_t s = range.begin(); s < range.end(); ++ s)
            {
                const auto row = s - range.begin();
                original(row) = fixture_dataset_t::value(s, 13);
                permuted(row) = imatrix(row, f);
            }

            UTEST_CHECK(std::is_permutation(begin(original), end(original), begin(permuted)));
        }
    };

    const auto range = make_range(17, 42);
    const auto samples = arange(range.begin(), range.end());
    {
        const auto targets = dataset.targets(samples);
        ::check_targets(targets, range);
    }
    {
        const auto inputs = dataset.inputs(samples);
        check_inputs(inputs, range, arange(0, 64));
    }
    {
        const auto inputs = dataset.inputs(samples, 22);
        check_inputs(inputs, range, make_tensor<tensor_size_t>(make_dims(1), 22));
    }
    {
        const auto inputs = dataset.inputs(samples, 13);
        check_inputs(inputs, range, make_tensor<tensor_size_t>(make_dims(1), 13));
    }
    {
        const auto features = make_tensor<tensor_size_t>(make_dims(3), 1, 7, 14);
        const auto inputs = dataset.inputs(samples, features);
        check_inputs(inputs, range, features);
    }
    {
        const auto features = make_tensor<tensor_size_t>(make_dims(3), 1, 7, 13);
        const auto inputs = dataset.inputs(samples, features);
        check_inputs(inputs, range, features);
    }
    {
        const auto features = make_tensor<tensor_size_t>(make_dims(3), 13, 1, 7);
        const auto inputs = dataset.inputs(samples, features);
        check_inputs(inputs, range, features);
    }
}

UTEST_END_MODULE()
