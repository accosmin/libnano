#include <utest/utest.h>
#include <nano/dataset/iterator.h>

using namespace nano;

template <typename tscalar>
std::ostream& operator<<(std::ostream& os, const std::vector<tscalar>& values)
{
    os << "{";
    for (const auto& value : values)
    {
        os << "{" << value << "}";
    }
    return os << "}";
}

UTEST_BEGIN_MODULE(test_dataset_iterator)

UTEST_CASE(data1D)
{
    tensor_mem_t<int, 1> data(16);
    data.full(-1);

    auto mask = make_mask(make_dims(16));

    for (int sample = 0; sample < 16; sample += 2)
    {
        setbit(mask, sample);
        data(sample) = sample + 3;
    }

    {
        const auto it = dataset_iterator_t<int, 1>{};
        UTEST_CHECK_EQUAL(it.size(), 0);
        UTEST_CHECK_EQUAL(it.index(), 0);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
    {
        const auto samples = arange(5, 10);

        auto it = make_iterator(data, mask, samples);
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 0);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 0);
            UTEST_CHECK_EQUAL(given, false);
            UTEST_CHECK_EQUAL(value, -1);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 1);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 1);
            UTEST_CHECK_EQUAL(given, true);
            UTEST_CHECK_EQUAL(value, 9);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 2);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 2);
            UTEST_CHECK_EQUAL(given, false);
            UTEST_CHECK_EQUAL(value, -1);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 3);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 3);
            UTEST_CHECK_EQUAL(given, true);
            UTEST_CHECK_EQUAL(value, 11);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 4);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 4);
            UTEST_CHECK_EQUAL(given, false);
            UTEST_CHECK_EQUAL(value, -1);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 5);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
    {
        const auto samples = arange(4, 16);
        const auto expected_indices = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        const auto expected_givens = std::vector<int>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
        const auto expected_values = std::vector<int>{7, -1, 9, -1, 11, -1, 13, -1, 15, -1, 17, -1};

        auto it = make_iterator(data, mask, samples);
        const auto it_end = make_end_iterator(samples);
        UTEST_CHECK_EQUAL(it.size(), 12);
        UTEST_CHECK_EQUAL(it.index(), 0);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        UTEST_CHECK_EQUAL(it_end.size(), 12);
        UTEST_CHECK_EQUAL(it_end.index(), 12);
        UTEST_CHECK_EQUAL(static_cast<bool>(it_end), false);

        std::vector<int> indices, givens, values;
        for ( ; it != it_end; ++ it)
        {
            const auto [index, given, value] = *it;
            indices.push_back(static_cast<int>(index));
            givens.push_back(given);
            values.push_back(value);
        }

        UTEST_CHECK_EQUAL(indices, expected_indices);
        UTEST_CHECK_EQUAL(givens, expected_givens);
        UTEST_CHECK_EQUAL(values, expected_values);

        indices.clear();
        givens.clear();
        values.clear();
        for (auto it = make_iterator(data, mask, samples); it; ++ it)
        {
            const auto [index, given, value] = *it;
            indices.push_back(static_cast<int>(index));
            givens.push_back(given);
            values.push_back(value);
        }

        UTEST_CHECK_EQUAL(indices, expected_indices);
        UTEST_CHECK_EQUAL(givens, expected_givens);
        UTEST_CHECK_EQUAL(values, expected_values);
    }
}

UTEST_CASE(data4D)
{
    tensor_mem_t<int, 4> data(16, 3, 2, 1);
    data.full(-1);

    auto mask = make_mask(make_dims(16));

    for (int sample = 0; sample < 16; sample += 2)
    {
        setbit(mask, sample);
        data.tensor(sample).full(sample + 3);
    }

    {
        const auto it = dataset_iterator_t<int, 1>{};
        UTEST_CHECK_EQUAL(it.size(), 0);
        UTEST_CHECK_EQUAL(it.index(), 0);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
    {
        const auto samples = arange(5, 8);

        auto it = make_iterator(data, mask, samples);
        UTEST_CHECK_EQUAL(it.size(), 3);
        UTEST_CHECK_EQUAL(it.index(), 0);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 0);
            UTEST_CHECK_EQUAL(given, false);
            UTEST_CHECK_EQUAL(value.min(), -1);
            UTEST_CHECK_EQUAL(value.max(), -1);
            UTEST_CHECK_EQUAL(value.dims(), make_dims(3, 2, 1));
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 3);
        UTEST_CHECK_EQUAL(it.index(), 1);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 1);
            UTEST_CHECK_EQUAL(given, true);
            UTEST_CHECK_EQUAL(value.min(), 9);
            UTEST_CHECK_EQUAL(value.max(), 9);
            UTEST_CHECK_EQUAL(value.dims(), make_dims(3, 2, 1));
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 3);
        UTEST_CHECK_EQUAL(it.index(), 2);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 2);
            UTEST_CHECK_EQUAL(given, false);
            UTEST_CHECK_EQUAL(value.min(), -1);
            UTEST_CHECK_EQUAL(value.max(), -1);
            UTEST_CHECK_EQUAL(value.dims(), make_dims(3, 2, 1));
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 3);
        UTEST_CHECK_EQUAL(it.index(), 3);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
}

UTEST_CASE(pairwise)
{
    const auto data1 = make_tensor<int>(make_dims(4, 2, 2, 1), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const auto data2 = make_tensor<int>(make_dims(4), -1, -2, -3, -4);

    auto mask1 = make_mask(make_dims(4));
    auto mask2 = make_mask(make_dims(4));

    setbit(mask1, 0);
    setbit(mask1, 1);
    setbit(mask1, 3);

    setbit(mask2, 1);
    setbit(mask2, 2);
    setbit(mask2, 3);

    const auto samples = arange(0, 4);

    {
        const auto it = dataset_pairwise_iterator_t<int, 4, int, 1>{};
        UTEST_CHECK_EQUAL(it.size(), 0);
        UTEST_CHECK_EQUAL(it.index(), 0);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
    {
        auto it = make_iterator(data1, mask1, data2, mask2, samples);
        for (auto i = 0; i < 4; ++ i, ++ it)
        {
            UTEST_CHECK_EQUAL(it.size(), 4);
            UTEST_CHECK_EQUAL(it.index(), i);
            UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
            {
                const auto [index, given1, value1, given2, value2] = *it;
                UTEST_CHECK_EQUAL(index, i);
                UTEST_CHECK_EQUAL(given1, getbit(mask1, i));
                UTEST_CHECK_EQUAL(given2, getbit(mask2, i));
                UTEST_CHECK_TENSOR_EQUAL(value1, data1.tensor(i));
                UTEST_CHECK_EQUAL(value2, data2(i));
            }
        }
        UTEST_CHECK_EQUAL(it.size(), 4);
        UTEST_CHECK_EQUAL(it.index(), 4);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
    {
        auto it = make_iterator(data1, mask1, data1, mask1, samples);
        for (auto i = 0; i < 4; ++ i, ++ it)
        {
            UTEST_CHECK_EQUAL(it.size(), 4);
            UTEST_CHECK_EQUAL(it.index(), i);
            UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
            {
                const auto [index, given1, value1, given2, value2] = *it;
                UTEST_CHECK_EQUAL(index, i);
                UTEST_CHECK_EQUAL(given1, getbit(mask1, i));
                UTEST_CHECK_EQUAL(given2, getbit(mask1, i));
                UTEST_CHECK_TENSOR_EQUAL(value1, data1.tensor(i));
                UTEST_CHECK_TENSOR_EQUAL(value2, data1.tensor(i));
            }
        }
        UTEST_CHECK_EQUAL(it.size(), 4);
        UTEST_CHECK_EQUAL(it.index(), 4);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
    {
        auto it = make_iterator(data2, mask2, data1, mask1, samples);
        for (auto i = 0; i < 4; ++ i, ++ it)
        {
            UTEST_CHECK_EQUAL(it.size(), 4);
            UTEST_CHECK_EQUAL(it.index(), i);
            UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
            {
                const auto [index, given1, value1, given2, value2] = *it;
                UTEST_CHECK_EQUAL(index, i);
                UTEST_CHECK_EQUAL(given1, getbit(mask2, i));
                UTEST_CHECK_EQUAL(given2, getbit(mask1, i));
                UTEST_CHECK_EQUAL(value1, data2(i));
                UTEST_CHECK_TENSOR_EQUAL(value2, data1.tensor(i));
            }
        }
        UTEST_CHECK_EQUAL(it.size(), 4);
        UTEST_CHECK_EQUAL(it.index(), 4);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
    {
        auto it = make_iterator(data2, mask2, data2, mask2, samples);
        for (auto i = 0; i < 4; ++ i, ++ it)
        {
            UTEST_CHECK_EQUAL(it.size(), 4);
            UTEST_CHECK_EQUAL(it.index(), i);
            UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
            {
                const auto [index, given1, value1, given2, value2] = *it;
                UTEST_CHECK_EQUAL(index, i);
                UTEST_CHECK_EQUAL(given1, getbit(mask2, i));
                UTEST_CHECK_EQUAL(given2, getbit(mask2, i));
                UTEST_CHECK_EQUAL(value1, data2(i));
                UTEST_CHECK_EQUAL(value2, data2(i));
            }
        }
        UTEST_CHECK_EQUAL(it.size(), 4);
        UTEST_CHECK_EQUAL(it.index(), 4);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
}

UTEST_CASE(loop_samples)
{
    const auto data1 = make_tensor<int>(make_dims(4, 2, 2, 1), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const auto data2 = make_tensor<int>(make_dims(4), -1, -2, -3, -4);

    auto mask1 = make_mask(make_dims(4));
    auto mask2 = make_mask(make_dims(4));

    const auto samples = arange(0, 4);

    {
        bool called = false;
        loop_samples2<1U, 1U>(data1, mask1, data2, mask2, samples, [&] (auto) { called = true; });
        UTEST_CHECK(!called);
    }
    {
        bool called = false;
        loop_samples2<2U, 1U>(data1, mask1, data2, mask2, samples, [&] (auto) { called = true; });
        UTEST_CHECK(!called);
    }
    {
        bool called = false;
        loop_samples2<4U, 4U>(data1, mask1, data2, mask2, samples, [&] (auto) { called = true; });
        UTEST_CHECK(!called);
    }
    {
        bool called = false;
        loop_samples2<4U, 1U>(data1, mask1, data2, mask2, samples, [&] (auto) { called = true; });
        UTEST_CHECK(called);
    }
    {
        bool called = false;
        loop_samples2<4U, 4U>(data1, mask1, data2.reshape(4, 1, 1, 1), mask2, samples, [&] (auto) { called = true; });
        UTEST_CHECK(called);
    }
    {
        bool called = false;
        loop_samples2<1U, 1U>(data1.reshape(-1), mask1, data2, mask2, samples, [&] (auto) { called = true; });
        UTEST_CHECK(called);
    }
}

UTEST_END_MODULE()
