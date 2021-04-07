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
    data.constant(-1);

    auto mask = make_mask(make_dims(16));

    for (int sample = 0; sample < 16; sample += 2)
    {
        setbit(mask, sample);
        data(sample) = sample + 3;
    }

    {
        const auto it = feature_iterator_t<int, 1>{};
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
        const auto it_end = make_end_iterator(data, mask, samples);
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
    data.constant(-1);

    auto mask = make_mask(make_dims(16));

    for (int sample = 0; sample < 16; sample += 2)
    {
        setbit(mask, sample);
        data.tensor(sample).constant(sample + 3);
    }

    {
        const auto it = feature_iterator_t<int, 1>{};
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

UTEST_END_MODULE()
