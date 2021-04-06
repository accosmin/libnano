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
            const auto [index, sample, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 0);
            UTEST_CHECK_EQUAL(sample, 5);
            UTEST_CHECK_EQUAL(given, false);
            UTEST_CHECK_EQUAL(value, -1);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 1);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, sample, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 1);
            UTEST_CHECK_EQUAL(sample, 6);
            UTEST_CHECK_EQUAL(given, true);
            UTEST_CHECK_EQUAL(value, 9);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 2);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, sample, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 2);
            UTEST_CHECK_EQUAL(sample, 7);
            UTEST_CHECK_EQUAL(given, false);
            UTEST_CHECK_EQUAL(value, -1);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 3);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, sample, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 3);
            UTEST_CHECK_EQUAL(sample, 8);
            UTEST_CHECK_EQUAL(given, true);
            UTEST_CHECK_EQUAL(value, 11);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 4);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        {
            const auto [index, sample, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 4);
            UTEST_CHECK_EQUAL(sample, 9);
            UTEST_CHECK_EQUAL(given, false);
            UTEST_CHECK_EQUAL(value, -1);
        }

        ++ it;
        UTEST_CHECK_EQUAL(it.size(), 5);
        UTEST_CHECK_EQUAL(it.index(), 5);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), false);
    }
    for (const auto& [samples, shuffled, expected_indices, expected_samplex, expected_givens, expected_values] :
        {
            std::make_tuple(
                arange(4, 16),
                indices_t{},
                std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                std::vector<int>{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                std::vector<int>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                std::vector<int>{7, -1, 9, -1, 11, -1, 13, -1, 15, -1, 17, -1}
            ),
            std::make_tuple(
                arange(4, 16),
                make_tensor<tensor_size_t>(make_dims(16), 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
                std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                std::vector<int>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
                std::vector<int>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
                std::vector<int>{-1, 13, -1, 11, -1, 9, -1, 7, -1, 5, -1, 3}
            )
        })
    {
        auto it = make_iterator(data, mask, samples, shuffled);
        const auto it_end = make_end_iterator(data, mask, samples);
        UTEST_CHECK_EQUAL(it.size(), 12);
        UTEST_CHECK_EQUAL(it.index(), 0);
        UTEST_CHECK_EQUAL(static_cast<bool>(it), true);
        UTEST_CHECK_EQUAL(it_end.size(), 12);
        UTEST_CHECK_EQUAL(it_end.index(), 12);
        UTEST_CHECK_EQUAL(static_cast<bool>(it_end), false);

        std::vector<int> indices, samplex, givens, values;
        for ( ; it != it_end; ++ it)
        {
            const auto [index, sample, given, value] = *it;
            indices.push_back(static_cast<int>(index));
            samplex.push_back(static_cast<int>(sample));
            givens.push_back(given);
            values.push_back(value);
        }

        UTEST_CHECK_EQUAL(indices, expected_indices);
        UTEST_CHECK_EQUAL(samplex, expected_samplex);
        UTEST_CHECK_EQUAL(givens, expected_givens);
        UTEST_CHECK_EQUAL(values, expected_values);

        indices.clear();
        samplex.clear();
        givens.clear();
        values.clear();
        for (auto it = make_iterator(data, mask, samples, shuffled); it; ++ it)
        {
            const auto [index, sample, given, value] = *it;
            indices.push_back(static_cast<int>(index));
            samplex.push_back(static_cast<int>(sample));
            givens.push_back(given);
            values.push_back(value);
        }

        UTEST_CHECK_EQUAL(indices, expected_indices);
        UTEST_CHECK_EQUAL(samplex, expected_samplex);
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
            const auto [index, sample, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 0);
            UTEST_CHECK_EQUAL(sample, 5);
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
            const auto [index, sample, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 1);
            UTEST_CHECK_EQUAL(sample, 6);
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
            const auto [index, sample, given, value] = *it;
            UTEST_CHECK_EQUAL(index, 2);
            UTEST_CHECK_EQUAL(sample, 7);
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
