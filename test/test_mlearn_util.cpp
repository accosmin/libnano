#include <utest/utest.h>
#include <nano/mlearn/util.h>

using namespace nano;

template <typename tindex, std::enable_if_t<std::is_integral_v<tindex>, bool> = true>
static tensor_mem_t<tindex, 2> exhaustive(const tensor_mem_t<tindex, 1>& counts)
{
    auto iter = combinatorial_iterator_t{counts};

    const auto dimensions = counts.size();
    const auto combinations = iter.size();

    tensor_mem_t<tindex, 2> product(combinations, dimensions);
    for (; iter; ++ iter)
    {
        product.tensor(iter.index()) = *iter;
    }

    return product;
}

UTEST_BEGIN_MODULE(test_mlearn_util)

UTEST_CASE(sample_with_replacement)
{
    for (auto trial = 0; trial < 100; ++ trial)
    {
        const auto indices = nano::sample_with_replacement(120, 50);

        UTEST_CHECK_EQUAL(indices.size(), 50);
        UTEST_CHECK_LESS(indices.max(), 120);
        UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
        UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
    }
}

UTEST_CASE(sample_without_replacement)
{
    for (auto trial = 0; trial < 100; ++ trial)
    {
        const auto indices = nano::sample_without_replacement(140, 60);

        UTEST_CHECK_EQUAL(indices.size(), 60);
        UTEST_CHECK_LESS(indices.max(), 140);
        UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
        UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
        UTEST_CHECK(std::adjacent_find(begin(indices), end(indices)) == end(indices));
    }
}

UTEST_CASE(sample_without_replacement_all)
{
    const auto indices = nano::sample_without_replacement(100, 100);

    UTEST_CHECK_EQUAL(indices, arange(0, 100));
}

UTEST_CASE(exhaustive)
{
    const auto config1 = make_tensor<tensor_size_t>(make_dims(1), 3);
    const auto config2 = make_tensor<tensor_size_t>(make_dims(2), 3, 2);
    const auto config3 = make_tensor<tensor_size_t>(make_dims(3), 3, 2, 2);
    const auto config4 = make_tensor<tensor_size_t>(make_dims(3), 2, 3, 3);

    const auto product1 = make_tensor<tensor_size_t>(make_dims(3, 1), 0, 1, 2);
    const auto product2 = make_tensor<tensor_size_t>(make_dims(6, 2),
        0, 0, 0, 1,
        1, 0, 1, 1,
        2, 0, 2, 1);
    const auto product3 = make_tensor<tensor_size_t>(make_dims(12, 3),
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
        2, 0, 0, 2, 0, 1, 2, 1, 0, 2, 1, 1);
    const auto product4 = make_tensor<tensor_size_t>(make_dims(18, 3),
        0, 0, 0, 0, 0, 1, 0, 0, 2,
        0, 1, 0, 0, 1, 1, 0, 1, 2,
        0, 2, 0, 0, 2, 1, 0, 2, 2,
        1, 0, 0, 1, 0, 1, 1, 0, 2,
        1, 1, 0, 1, 1, 1, 1, 1, 2,
        1, 2, 0, 1, 2, 1, 1, 2, 2);

    UTEST_CHECK_EQUAL(exhaustive(config1), product1);
    UTEST_CHECK_EQUAL(exhaustive(config2), product2);
    UTEST_CHECK_EQUAL(exhaustive(config3), product3);
    UTEST_CHECK_EQUAL(exhaustive(config4), product4);
}

UTEST_END_MODULE()
