#include <utest/utest.h>
#include <nano/mlearn/cluster.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_mlearn_cluster)

UTEST_CASE(_default)
{
    const auto split = cluster_t{};
    UTEST_CHECK_EQUAL(split.groups(), 0);
    UTEST_CHECK_EQUAL(split.samples(), 0);
}

UTEST_CASE(empty)
{
    const auto split = cluster_t{7};
    UTEST_CHECK_EQUAL(split.groups(), 1);
    UTEST_CHECK_EQUAL(split.count(0), 0);
    UTEST_CHECK_EQUAL(split.samples(), 7);
}

UTEST_CASE(assign)
{
    auto split = cluster_t{7, 3};
    UTEST_CHECK_EQUAL(split.groups(), 3);
    UTEST_CHECK_EQUAL(split.count(0), 0);
    UTEST_CHECK_EQUAL(split.count(1), 0);
    UTEST_CHECK_EQUAL(split.count(2), 0);
    UTEST_CHECK_EQUAL(split.samples(), 7);

    split.assign(0, 0);
    split.assign(1, 0);
    split.assign(2, 1);
    split.assign(3, 1);
    split.assign(4, 2);
    split.assign(5, 2);
    split.assign(6, 1);

    UTEST_CHECK_EQUAL(split.groups(), 3);
    UTEST_CHECK_EQUAL(split.count(0), 2);
    UTEST_CHECK_EQUAL(split.count(1), 3);
    UTEST_CHECK_EQUAL(split.count(2), 2);
    UTEST_CHECK_EQUAL(split.samples(), 7);
    UTEST_CHECK_EQUAL(split.group(0), 0);
    UTEST_CHECK_EQUAL(split.group(1), 0);
    UTEST_CHECK_EQUAL(split.group(2), 1);
    UTEST_CHECK_EQUAL(split.group(3), 1);
    UTEST_CHECK_EQUAL(split.group(4), 2);
    UTEST_CHECK_EQUAL(split.group(5), 2);
    UTEST_CHECK_EQUAL(split.group(6), 1);

    split.assign(4, 1);
    split.assign(5, 1);
    split.assign(6, 2);
    split.assign(6, 1);

    UTEST_CHECK_EQUAL(split.groups(), 3);
    UTEST_CHECK_EQUAL(split.count(0), 2);
    UTEST_CHECK_EQUAL(split.count(1), 5);
    UTEST_CHECK_EQUAL(split.count(2), 0);
    UTEST_CHECK_EQUAL(split.samples(), 7);

    const auto indices0 = split.indices(0);
    const auto indices1 = split.indices(1);
    const auto indices2 = split.indices(2);

    UTEST_REQUIRE_EQUAL(indices0.size(), 2);
    UTEST_CHECK_EQUAL(indices0(0), 0);
    UTEST_CHECK_EQUAL(indices0(1), 1);

    UTEST_REQUIRE_EQUAL(indices1.size(), 5);
    UTEST_CHECK_EQUAL(indices1(0), 2);
    UTEST_CHECK_EQUAL(indices1(1), 3);
    UTEST_CHECK_EQUAL(indices1(2), 4);
    UTEST_CHECK_EQUAL(indices1(3), 5);
    UTEST_CHECK_EQUAL(indices1(4), 6);

    UTEST_REQUIRE_EQUAL(indices2.size(), 0);
}

UTEST_CASE(loop)
{
    auto indices = indices_t{3};
    indices(0) = 0;
    indices(1) = 4;
    indices(2) = 5;

    auto split = cluster_t{7, indices};
    UTEST_CHECK_EQUAL(split.groups(), 1);
    UTEST_CHECK_EQUAL(split.count(0), 3);
    UTEST_CHECK_EQUAL(split.samples(), 7);

    indices_t all_indices(7);
    all_indices.full(-1);
    split.loop(0, [&] (const tensor_size_t index) { all_indices(index) = +1; });
    UTEST_CHECK_EQUAL(all_indices(0), +1);
    UTEST_CHECK_EQUAL(all_indices(1), -1);
    UTEST_CHECK_EQUAL(all_indices(2), -1);
    UTEST_CHECK_EQUAL(all_indices(3), -1);
    UTEST_CHECK_EQUAL(all_indices(4), +1);
    UTEST_CHECK_EQUAL(all_indices(5), +1);
    UTEST_CHECK_EQUAL(all_indices(6), -1);

    split.assign(3, 0);

    UTEST_CHECK_EQUAL(split.groups(), 1);
    UTEST_CHECK_EQUAL(split.count(0), 4);
    UTEST_CHECK_EQUAL(split.samples(), 7);

    split.loop(0, [&] (const tensor_size_t index) { all_indices(index) = +1; });
    UTEST_CHECK_EQUAL(all_indices(0), +1);
    UTEST_CHECK_EQUAL(all_indices(1), -1);
    UTEST_CHECK_EQUAL(all_indices(2), -1);
    UTEST_CHECK_EQUAL(all_indices(3), +1);
    UTEST_CHECK_EQUAL(all_indices(4), +1);
    UTEST_CHECK_EQUAL(all_indices(5), +1);
    UTEST_CHECK_EQUAL(all_indices(6), -1);
}

UTEST_END_MODULE()
