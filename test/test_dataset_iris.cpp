#include <utest/utest.h>
#include <nano/dataset/tabular.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_iris)

UTEST_CASE(load)
{
    const auto dataset = tabular_dataset_t::all().get("iris");

    UTEST_REQUIRE(dataset);
    UTEST_REQUIRE_NOTHROW(dataset->load());

    UTEST_CHECK_EQUAL(dataset->features(), 4);
    UTEST_CHECK(dataset->target().discrete() && !dataset->target().optional());
    UTEST_CHECK(!dataset->feature(0).discrete() && !dataset->feature(0).optional());
    UTEST_CHECK(!dataset->feature(1).discrete() && !dataset->feature(1).optional());
    UTEST_CHECK(!dataset->feature(2).discrete() && !dataset->feature(2).optional());
    UTEST_CHECK(!dataset->feature(3).discrete() && !dataset->feature(3).optional());

    UTEST_CHECK_EQUAL(dataset->idims(), make_dims(4, 1, 1));
    UTEST_CHECK_EQUAL(dataset->tdims(), make_dims(3, 1, 1));

    UTEST_CHECK_EQUAL(dataset->samples(), 150);
    UTEST_CHECK_EQUAL(dataset->test_samples(), arange(0, 0));
    UTEST_CHECK_EQUAL(dataset->train_samples(), arange(0, 150));

    UTEST_CHECK_EQUAL(dataset->type(), task_type::sclassification);
}

UTEST_END_MODULE()
