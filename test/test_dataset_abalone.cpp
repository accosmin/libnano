#include <utest/utest.h>
#include <nano/dataset/tabular.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_abalone)

UTEST_CASE(load)
{
    const auto dataset = tabular_dataset_t::all().get("abalone");

    UTEST_REQUIRE(dataset);
    UTEST_REQUIRE_NOTHROW(dataset->load());

    UTEST_CHECK_EQUAL(dataset->features(), 8);
    UTEST_CHECK(dataset->target().discrete() && !dataset->target().optional());
    UTEST_CHECK(dataset->feature(0).discrete() && !dataset->feature(0).optional());
    UTEST_CHECK(!dataset->feature(1).discrete() && !dataset->feature(1).optional());
    UTEST_CHECK(!dataset->feature(2).discrete() && !dataset->feature(2).optional());
    UTEST_CHECK(!dataset->feature(3).discrete() && !dataset->feature(3).optional());
    UTEST_CHECK(!dataset->feature(4).discrete() && !dataset->feature(4).optional());
    UTEST_CHECK(!dataset->feature(5).discrete() && !dataset->feature(5).optional());
    UTEST_CHECK(!dataset->feature(6).discrete() && !dataset->feature(6).optional());
    UTEST_CHECK(!dataset->feature(7).discrete() && !dataset->feature(7).optional());

    UTEST_CHECK_EQUAL(dataset->idims(), make_dims(8, 1, 1));
    UTEST_CHECK_EQUAL(dataset->tdims(), make_dims(29, 1, 1));

    UTEST_CHECK_EQUAL(dataset->samples(), 4177);
    UTEST_CHECK_EQUAL(dataset->train_samples(), arange(0, 3133));
    UTEST_CHECK_EQUAL(dataset->test_samples(), arange(3133, 4177));

    UTEST_CHECK_EQUAL(dataset->type(), task_type::sclassification);
}

UTEST_END_MODULE()
