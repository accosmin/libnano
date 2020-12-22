#include <utest/utest.h>
#include <nano/dataset/imclass.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_fashion_mnist)

UTEST_CASE(load)
{
    const auto dataset = imclass_dataset_t::all().get("fashion-mnist");

    UTEST_REQUIRE(dataset);
    UTEST_REQUIRE_NOTHROW(dataset->load());

    UTEST_CHECK(dataset->target().discrete());
    UTEST_CHECK(!dataset->target().optional());
    UTEST_CHECK_EQUAL(dataset->target().labels().size(), 10U);

    UTEST_CHECK_EQUAL(dataset->idims(), make_dims(28, 28, 1));
    UTEST_CHECK_EQUAL(dataset->tdims(), make_dims(10, 1, 1));

    UTEST_CHECK_EQUAL(dataset->samples(), 70000);
    UTEST_CHECK_EQUAL(dataset->train_samples(), arange(0, 60000));
    UTEST_CHECK_EQUAL(dataset->test_samples(), arange(60000, 70000));

    UTEST_CHECK_EQUAL(dataset->type(), task_type::sclassification);
}

UTEST_END_MODULE()
