#include <utest/utest.h>
#include <nano/dataset/imclass.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_cifar100c)

UTEST_CASE(load)
{
    const auto dataset = imclass_dataset_t::all().get("cifar100c");

    UTEST_REQUIRE(dataset);
    UTEST_REQUIRE_NOTHROW(dataset->load());

    UTEST_CHECK(dataset->target().discrete());
    UTEST_CHECK(!dataset->target().optional());
    UTEST_CHECK_EQUAL(dataset->target().labels().size(), 20U);

    UTEST_CHECK_EQUAL(dataset->idims(), make_dims(32, 32, 3));
    UTEST_CHECK_EQUAL(dataset->tdims(), make_dims(20, 1, 1));

    UTEST_CHECK_EQUAL(dataset->samples(), 60000);
    UTEST_CHECK_EQUAL(dataset->train_samples(), arange(0, 50000));
    UTEST_CHECK_EQUAL(dataset->test_samples(), arange(50000, 60000));

    UTEST_CHECK_EQUAL(dataset->type(), task_type::sclassification);
}

UTEST_END_MODULE()
