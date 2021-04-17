#include <utest/utest.h>
#include <nano/dataset.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_factory)

UTEST_CASE(factory)
{
    const auto& datasets = dataset_t::all();
    UTEST_CHECK_EQUAL(datasets.ids().size(), 12U);
    UTEST_CHECK(datasets.get("abalone") != nullptr);
    UTEST_CHECK(datasets.get("adult") != nullptr);
    UTEST_CHECK(datasets.get("bank-marketing") != nullptr);
    UTEST_CHECK(datasets.get("breast-cancer") != nullptr);
    UTEST_CHECK(datasets.get("forest-fires") != nullptr);
    UTEST_CHECK(datasets.get("iris") != nullptr);
    UTEST_CHECK(datasets.get("wine") != nullptr);
    UTEST_CHECK(datasets.get("cifar100c") != nullptr);
    UTEST_CHECK(datasets.get("cifar100f") != nullptr);
    UTEST_CHECK(datasets.get("cifar10") != nullptr);
    UTEST_CHECK(datasets.get("fashion-mnist") != nullptr);
    UTEST_CHECK(datasets.get("mnist") != nullptr);
}

UTEST_END_MODULE()
