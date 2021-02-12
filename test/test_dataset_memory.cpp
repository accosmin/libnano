#include <utest/utest.h>
#include <nano/dataset/memory.h>

using namespace nano;

// TODO: create in-memory dataset with various feature types (sclass, mclass, scalar or structured) w/o optional
// TODO: check that the flatten & the feature iterators work as expected
// TODO: check that feature normalization works
// TODO: check that feature extraction works (e.g sign(x), sign(x)*log(1+x^2), polynomial expansion)

UTEST_BEGIN_MODULE(test_dataset_memory)

UTEST_CASE(resize)
{

}

UTEST_END_MODULE()
