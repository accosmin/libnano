#include <utest/utest.h>
#include <nano/dataset/mask.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_mask)

UTEST_CASE(mask)
{
    for (const tensor_size_t samples : {1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33})
    {
        auto mask = make_mask(make_dims(samples));
        UTEST_CHECK_EQUAL(mask.size(), ((samples + 7) / 8));
        UTEST_CHECK(optional(mask, samples));

        for (auto sample = tensor_size_t{0}; sample < samples; ++ sample)
        {
            UTEST_CHECK(!getbit(mask, sample));
        }

        for (auto sample = tensor_size_t{0}; sample < samples; sample += 3)
        {
            setbit(mask, sample);
        }
        UTEST_CHECK(optional(mask, samples) == (samples > 1));

        for (auto sample = tensor_size_t{0}; sample < samples; ++ sample)
        {
            const auto bit = (sample % 3) == 0;
            UTEST_CHECK(getbit(mask, sample) == bit);
        }

        for (auto sample = tensor_size_t{0}; sample < samples; ++ sample)
        {
            setbit(mask, sample);
        }
        UTEST_CHECK(!optional(mask, samples));

        for (auto sample = tensor_size_t{0}; sample < samples; ++ sample)
        {
            UTEST_CHECK(getbit(mask, sample));
        }
    }
}

UTEST_END_MODULE()
