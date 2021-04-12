#include <utest/utest.h>
#include <nano/dataset/memfixed.h>

using namespace nano;

class fixture_dataset_t final : public memfixed_dataset_t<uint8_t>
{
public:

    using memfixed_dataset_t::resize;
    using memfixed_dataset_t::target;

    void load() override
    {
        for (tensor_size_t s = 0; s < samples(); ++ s)
        {
            auto&& input = this->input(s);
            for (tensor_size_t f = 0; f < features(); ++ f)
            {
                input(f) = value(s, f);
            }
            target(s).full(static_cast<scalar_t>(-s));
        }
    }

    static uint8_t value(tensor_size_t sample, tensor_size_t feature)
    {
        return static_cast<uint8_t>((sample + feature) % 256);
    }

    feature_t target() const override
    {
        return feature_t{"fixture"};
    }
};

inline void check_targets(const tensor4d_t& targets, tensor_range_t range)
{
    UTEST_REQUIRE_EQUAL(targets.dims(), nano::make_dims(range.size(), 3, 1, 1));
    for (tensor_size_t s = range.begin(); s < range.end(); ++ s)
    {
        const auto row = s - range.begin();
        UTEST_CHECK_CLOSE(targets.vector(row).minCoeff(), static_cast<scalar_t>(-s), 1e-8);
        UTEST_CHECK_CLOSE(targets.vector(row).maxCoeff(), static_cast<scalar_t>(-s), 1e-8);
    }
}
