#include <utest/utest.h>
#include <nano/dataset/generator.h>

using namespace nano;

static auto make_features()
{
    return features_t
    {
        feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}),
        feature_t{"sclass2"}.sclass(2),
        feature_t{"sclass10"}.sclass(10),
        feature_t{"i8"}.scalar(feature_type::int8),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)),
        feature_t{"u16_struct"}.scalar(feature_type::uint16, make_dims(1, 1, 1))
    };
}

class fixture_dataset_t final : public memory_dataset_t
{
public:

    fixture_dataset_t(tensor_size_t samples, size_t target) :
        m_samples(samples),
        m_features(make_features()),
        m_target(target)
    {
    }

    void load() override
    {
        resize(m_samples, m_features, m_target);

        tensor_mem_t<tensor_size_t, 1> hits(3);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            hits(0) = sample % 2;
            hits(1) = 1 - (sample % 2);
            hits(2) = (sample % 4) == 0;
            set(sample, 0, hits);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 1, sample % 2);
        }
        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            set(sample, 2, sample % 10);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 3, sample);
        }
        for (tensor_size_t sample = 0; sample < m_samples; sample += 3)
        {
            set(sample, 4, sample + 7);
        }

        tensor_mem_t<tensor_size_t, 3> values(2, 1, 2);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            values.constant(sample);
            values(0) = sample + 1;
            set(sample, 5, values);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            set(sample, 6, sample + 1);
        }
    }

private:

    tensor_size_t   m_samples{0};
    features_t      m_features;
    size_t          m_target;
};

static auto make_dataset(tensor_size_t samples, size_t target)
{
    auto dataset = fixture_dataset_t{samples, target};
    UTEST_CHECK_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    return dataset;
}

// TODO: check feature indices selection
// TODO: check feature access with both iterators
// TODO: check that the flatten & the feature iterators work as expected
// TODO: check that feature normalization works
// TODO: check that feature extraction works (e.g sign(x), sign(x)*log(1+x^2), polynomial expansion)

UTEST_BEGIN_MODULE(test_dataset_generator)

UTEST_CASE(identity)
{
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), string_t::npos);

    const auto generator = identity_generator_t{dataset, samples};

    UTEST_CHECK_EQUAL(generator.features(), 13);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"mclass3_m0"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"mclass3_m1"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"mclass3_m2"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"sclass2"}.sclass(2));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"sclass10"}.sclass(10));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"i8"}.scalar(feature_type::int8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(6), feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(7), feature_t{"u8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)));
    UTEST_CHECK_EQUAL(generator.feature(8), feature_t{"u8_struct_0"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(9), feature_t{"u8_struct_1"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(10), feature_t{"u8_struct_2"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(11), feature_t{"u8_struct_3"}.scalar(feature_type::uint8, make_dims(1, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(12), feature_t{"u16_struct"}.scalar(feature_type::uint16, make_dims(1, 1, 1)));

    UTEST_CHECK_EQUAL(generator.columns(), 22);
}

UTEST_END_MODULE()
