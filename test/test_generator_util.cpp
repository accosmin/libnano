#include <utest/utest.h>
#include <nano/generator/util.h>

using namespace nano;

template <typename tscalar>
std::ostream& operator<<(std::ostream& stream, const std::vector<tscalar>& values)
{
    for (const auto& value : values)
    {
        stream << value << ',';
    }
    return stream;
}

static auto make_features()
{
    return features_t
    {
        feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}),
        feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)),
        feature_t{"f64"}.scalar(feature_type::float64),
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
        for (tensor_size_t sample = 0; sample < m_samples; sample += 3)
        {
            hits(0) = sample % 2;
            hits(1) = 1 - (sample % 2);
            hits(2) = (sample % 6) == 0;
            set(sample, 0, hits);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 1, (sample % 3 == 0) ? 0 : 1);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 2, sample);
        }

        tensor_mem_t<tensor_size_t, 3> values(2, 1, 2);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            values.full(sample);
            values(0) = sample + 1;
            set(sample, 3, values);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 4, 1 - sample);
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

UTEST_BEGIN_MODULE(test_generator_util)

UTEST_CASE(select_scalar_components)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, indices_t{});
        const auto expected = std::vector<tensor_size_t>{2, 0, 4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(2));
        const auto expected = std::vector<tensor_size_t>{2, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(3));
        const auto expected = std::vector<tensor_size_t>{};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(2, 3, 4));
        const auto expected = std::vector<tensor_size_t>{2, 0, 4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, indices_t{});
        const auto expected = std::vector<tensor_size_t>{2, 0, 3, 0, 3, 1, 3, 2, 3, 3, 4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, make_indices(1, 4));
        const auto expected = std::vector<tensor_size_t>{4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, make_indices(1, 3,  4));
        const auto expected = std::vector<tensor_size_t>{3, 0, 3, 1, 3, 2, 3, 3, 4, 0};
        UTEST_CHECK_EQUAL(indices, expected);
    }
}

UTEST_END_MODULE()
