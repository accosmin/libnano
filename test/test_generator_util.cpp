#include <utest/utest.h>
#include <nano/generator/util.h>

using namespace nano;

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

UTEST_CASE(select_scalar)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        const auto mapping = select_scalar(dataset, struct2scalar::off);
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(2, 6),
            2, -1, 0, 1, 1, 1,
            4, -1, 0, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping = select_scalar(dataset, struct2scalar::on);
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(6, 6),
            2, -1, 0, 1, 1, 1,
            3, 0, 0, 2, 1, 2,
            3, 1, 0, 2, 1, 2,
            3, 2, 0, 2, 1, 2,
            3, 3, 0, 2, 1, 2,
            4, -1, 0, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping = select_scalar(dataset, struct2scalar::on, make_indices(0, 1, 3, 4));
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(5, 6),
            3, 0, 0, 2, 1, 2,
            3, 1, 0, 2, 1, 2,
            3, 2, 0, 2, 1, 2,
            3, 3, 0, 2, 1, 2,
            4, -1, 0, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
}

UTEST_CASE(for_each_struct)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        const auto mapping = select_struct(dataset);
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(1, 6),
            3, -1, 0, 2, 1, 2);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping = select_struct(dataset, make_indices(2, 3, 4));
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(1, 6),
            3, -1, 0, 2, 1, 2);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping = select_struct(dataset, make_indices(2, 4));
        const auto expected_mapping = feature_mapping_t{0, 6};
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
}

UTEST_CASE(for_each_sclass)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        const auto mapping = select_sclass(dataset, sclass2binary::off);
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(1, 6),
            1, -1, 2, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping = select_sclass(dataset, sclass2binary::on);
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(2, 6),
            1, 0, 2, 1, 1, 1,
            1, 1, 2, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping = select_sclass(dataset, sclass2binary::on, make_indices(0, 1, 2));
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(2, 6),
            1, 0, 2, 1, 1, 1,
            1, 1, 2, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
}

UTEST_CASE(for_each_mclass)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        const auto mapping = select_mclass(dataset, mclass2binary::off);
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(1, 6),
            0, -1, 3, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping = select_mclass(dataset, mclass2binary::on);
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(3, 6),
            0, 0, 3, 1, 1, 1,
            0, 1, 3, 1, 1, 1,
            0, 2, 3, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping = select_mclass(dataset, mclass2binary::on, make_indices(0, 1, 2, 3, 4));
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(3, 6),
            0, 0, 3, 1, 1, 1,
            0, 1, 3, 1, 1, 1,
            0, 2, 3, 1, 1, 1);
        UTEST_CHECK_TENSOR_EQUAL(mapping, expected_mapping);
    }
}

UTEST_END_MODULE()
