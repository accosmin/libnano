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
        for (tensor_size_t sample = 0; sample < m_samples; sample += 3)
        {
            hits(0) = sample % 2;
            hits(1) = 1 - (sample % 2);
            hits(2) = (sample % 6) == 0;
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

    static constexpr auto nan = std::numeric_limits<scalar_t>::quiet_NaN();

    static auto expected_select0() { return make_tensor<int32_t>(make_dims(10), 0, -1, -1, 1, -1, -1, 0, -1, -1, 1); }
    static auto expected_select1() { return make_tensor<int32_t>(make_dims(10), 1, -1, -1, 0, -1, -1, 1, -1, -1, 0); }
    static auto expected_select2() { return make_tensor<int32_t>(make_dims(10), 1, -1, -1, 0, -1, -1, 1, -1, -1, 0); }
    static auto expected_select3() { return make_tensor<int32_t>(make_dims(10), 0, 1, 0, 1, 0, 1, 0, 1, 0, 1); }
    static auto expected_select4() { return make_tensor<int32_t>(make_dims(10), 0, -1, 2, -1, 4, -1, 6, -1, 8, -1); }
    static auto expected_select5() { return make_tensor<scalar_t>(make_dims(10), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9); }
    static auto expected_select6() { return make_tensor<scalar_t>(make_dims(10), 7.0, nan, nan, 10.0, nan, nan, 13.0, nan, nan, 16.0); }
    static auto expected_select7() { return make_tensor<scalar_t>(make_dims(10, 2, 1, 2),
        1.0, 0.0, 0.0, 0.0, nan, nan, nan, nan,
        3.0, 2.0, 2.0, 2.0, nan, nan, nan, nan,
        5.0, 4.0, 4.0, 4.0, nan, nan, nan, nan,
        7.0, 6.0, 6.0, 6.0, nan, nan, nan, nan,
        9.0, 8.0, 8.0, 8.0, nan, nan, nan, nan); }
    static auto expected_select8() { return make_tensor<scalar_t>(make_dims(10), 1.0, nan, 3.0, nan, 5.0, nan, 7.0, nan, 9.0, nan); }
    static auto expected_select9() { return make_tensor<scalar_t>(make_dims(10), 0.0, nan, 2.0, nan, 4.0, nan, 6.0, nan, 8.0, nan); }
    static auto expected_select10() { return make_tensor<scalar_t>(make_dims(10), 0.0, nan, 2.0, nan, 4.0, nan, 6.0, nan, 8.0, nan); }
    static auto expected_select11() { return make_tensor<scalar_t>(make_dims(10), 0.0, nan, 2.0, nan, 4.0, nan, 6.0, nan, 8.0, nan); }
    static auto expected_select12() { return make_tensor<scalar_t>(make_dims(10), 1.0, nan, 3.0, nan, 5.0, nan, 7.0, nan, 9.0, nan); }

    static auto expected_flatten()
    {
        return make_tensor<scalar_t>(make_dims(10, 22),
            -1, +1, +1, +1, -1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7, 1, 0, 0, 0, 1,
            +0, +0, +0, -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, 1, 0, 0, 0, 0, 0, 0,
            +0, +0, +0, +1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, 2, 0, 3, 2, 2, 2, 3,
            +1, -1, -1, -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, 3, 10, 0, 0, 0, 0, 0,
            +0, +0, +0, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, 4, 0, 5, 4, 4, 4, 5,
            +0, +0, +0, -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, 5, 0, 0, 0, 0, 0, 0,
            -1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, 6, 13, 7, 6, 6, 6, 7,
            +0, +0, +0, -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, 7, 0, 0, 0, 0, 0, 0,
            +0, +0, +0, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, -1, 8, 0, 9, 8, 8, 8, 9,
            +1, -1, -1, -1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, 9, 16, 0, 0, 0, 0, 0);
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

template <typename tgenerator>
static void check_sclass(const tgenerator& generator, tensor_size_t feature, indices_cmap_t samples, const sclass_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    UTEST_CHECK_NOTHROW(generator.select(feature, samples, sclass_buffer));
    UTEST_CHECK_THROW(generator.select(feature, samples, scalar_buffer), std::runtime_error);
    UTEST_CHECK_THROW(generator.select(feature, samples, struct_buffer), std::runtime_error);
    UTEST_CHECK_TENSOR_EQUAL(sclass_buffer, expected);
}

template <typename tgenerator>
static void check_scalar(const tgenerator& generator, tensor_size_t feature, indices_cmap_t samples, const scalar_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    UTEST_CHECK_THROW(generator.select(feature, samples, sclass_buffer), std::runtime_error);
    UTEST_CHECK_NOTHROW(generator.select(feature, samples, scalar_buffer));
    UTEST_CHECK_THROW(generator.select(feature, samples, struct_buffer), std::runtime_error);
    UTEST_CHECK_TENSOR_CLOSE(scalar_buffer, expected, 1e-12);
}

template <typename tgenerator>
static void check_struct(const tgenerator& generator, tensor_size_t feature, indices_cmap_t samples, const struct_mem_t& expected)
{
    sclass_mem_t sclass_buffer;
    scalar_mem_t scalar_buffer;
    struct_mem_t struct_buffer;

    UTEST_CHECK_THROW(generator.select(feature, samples, sclass_buffer), std::runtime_error);
    UTEST_CHECK_THROW(generator.select(feature, samples, scalar_buffer), std::runtime_error);
    UTEST_CHECK_NOTHROW(generator.select(feature, samples, struct_buffer));
    UTEST_CHECK_TENSOR_CLOSE(struct_buffer, expected, 1e-12);
}

// TODO: check that the flatten & the feature iterators work as expected
// TODO: check that feature scaling scaling works
// TODO: check that feature extraction works (e.g sign(x), sign(x)*log(1+x^2), polynomial expansion)

UTEST_BEGIN_MODULE(test_dataset_generator)

UTEST_CASE(identity)
{
    const auto samples = ::nano::arange(0, 10);
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

    cluster_t original(dataset.features(), 1);
    generator.original(0, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(1), 0));
    generator.original(1, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(1), 0));
    generator.original(2, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(1), 0));
    generator.original(3, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(2), 0, 1));
    generator.original(4, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(3), 0, 1, 2));
    generator.original(5, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(4), 0, 1, 2, 3));
    generator.original(6, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(5), 0, 1, 2, 3, 4));
    generator.original(7, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(6), 0, 1, 2, 3, 4, 5));
    generator.original(8, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(6), 0, 1, 2, 3, 4, 5));
    generator.original(9, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(6), 0, 1, 2, 3, 4, 5));
    generator.original(10, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(6), 0, 1, 2, 3, 4, 5));
    generator.original(11, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(6), 0, 1, 2, 3, 4, 5));
    generator.original(12, original);
    UTEST_CHECK_TENSOR_EQUAL(original.indices(0), make_tensor<tensor_size_t>(make_dims(7), 0, 1, 2, 3, 4, 5, 6));

    check_sclass(generator, 0, samples, dataset.expected_select0());
    check_sclass(generator, 1, samples, dataset.expected_select1());
    check_sclass(generator, 2, samples, dataset.expected_select2());
    check_sclass(generator, 3, samples, dataset.expected_select3());
    check_sclass(generator, 4, samples, dataset.expected_select4());
    check_scalar(generator, 5, samples, dataset.expected_select5());
    check_scalar(generator, 6, samples, dataset.expected_select6());
    check_struct(generator, 7, samples, dataset.expected_select7());
    check_scalar(generator, 8, samples, dataset.expected_select8());
    check_scalar(generator, 9, samples, dataset.expected_select9());
    check_scalar(generator, 10, samples, dataset.expected_select10());
    check_scalar(generator, 11, samples, dataset.expected_select11());
    check_scalar(generator, 12, samples, dataset.expected_select12());

    tensor2d_t flatten_buffer(samples.size(), 22);
    UTEST_CHECK_EQUAL(generator.columns(), 22);
    UTEST_CHECK_NOTHROW(generator.flatten(samples, flatten_buffer.tensor(), 0));
    UTEST_CHECK_TENSOR_CLOSE(flatten_buffer, dataset.expected_flatten(), 1e-12);
}

UTEST_CASE(unsupervised)
{
    const auto samples = ::nano::arange(0, 10);
    const auto dataset = make_dataset(samples.size(), string_t::npos);

    auto generator = dataset_generator_t{dataset, samples};
    generator.add<identity_generator_t>();

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

    {
        const auto features = make_tensor<tensor_size_t>(make_dims(2), 0, 5);
        const auto expected = make_tensor<tensor_size_t>(make_dims(2), 0, 3);
        UTEST_CHECK_TENSOR_EQUAL(generator.original(features), expected);
    }
    {
        const auto features = make_tensor<tensor_size_t>(make_dims(5), 0, 1, 5, 11, 12);
        const auto expected = make_tensor<tensor_size_t>(make_dims(4), 0, 3, 5, 6);
        UTEST_CHECK_TENSOR_EQUAL(generator.original(features), expected);
    }

    check_sclass(generator, 0, samples, dataset.expected_select0());
    check_sclass(generator, 1, samples, dataset.expected_select1());
    check_sclass(generator, 2, samples, dataset.expected_select2());
    check_sclass(generator, 3, samples, dataset.expected_select3());
    check_sclass(generator, 4, samples, dataset.expected_select4());
    check_scalar(generator, 5, samples, dataset.expected_select5());
    check_scalar(generator, 6, samples, dataset.expected_select6());
    check_struct(generator, 7, samples, dataset.expected_select7());
    check_scalar(generator, 8, samples, dataset.expected_select8());
    check_scalar(generator, 9, samples, dataset.expected_select9());
    check_scalar(generator, 10, samples, dataset.expected_select10());
    check_scalar(generator, 11, samples, dataset.expected_select11());
    check_scalar(generator, 12, samples, dataset.expected_select12());

    tensor2d_t flatten_buffer;
    tensor2d_cmap_t flatten_cmap;
    UTEST_CHECK_EQUAL(generator.columns(), 22);
    UTEST_CHECK_NOTHROW(flatten_cmap = generator.flatten(make_range(0, samples.size()), flatten_buffer));
    UTEST_CHECK_TENSOR_CLOSE(flatten_cmap, dataset.expected_flatten(), 1e-12);

    UTEST_CHECK_EQUAL(generator.target(), feature_t{});
    UTEST_CHECK_EQUAL(generator.target_dims(), make_dims(0, 0, 0));

    // TODO: check targets
    // TODO: check select, flatten and target stats
}

UTEST_END_MODULE()
