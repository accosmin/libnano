#include <utest/utest.h>
#include <nano/dataset/dataset.h>

using namespace nano;

template <typename tvalue, size_t trank>
static auto make_tensor(tvalue value, tensor_dims_t<trank> dims)
{
    tensor_mem_t<tvalue, trank> values(dims);
    values.constant(value);
    return values;
}

template <typename tscalar, size_t trank>
static auto check_inputs(const memory_dataset_t& dataset, tensor_size_t index,
    const feature_t& gt_feature, const tensor_mem_t<tscalar, trank>& gt_data, const mask_cmap_t& gt_mask)
{
    dataset.visit_inputs(index, [&] (const auto& feature, const auto& data, const auto& mask)
    {
        UTEST_CHECK_EQUAL(feature, gt_feature);
        if constexpr (std::is_same<decltype(data), const tensor_cmap_t<tscalar, trank>&>::value)
        {
            UTEST_CHECK_TENSOR_EQUAL(data, gt_data);
            UTEST_CHECK_TENSOR_EQUAL(mask, gt_mask);
        }
        else
        {
            UTEST_REQUIRE(false);
        }
    });
}

template <typename tscalar, size_t trank>
static auto check_target(const memory_dataset_t& dataset,
    const feature_t& gt_feature, const tensor_mem_t<tscalar, trank>& gt_data, const mask_cmap_t& gt_mask)
{
    dataset.visit_target([&] (const auto& feature, const auto& data, const auto& mask)
    {
        UTEST_CHECK_EQUAL(feature, gt_feature);
        if constexpr (std::is_same<decltype(data), const tensor_cmap_t<tscalar, trank>&>::value)
        {
            UTEST_CHECK_TENSOR_EQUAL(data, gt_data);
            UTEST_CHECK_TENSOR_EQUAL(mask, gt_mask);
        }
        else
        {
            UTEST_REQUIRE(false);
        }
    });
}

static auto make_features()
{
    return features_t
    {
        feature_t{"i8"}.scalar(feature_type::int8),
        feature_t{"i16"}.scalar(feature_type::int16),
        feature_t{"i32"}.scalar(feature_type::int32),
        feature_t{"i64"}.scalar(feature_type::int64),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"f64"}.scalar(feature_type::float64),

        feature_t{"ui8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)),
        feature_t{"ui16_struct"}.scalar(feature_type::uint16, make_dims(1, 1, 1)),
        feature_t{"ui32_struct"}.scalar(feature_type::uint32, make_dims(1, 2, 1)),
        feature_t{"ui64_struct"}.scalar(feature_type::uint64, make_dims(1, 1, 2)),

        feature_t{"sclass2"}.sclass(2),
        feature_t{"sclass10"}.sclass(10),

        feature_t{"mclass3"}.mclass(3),
    };
}

class fixture_dataset_t final : public memory_dataset_t
{
public:

    fixture_dataset_t(tensor_size_t samples, features_t features, size_t target) :
        m_samples(samples),
        m_features(std::move(features)),
        m_target(target)
    {
    }

    void load() override
    {
        resize(m_samples, m_features, m_target);

        // scalars
        for (tensor_size_t feature = 0; feature < 6; ++ feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; sample += feature + 1)
            {
                this->set(sample, feature, sample + feature);
            }
        }

        // structured
        for (tensor_size_t feature = 6; feature < 10; ++ feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
            {
                this->set(sample, feature,
                    make_tensor(sample % feature, m_features[static_cast<size_t>(feature)].dims()));
            }
        }

        // single label
        for (tensor_size_t sample = 0, feature = 10; sample < m_samples; sample += 2)
        {
            this->set(sample, feature, sample % 2);
        }
        for (tensor_size_t sample = 0, feature = 11; sample < m_samples; sample += 3)
        {
            this->set(sample, feature, sample % 10);
        }

        // multi label
        for (tensor_size_t sample = 0, feature = 12; sample < m_samples; sample += 4)
        {
            this->set(sample, feature, make_tensor(sample % 3, make_dims(3)));
        }
    }

    static auto mask0() { return make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    static auto mask1() { return make_tensor<uint8_t>(make_dims(4), 0xAA, 0xAA, 0xAA, 0x80); }
    static auto mask2() { return make_tensor<uint8_t>(make_dims(4), 0x92, 0x49, 0x24, 0x80); }
    static auto mask3() { return make_tensor<uint8_t>(make_dims(4), 0x88, 0x88, 0x88, 0x80); }
    static auto mask4() { return make_tensor<uint8_t>(make_dims(4), 0x84, 0x21, 0x08, 0x00); }
    static auto mask5() { return make_tensor<uint8_t>(make_dims(4), 0x82, 0x08, 0x20, 0x80); }
    static auto mask6() { return make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    static auto mask7() { return make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    static auto mask8() { return make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    static auto mask9() { return make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    static auto mask10() { return make_tensor<uint8_t>(make_dims(4), 0xAA, 0xAA, 0xAA, 0x80); }
    static auto mask11() { return make_tensor<uint8_t>(make_dims(4), 0x92, 0x49, 0x24, 0x80); }
    static auto mask12() { return make_tensor<uint8_t>(make_dims(4), 0x88, 0x88, 0x88, 0x80); }

    static auto data0() { return make_tensor<int8_t>(make_dims(25, 1, 1, 1),
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24);
    }
    static auto data1() { return make_tensor<int16_t>(make_dims(25, 1, 1, 1),
        1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21, 0, 23, 0, 25);
    }
    static auto data2() { return make_tensor<int32_t>(make_dims(25, 1, 1, 1),
        2, 0, 0, 5, 0, 0, 8, 0, 0, 11, 0, 0, 14, 0, 0, 17, 0, 0, 20, 0, 0, 23, 0, 0, 26);
    }
    static auto data3() { return make_tensor<int64_t>(make_dims(25, 1, 1, 1),
        3, 0, 0, 0, 7, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 19, 0, 0, 0, 23, 0, 0, 0, 27);
    }
    static auto data4() { return make_tensor<float>(make_dims(25, 1, 1, 1),
        4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 14, 0, 0, 0, 0, 19, 0, 0, 0, 0, 24, 0, 0, 0, 0);
    }
    static auto data5() { return make_tensor<double>(make_dims(25, 1, 1, 1),
        5, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 29);
    }
    static auto data6() { return make_tensor<uint8_t>(make_dims(25, 2, 1, 2),
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0,
        0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0,
        0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0);
    }
    static auto data7() { return make_tensor<uint16_t>(make_dims(25, 1, 1, 1),
        0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3);
    }
    static auto data8() { return make_tensor<uint32_t>(make_dims(25, 1, 2, 1),
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3, 3, 4,
        4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0);
    }
    static auto data9() { return make_tensor<uint64_t>(make_dims(25, 1, 1, 2),
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 0, 1, 1, 2, 2, 3,
        3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6);
    }
    static auto data10() { return make_tensor<uint8_t>(make_dims(25),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    static auto data11() { return make_tensor<uint8_t>(make_dims(25),
        0, 0, 0, 3, 0, 0, 6, 0, 0, 9, 0, 0, 2, 0, 0, 5, 0, 0, 8, 0, 0, 1, 0, 0, 4);
    }
    static auto data12() { return make_tensor<uint8_t>(make_dims(25, 3),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
        2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

private:

    tensor_size_t   m_samples{0};
    features_t      m_features;
    size_t          m_target;
};

static auto make_dataset(tensor_size_t samples, const features_t& features, size_t target)
{
    auto dataset = fixture_dataset_t{samples, features, target};
    UTEST_CHECK_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    return dataset;
}

UTEST_BEGIN_MODULE(test_dataset_memory)

UTEST_CASE(check_samples)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 100);
    auto dataset = make_dataset(samples.size(), features, string_t::npos);
    {
        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 0);

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 100);
        UTEST_CHECK_EQUAL(train_samples, ::nano::arange(0, 100));
    }
    {
        dataset.testing(make_range(0, 10));
        dataset.testing(make_range(20, 50));

        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 40);
        UTEST_CHECK_EQUAL(test_samples.slice(0, 10), ::nano::arange(0, 10));
        UTEST_CHECK_EQUAL(test_samples.slice(10, 40), ::nano::arange(20, 50));

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 60);
        UTEST_CHECK(train_samples.slice(0, 10) == ::nano::arange(10, 20));
        UTEST_CHECK(train_samples.slice(10, 60) == ::nano::arange(50, 100));
    }
    {
        dataset.no_testing();

        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 0);

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 100);
        UTEST_CHECK_EQUAL(train_samples, ::nano::arange(0, 100));
    }
}

UTEST_CASE(dataset_target_NA)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), features, string_t::npos);

    UTEST_CHECK_EQUAL(dataset.features(), 13);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);

    check_inputs(dataset, 0, features[0U], fixture_dataset_t::data0(), fixture_dataset_t::mask0());
    check_inputs(dataset, 1, features[1U], fixture_dataset_t::data1(), fixture_dataset_t::mask1());
    check_inputs(dataset, 2, features[2U], fixture_dataset_t::data2(), fixture_dataset_t::mask2());
    check_inputs(dataset, 3, features[3U], fixture_dataset_t::data3(), fixture_dataset_t::mask3());
    check_inputs(dataset, 4, features[4U], fixture_dataset_t::data4(), fixture_dataset_t::mask4());
    check_inputs(dataset, 5, features[5U], fixture_dataset_t::data5(), fixture_dataset_t::mask5());
    check_inputs(dataset, 6, features[6U], fixture_dataset_t::data6(), fixture_dataset_t::mask6());
    check_inputs(dataset, 7, features[7U], fixture_dataset_t::data7(), fixture_dataset_t::mask7());
    check_inputs(dataset, 8, features[8U], fixture_dataset_t::data8(), fixture_dataset_t::mask8());
    check_inputs(dataset, 9, features[9U], fixture_dataset_t::data9(), fixture_dataset_t::mask9());
    check_inputs(dataset, 10, features[10U], fixture_dataset_t::data10(), fixture_dataset_t::mask10());
    check_inputs(dataset, 11, features[11U], fixture_dataset_t::data11(), fixture_dataset_t::mask11());
    check_inputs(dataset, 12, features[12U], fixture_dataset_t::data12(), fixture_dataset_t::mask12());
}

UTEST_CASE(dataset_target_0U)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), features, 0U);

    UTEST_CHECK_EQUAL(dataset.features(), 12);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::regression);

    check_target(dataset, features[0U], fixture_dataset_t::data0(), fixture_dataset_t::mask0());
    check_inputs(dataset, 0, features[1U], fixture_dataset_t::data1(), fixture_dataset_t::mask1());
    check_inputs(dataset, 1, features[2U], fixture_dataset_t::data2(), fixture_dataset_t::mask2());
    check_inputs(dataset, 2, features[3U], fixture_dataset_t::data3(), fixture_dataset_t::mask3());
    check_inputs(dataset, 3, features[4U], fixture_dataset_t::data4(), fixture_dataset_t::mask4());
    check_inputs(dataset, 4, features[5U], fixture_dataset_t::data5(), fixture_dataset_t::mask5());
    check_inputs(dataset, 5, features[6U], fixture_dataset_t::data6(), fixture_dataset_t::mask6());
    check_inputs(dataset, 6, features[7U], fixture_dataset_t::data7(), fixture_dataset_t::mask7());
    check_inputs(dataset, 7, features[8U], fixture_dataset_t::data8(), fixture_dataset_t::mask8());
    check_inputs(dataset, 8, features[9U], fixture_dataset_t::data9(), fixture_dataset_t::mask9());
    check_inputs(dataset, 9, features[10U], fixture_dataset_t::data10(), fixture_dataset_t::mask10());
    check_inputs(dataset, 10, features[11U], fixture_dataset_t::data11(), fixture_dataset_t::mask11());
    check_inputs(dataset, 11, features[12U], fixture_dataset_t::data12(), fixture_dataset_t::mask12());
}

UTEST_CASE(dataset_target_11U)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), features, 11U);

    UTEST_CHECK_EQUAL(dataset.features(), 12);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::sclassification);

    check_inputs(dataset, 0, features[0U], fixture_dataset_t::data0(), fixture_dataset_t::mask0());
    check_inputs(dataset, 1, features[1U], fixture_dataset_t::data1(), fixture_dataset_t::mask1());
    check_inputs(dataset, 2, features[2U], fixture_dataset_t::data2(), fixture_dataset_t::mask2());
    check_inputs(dataset, 3, features[3U], fixture_dataset_t::data3(), fixture_dataset_t::mask3());
    check_inputs(dataset, 4, features[4U], fixture_dataset_t::data4(), fixture_dataset_t::mask4());
    check_inputs(dataset, 5, features[5U], fixture_dataset_t::data5(), fixture_dataset_t::mask5());
    check_inputs(dataset, 6, features[6U], fixture_dataset_t::data6(), fixture_dataset_t::mask6());
    check_inputs(dataset, 7, features[7U], fixture_dataset_t::data7(), fixture_dataset_t::mask7());
    check_inputs(dataset, 8, features[8U], fixture_dataset_t::data8(), fixture_dataset_t::mask8());
    check_inputs(dataset, 9, features[9U], fixture_dataset_t::data9(), fixture_dataset_t::mask9());
    check_inputs(dataset, 10, features[10U], fixture_dataset_t::data10(), fixture_dataset_t::mask10());
    check_target(dataset, features[11U], fixture_dataset_t::data11(), fixture_dataset_t::mask11());
    check_inputs(dataset, 11, features[12U], fixture_dataset_t::data12(), fixture_dataset_t::mask12());
}

UTEST_CASE(dataset_target_12U)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), features, 12U);

    UTEST_CHECK_EQUAL(dataset.features(), 12);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::mclassification);

    check_inputs(dataset, 0, features[0U], fixture_dataset_t::data0(), fixture_dataset_t::mask0());
    check_inputs(dataset, 1, features[1U], fixture_dataset_t::data1(), fixture_dataset_t::mask1());
    check_inputs(dataset, 2, features[2U], fixture_dataset_t::data2(), fixture_dataset_t::mask2());
    check_inputs(dataset, 3, features[3U], fixture_dataset_t::data3(), fixture_dataset_t::mask3());
    check_inputs(dataset, 4, features[4U], fixture_dataset_t::data4(), fixture_dataset_t::mask4());
    check_inputs(dataset, 5, features[5U], fixture_dataset_t::data5(), fixture_dataset_t::mask5());
    check_inputs(dataset, 6, features[6U], fixture_dataset_t::data6(), fixture_dataset_t::mask6());
    check_inputs(dataset, 7, features[7U], fixture_dataset_t::data7(), fixture_dataset_t::mask7());
    check_inputs(dataset, 8, features[8U], fixture_dataset_t::data8(), fixture_dataset_t::mask8());
    check_inputs(dataset, 9, features[9U], fixture_dataset_t::data9(), fixture_dataset_t::mask9());
    check_inputs(dataset, 10, features[10U], fixture_dataset_t::data10(), fixture_dataset_t::mask10());
    check_inputs(dataset, 11, features[11U], fixture_dataset_t::data11(), fixture_dataset_t::mask11());
    check_target(dataset, features[12U], fixture_dataset_t::data12(), fixture_dataset_t::mask12());
}

UTEST_END_MODULE()
