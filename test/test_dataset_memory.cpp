#include <utest/utest.h>
#include <nano/dataset/memory.h>

using namespace nano;

// TODO: create in-memory dataset with various feature types (sclass, mclass, scalar or structured) w/o optional
// TODO: check that the flatten & the feature iterators work as expected
// TODO: check that feature normalization works
// TODO: check that feature extraction works (e.g sign(x), sign(x)*log(1+x^2), polynomial expansion)

template <typename tvalue, size_t trank>
static auto make_tensor(tvalue value, tensor_dims_t<trank> dims)
{
    tensor_mem_t<tvalue, trank> values(dims);
    values.constant(value);
    return values;
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

        feature_t{"ui8_struct"}.scalar(feature_type::uint8, make_dims(3, 8, 8)),
        feature_t{"ui16_struct"}.scalar(feature_type::uint16, make_dims(3, 8, 8)),
        feature_t{"ui32_struct"}.scalar(feature_type::uint32, make_dims(3, 8, 8)),
        feature_t{"ui64_struct"}.scalar(feature_type::uint64, make_dims(3, 8, 8)),

        feature_t{"sclass2"}.sclass(2),
        feature_t{"sclass10"}.sclass(10),

        feature_t{"mclass10"}.mclass(10),
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

        for (tensor_size_t feature = 0; feature < 6; ++ feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; sample += feature + 1)
            {
                check_set_scalar(sample, feature, sample + feature, make_dims(1, 1, 1));
            }
        }

        for (tensor_size_t feature = 6; feature < 10; ++ feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; sample += feature + 1)
            {
                check_set_scalar(sample, feature, sample % feature, make_dims(3, 8, 8));
            }
        }

        for (tensor_size_t sample = 0, feature = 10; sample < m_samples; sample += 2)
        {
            check_set_sclass(sample, feature, sample % 2, 2);
        }
        for (tensor_size_t sample = 0, feature = 11; sample < m_samples; sample += 3)
        {
            check_set_sclass(sample, feature, sample % 10, 10);
        }

        for (tensor_size_t sample = 0, feature = 12; sample < m_samples; sample += 4)
        {
            check_set_mclass(sample, feature, sample % 10, 10);
        }
    }

private:

    template <typename tvalue>
    void check_set(tensor_size_t sample, tensor_size_t feature, const tvalue& value)
    {
        const auto target = static_cast<tensor_size_t>(m_target);
        if (target < 0)
        {
            set(sample, feature, value);
        }
        else if (feature == target)
        {
            set(sample, value);
        }
        else if (feature < target)
        {
            set(sample, feature, value);
        }
        else
        {
            set(sample, feature - 1, value);
        }
    }

    template <typename tvalue>
    void check_set_scalar(tensor_size_t sample, tensor_size_t feature, tvalue value, tensor3d_dims_t dims)
    {
        // check if possible to set with scalar
        if (::nano::size(dims) == 1)
        {
            UTEST_REQUIRE_NOTHROW(check_set(sample, feature, value));
            UTEST_REQUIRE_NOTHROW(check_set(sample, feature, std::to_string(value)));
        }
        else
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, value), std::runtime_error);
        }

        // should be possible to set with compatible tensor
        const auto values = make_tensor<tvalue, 3>(value, dims);
        UTEST_REQUIRE_NOTHROW(check_set(sample, feature, values));

        const auto values1d = make_tensor<tvalue, 1>(value, make_dims(::nano::size(dims)));
        UTEST_REQUIRE_NOTHROW(check_set(sample, feature, values1d));

        // cannot set with incompatible tensor
        {
            const auto values_nok = make_tensor<tvalue, 1>(value, make_dims(::nano::size(dims) + 1));
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }
        {
            const auto [dim0, dim1, dim2] = dims;
            const auto values_nok = make_tensor<tvalue, 3>(value, make_dims(dim0, dim1 + 1, dim2));
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // cannot set with invalid string
        UTEST_REQUIRE_THROW(check_set(sample, feature, "N/A"), std::runtime_error);

        // cannot set if the sample index is invalid
        for (const auto invalid_sample : {-1, 10001})
        {
            if (::nano::size(dims) == 1)
            {
                UTEST_REQUIRE_THROW(check_set(invalid_sample, feature, value), std::runtime_error);
            }
            UTEST_REQUIRE_THROW(check_set(invalid_sample, feature, values), std::runtime_error);
        }
    }

    template <typename tvalue>
    void check_set_sclass(tensor_size_t sample, tensor_size_t feature, tvalue value, tensor_size_t classes)
    {
        // cannot set multi-label indices
        for (const auto& values_nok : {
            make_tensor<tvalue, 1>(value, make_dims(1)),
            make_tensor<tvalue, 1>(value, make_dims(classes)),
        })
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // cannot set multivariate scalars
        for (const auto& values_nok : {
            make_tensor<tvalue, 3>(value, make_dims(1, 1, 1)),
            make_tensor<tvalue, 3>(value, make_dims(2, 1, 3)),
        })
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // cannot set with out-of-bounds class indices
        UTEST_REQUIRE_THROW(check_set(sample, feature, static_cast<tvalue>(-1)), std::runtime_error);
        UTEST_REQUIRE_THROW(check_set(sample, feature, static_cast<tvalue>(classes)), std::runtime_error);
        UTEST_REQUIRE_THROW(check_set(sample, feature, static_cast<tvalue>(classes + 1)), std::runtime_error);

        // check if possible to set with valid class index
        UTEST_REQUIRE_NOTHROW(check_set(sample, feature, value));

        // cannot set if the sample index is invalid
        for (const auto invalid_sample : {-1, 10001})
        {
            UTEST_REQUIRE_THROW(check_set(invalid_sample, feature, value), std::runtime_error);
        }
    }

    template <typename tvalue>
    void check_set_mclass(tensor_size_t sample, tensor_size_t feature, tvalue value, tensor_size_t classes)
    {
        // cannot set multi-label indices of invalid size
        for (const auto& values_nok : {
            make_tensor<tvalue, 1>(0, make_dims(classes + 1)),
            make_tensor<tvalue, 1>(0, make_dims(classes + 2)),
        })
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // cannot set scalars
        UTEST_REQUIRE_THROW(check_set(sample, feature, value), std::runtime_error);

        // cannot set multivariate scalars
        for (const auto& values_nok : {
            make_tensor<tvalue, 3>(value, make_dims(1, 1, 1)),
            make_tensor<tvalue, 3>(value, make_dims(2, 1, 3)),
        })
        {
            UTEST_REQUIRE_THROW(check_set(sample, feature, values_nok), std::runtime_error);
        }

        // check if possible to set with valid class hits
        tensor_mem_t<tvalue, 1> values{classes};
        values.zero();
        values(0) = 1;
        values(static_cast<tensor_size_t>(value)) = 1;
        UTEST_REQUIRE_NOTHROW(check_set(sample, feature, values));

        // cannot set if the sample index is invalid
        for (const auto invalid_sample : {-1, 10001})
        {
            UTEST_REQUIRE_THROW(check_set(invalid_sample, feature, values), std::runtime_error);
        }
    }

    tensor_size_t   m_samples{0};
    features_t      m_features;
    size_t          m_target;
};

UTEST_BEGIN_MODULE(test_dataset_memory)

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

UTEST_CASE(resize)
{
    const auto features = make_features();
    for (const auto target : {size_t{0U}, size_t{6U}, size_t{10U}, size_t{12U}, string_t::npos})
    {
        auto dataset = fixture_dataset_t{42, features, target};
        UTEST_CHECK_NOTHROW(dataset.load());
        UTEST_CHECK_EQUAL(dataset.samples(), 42);
    }
}

UTEST_END_MODULE()
