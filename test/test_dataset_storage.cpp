#include <utest/utest.h>
#include <nano/dataset/storage.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_storage)

UTEST_CASE(scalar)
{
    for (const auto dims : {make_dims(3, 1, 2), make_dims(1, 1, 1)})
    {
        const auto feature = feature_t{"feature"}.scalar(feature_type::float32, dims);

        const auto storage = feature_storage_t{feature};
        UTEST_CHECK_EQUAL(storage.dims(), dims);
        UTEST_CHECK_EQUAL(storage.classes(), 0);
        UTEST_CHECK_EQUAL(storage.name(), "feature");
        UTEST_CHECK_EQUAL(storage.feature(), feature);

        auto values = make_full_tensor<scalar_t>(cat_dims(42, dims), std::numeric_limits<scalar_t>::quiet_NaN());

        for (tensor_size_t sample : {0, 11})
        {
            const auto value = 14.6F;
            const auto expected_value = make_full_tensor<scalar_t>(dims, value);

            // check if possible to set with scalar
            if (::nano::size(dims) == 1)
            {
                UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, value));
                UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, std::to_string(value)));
            }
            else
            {
                UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, value), std::runtime_error);
            }

            // should be possible to set with compatible tensor
            const auto values3d = make_full_tensor<float>(dims, value);
            UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, values3d));

            const auto values1d = make_full_tensor<float>(make_dims(::nano::size(dims)), value);
            UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, values1d));

            // cannot set with incompatible tensor
            {
                const auto values_nok = make_full_tensor<float>(make_dims(::nano::size(dims) + 1), value);
                UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
            }
            {
                const auto [dim0, dim1, dim2] = dims;
                const auto values_nok = make_full_tensor<scalar_t>(make_dims(dim0, dim1 + 1, dim2), value);
                UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
            }

            // cannot set with invalid string
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, "N/A"), std::runtime_error);

            // the expected feature value should be there
            UTEST_CHECK_TENSOR_CLOSE(values.tensor(sample), expected_value, 1e-12);
        }
    }
}

UTEST_CASE(sclass)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto storage = feature_storage_t{feature};
    UTEST_CHECK_EQUAL(storage.classes(), 3);
    UTEST_CHECK_EQUAL(storage.name(), "feature");
    UTEST_CHECK_EQUAL(storage.feature(), feature);

    tensor_mem_t<uint8_t, 1> values(42);
    values.zero();
    for (tensor_size_t sample : {2, 7})
    {
        const auto value = feature.classes() - 1;
        const auto expected_value = value;

        // cannot set multi-label indices
        for (const auto& values_nok : {
            make_full_tensor<uint8_t>(make_dims(1), value),
            make_full_tensor<uint8_t>(make_dims(feature.classes()), value),
        })
        {
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
        }

        // cannot set multivariate scalars
        for (const auto& values_nok : {
            make_full_tensor<int>(make_dims(1, 1, 1), value),
            make_full_tensor<int>(make_dims(2, 1, 3), value),
        })
        {
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
        }

        // cannot set with out-of-bounds class indices
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, -1), std::runtime_error);
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, feature.classes()), std::runtime_error);
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, feature.classes() + 1), std::runtime_error);

        // check if possible to set with valid class index
        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, value));

        // the expected feature value should be there
        UTEST_CHECK_EQUAL(values(sample), expected_value);
    }
}

UTEST_CASE(mclass)
{
    const auto feature = feature_t{"feature"}.sclass(3);

    const auto storage = feature_storage_t{feature};
    UTEST_CHECK_EQUAL(storage.classes(), 3);
    UTEST_CHECK_EQUAL(storage.name(), "feature");
    UTEST_CHECK_EQUAL(storage.feature(), feature);

    tensor_mem_t<uint8_t, 2> values(42, feature.classes());
    values.zero();
    for (tensor_size_t sample : {11, 17})
    {
        const auto value = make_tensor<uint16_t>(make_dims(feature.classes()), 1, 0, 1);
        const auto expected_value = make_tensor<uint8_t>(make_dims(feature.classes()), 1, 0, 1);

        // cannot set multi-label indices of invalid size
        for (const auto& values_nok : {
            make_full_tensor<uint8_t>(make_dims(feature.classes() - 1), 0),
            make_full_tensor<uint8_t>(make_dims(feature.classes() + 1), 0),
        })
        {
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
        }

        // cannot set scalars or strings
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, 1), std::runtime_error);
        UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, "2"), std::runtime_error);

        // cannot set multivariate scalars
        for (const auto& values_nok : {
            make_full_tensor<uint8_t>(make_dims(1, 1, 1), 1),
            make_full_tensor<uint8_t>(make_dims(2, 1, 3), 1),
        })
        {
            UTEST_REQUIRE_THROW(storage.set(values.tensor(), sample, values_nok), std::runtime_error);
        }

        // check if possible to set with valid class hits
        UTEST_REQUIRE_NOTHROW(storage.set(values.tensor(), sample, value));

        // the expected feature value should be there
        UTEST_CHECK_TENSOR_EQUAL(values.tensor(sample), expected_value);
    }
}

UTEST_END_MODULE()
