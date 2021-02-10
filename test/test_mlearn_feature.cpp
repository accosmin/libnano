#include <utest/utest.h>
#include <nano/mlearn/feature.h>

using namespace nano;

template <typename tvalue, size_t trank>
static auto make_tensor(tvalue value, tensor_dims_t<trank> dims)
{
    tensor_mem_t<tvalue, trank> values(dims);
    values.constant(value);
    return values;
}

template <typename tvalue>
static void check_set_scalar(feature_storage_t& storage, tensor_size_t sample, tvalue value, tensor3d_dims_t dims)
{
    // cannot set multi-label indices
    for (const auto& values_nok : {
        make_tensor<tvalue, 1>(value, make_dims(1)),
        make_tensor<tvalue, 1>(value, make_dims(10)),
    })
    {
        UTEST_REQUIRE_THROW(storage.set(sample, values_nok), std::runtime_error);
    }

    // check if possible to set with scalar
    if (::nano::size(dims) == 1)
    {
        UTEST_REQUIRE_NOTHROW(storage.set(sample, value));
    }
    else
    {
        UTEST_REQUIRE_THROW(storage.set(sample, value), std::runtime_error);
    }

    // should be possible to set with compatible tensor
    const auto values = make_tensor<tvalue, 3>(value, dims);
    UTEST_REQUIRE_NOTHROW(storage.set(sample, values));

    // cannot set with incompatible tensor
    const auto values_nok = make_tensor<tvalue, 3>(
        value,
        make_dims(
            std::get<0>(dims),
            std::get<1>(dims) + 1,
            std::get<2>(dims)));
    UTEST_REQUIRE_THROW(storage.set(sample, values_nok), std::runtime_error);

    // cannot set if the sample index is invalid
    for (const auto invalid_sample : {-1, 100})
    {
        if (::nano::size(dims) == 1)
        {
            UTEST_REQUIRE_THROW(storage.set(invalid_sample, value), std::runtime_error);
        }
        UTEST_REQUIRE_THROW(storage.set(invalid_sample, values), std::runtime_error);
    }
}

template <typename tvalue>
static void check_set_sclass(feature_storage_t& storage, tensor_size_t sample, tvalue value)
{
    // cannot set multi-label indices
    for (const auto& values_nok : {
        make_tensor<tvalue, 1>(value, make_dims(1)),
        make_tensor<tvalue, 1>(value, make_dims(10)),
    })
    {
        UTEST_REQUIRE_THROW(storage.set(sample, values_nok), std::runtime_error);
    }

    // cannot set multivariate scalars
    for (const auto& values_nok : {
        make_tensor<tvalue, 3>(value, make_dims(1, 1, 1)),
        make_tensor<tvalue, 3>(value, make_dims(2, 1, 3)),
    })
    {
        UTEST_REQUIRE_THROW(storage.set(sample, values_nok), std::runtime_error);
    }

    // cannot set with out-of-bounds class indices
    UTEST_REQUIRE_THROW(storage.set(sample, static_cast<tvalue>(-1)), std::runtime_error);
    UTEST_REQUIRE_THROW(storage.set(sample, static_cast<tvalue>(71)), std::runtime_error);

    // check if possible to set with valid class index
    UTEST_REQUIRE_NOTHROW(storage.set(sample, value));

    // cannot set if the sample index is invalid
    for (const auto invalid_sample : {-1, 100})
    {
        UTEST_REQUIRE_THROW(storage.set(invalid_sample, value), std::runtime_error);
    }
}

template <typename tvalue>
static void check_set_mclass(feature_storage_t& storage, tensor_size_t sample, tvalue hit0, tvalue hit1, tvalue hit2)
{
    // cannot set multi-label indices of invalid size
    for (const auto& values_nok : {
        make_tensor<tvalue, 1>(hit0, make_dims(1)),
        make_tensor<tvalue, 1>(hit0, make_dims(4)),
    })
    {
        UTEST_REQUIRE_THROW(storage.set(sample, values_nok), std::runtime_error);
    }

    // cannot set scalars
    UTEST_REQUIRE_THROW(storage.set(sample, hit0), std::runtime_error);

    // cannot set multivariate scalars
    for (const auto& values_nok : {
        make_tensor<tvalue, 3>(hit0, make_dims(1, 1, 1)),
        make_tensor<tvalue, 3>(hit0, make_dims(2, 1, 3)),
    })
    {
        UTEST_REQUIRE_THROW(storage.set(sample, values_nok), std::runtime_error);
    }

    // check if possible to set with valid class hits
    tensor_mem_t<tvalue, 1> values(make_dims(3), {hit0, hit1, hit2});
    UTEST_REQUIRE_NOTHROW(storage.set(sample, values));

    // cannot set if the sample index is invalid
    for (const auto invalid_sample : {-1, 100})
    {
        UTEST_REQUIRE_THROW(storage.set(invalid_sample, values), std::runtime_error);
    }
}

UTEST_BEGIN_MODULE(test_mlearn_feature)

UTEST_CASE(_default)
{
    feature_t feature;
    UTEST_CHECK_EQUAL(static_cast<bool>(feature), false);
    UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::unsupervised);

    feature = feature_t{"feature"};
    UTEST_CHECK_EQUAL(static_cast<bool>(feature), true);
    UTEST_CHECK_EQUAL(feature.dims(), make_dims(1, 1, 1));
    UTEST_CHECK_EQUAL(feature.type(), feature_type::float32);
    UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::regression);

    UTEST_CHECK(feature_t::missing(feature_t::placeholder_value()));
    UTEST_CHECK(!feature_t::missing(0));
}

UTEST_CASE(task_type)
{
    {
        auto feature = feature_t{};
        UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::unsupervised);
    }
    {
        auto feature = feature_t{"feature"}.sclass(7);
        UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::sclassification);
    }
    {
        auto feature = feature_t{"feature"}.mclass(7);
        UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::mclassification);
    }
    {
        auto feature = feature_t{"feature"};
        UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::regression);
    }
    {
        auto feature = feature_t{"feature"}.scalar();
        UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::regression);
    }
    {
        auto feature = feature_t{"feature"}.scalar(feature_type::float32, make_dims(1, 1, 2));
        UTEST_CHECK_EQUAL(feature.dims(), make_dims(1, 1, 2));
        UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::regression);
    }
    {
        auto feature = feature_t{"feature"}.scalar(feature_type::float64, make_dims(3, 2, 1));
        UTEST_CHECK_EQUAL(feature.dims(), make_dims(3, 2, 1));
        UTEST_CHECK_EQUAL(static_cast<task_type>(feature), task_type::regression);
    }
}

UTEST_CASE(discrete)
{
    auto feature = feature_t{"cate"};
    UTEST_CHECK(!feature.discrete());

    feature.sclass(4);
    UTEST_CHECK(feature.discrete());
    UTEST_CHECK_EQUAL(feature.label(0), "");
    UTEST_CHECK_EQUAL(feature.label(1), "");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");
    UTEST_CHECK_EQUAL(feature.type(), feature_type::sclass);

    UTEST_CHECK_EQUAL(feature.set_label(""), string_t::npos);
    UTEST_CHECK_EQUAL(feature.label(0), "");
    UTEST_CHECK_EQUAL(feature.label(1), "");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate0"), 0U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate1"), 1U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate1"), 1U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate2"), 2U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "cate2");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate3"), 3U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "cate2");
    UTEST_CHECK_EQUAL(feature.label(3), "cate3");

    UTEST_CHECK_EQUAL(feature.set_label("cate4"), string_t::npos);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "cate2");
    UTEST_CHECK_EQUAL(feature.label(3), "cate3");
}

UTEST_CASE(compare)
{
    const auto make_feature_cont = [] (const string_t& name, feature_type type = feature_type::float32)
    {
        auto feature = feature_t{name}.scalar(type);
        UTEST_CHECK(!feature.discrete());
        UTEST_CHECK(!feature.optional());
        UTEST_CHECK_EQUAL(feature.type(), type);
        UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
        UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);
        return feature;
    };

    const auto make_feature_cont_opt = [] (const string_t& name, feature_type type = feature_type::float32)
    {
        auto feature = feature_t{name}.scalar(type).optional(true);
        UTEST_CHECK(!feature.discrete());
        UTEST_CHECK(feature.optional());
        UTEST_CHECK_EQUAL(feature.type(), type);
        UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
        UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);
        return feature;
    };

    const auto make_feature_cate = [] (const string_t& name, feature_type type = feature_type::sclass)
    {
        assert(type == feature_type::sclass || type == feature_type::mclass);
        auto feature = feature_t{name};
        switch (type)
        {
        case feature_type::sclass:  feature.sclass(strings_t{"cate0", "cate1", "cate2"}); break;
        default:                    feature.mclass(strings_t{"cate0", "cate1", "cate2"}); break;
        }
        UTEST_CHECK(feature.discrete());
        UTEST_CHECK(!feature.optional());
        UTEST_CHECK_EQUAL(feature.type(), type);
        UTEST_CHECK_EQUAL(feature.label(0), "cate0");
        UTEST_CHECK_EQUAL(feature.label(1), "cate1");
        UTEST_CHECK_EQUAL(feature.label(2), "cate2");
        UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
        UTEST_CHECK_THROW(feature.label(+3), std::out_of_range);
        UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());
        return feature;
    };

    const auto make_feature_cate_opt = [] (const string_t& name, feature_type type = feature_type::sclass)
    {
        assert(type == feature_type::sclass || type == feature_type::mclass);
        auto feature = feature_t{name}.optional(true);
        switch (type)
        {
        case feature_type::sclass:  feature.sclass(strings_t{"cate_opt0", "cate_opt1"}); break;
        default:                    feature.mclass(strings_t{"cate_opt0", "cate_opt1"}); break;
        }
        UTEST_CHECK(feature.discrete());
        UTEST_CHECK(feature.optional());
        UTEST_CHECK_EQUAL(feature.type(), type);
        UTEST_CHECK_EQUAL(feature.label(0), "cate_opt0");
        UTEST_CHECK_EQUAL(feature.label(1), "cate_opt1");
        UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
        UTEST_CHECK_THROW(feature.label(+2), std::out_of_range);
        UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());
        return feature;
    };

    const auto to_string = [] (const feature_t& feature)
    {
        std::stringstream stream;
        stream << feature;
        return stream.str();
    };

    UTEST_CHECK_EQUAL(make_feature_cont("f"), make_feature_cont("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont("gf"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont("f", feature_type::float64));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont("f")), "name=f,type=float32,labels[],mandatory");

    UTEST_CHECK_EQUAL(make_feature_cont_opt("f"), make_feature_cont_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont_opt("f"), make_feature_cont_opt("ff"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont_opt("f")), "name=f,type=float32,labels[],optional");

    UTEST_CHECK_EQUAL(make_feature_cate("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate("f"), make_feature_cate("x"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate("f")), "name=f,type=sclass,labels[cate0,cate1,cate2],mandatory");

    UTEST_CHECK_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("x"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("f", feature_type::mclass));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate_opt("f")), "name=f,type=sclass,labels[cate_opt0,cate_opt1],optional");

    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cate_opt("f"));
}

UTEST_CASE(feature_info)
{
    {
        const auto info = feature_info_t{};
        UTEST_CHECK_CLOSE(info.importance(), 0.0, 1e-12);
    }
    {
        const auto info = feature_info_t{7, 13, 42.0};
        UTEST_CHECK_EQUAL(info.feature(), 7);
        UTEST_CHECK_EQUAL(info.count(), 13);
        UTEST_CHECK_CLOSE(info.importance(), 42.0, 1e-12);
    }
    {
        auto infos = feature_infos_t{
            feature_info_t{5, 1, 45.0},
            feature_info_t{6, 2, 36.0},
            feature_info_t{4, 7, 41.0}
        };

        feature_info_t::sort_by_index(infos);
        UTEST_REQUIRE_EQUAL(infos.size(), 3U);
        UTEST_CHECK_EQUAL(infos[0].feature(), 4);
        UTEST_CHECK_EQUAL(infos[1].feature(), 5);
        UTEST_CHECK_EQUAL(infos[2].feature(), 6);

        feature_info_t::sort_by_importance(infos);
        UTEST_REQUIRE_EQUAL(infos.size(), 3U);
        UTEST_CHECK_EQUAL(infos[0].feature(), 5);
        UTEST_CHECK_EQUAL(infos[1].feature(), 4);
        UTEST_CHECK_EQUAL(infos[2].feature(), 6);
    }
}

UTEST_CASE(feature_storage_missing)
{
    UTEST_CHECK(feature_storage_t::missing(tensor_size_t(-1)));
    UTEST_CHECK(!feature_storage_t::missing(tensor_size_t(+0)));
    UTEST_CHECK(!feature_storage_t::missing(tensor_size_t(+1)));
    UTEST_CHECK(!feature_storage_t::missing(tensor_size_t(+123)));

    UTEST_CHECK(feature_storage_t::missing(std::numeric_limits<scalar_t>::infinity()));
    UTEST_CHECK(feature_storage_t::missing(std::numeric_limits<scalar_t>::quiet_NaN()));

    UTEST_CHECK(!feature_storage_t::missing(-1.0));
    UTEST_CHECK(!feature_storage_t::missing(+0.0));
    UTEST_CHECK(!feature_storage_t::missing(+1.0));
    UTEST_CHECK(!feature_storage_t::missing(+123.0));
}

UTEST_CASE(feature_storage)
{
    const auto storage = feature_storage_t{};
    UTEST_CHECK_EQUAL(storage.samples(), 0);
    UTEST_CHECK_EQUAL(storage.optional(), false);
}

UTEST_CASE(feature_storage_optional)
{
    for (tensor_size_t samples = 1; samples < 32; ++ samples)
    {
        auto storage = feature_storage_t{feature_t{"feature"}.scalar(feature_type::int32), samples};

        UTEST_CHECK_EQUAL(storage.optional(), true);
        for (tensor_size_t sample = 0; sample + 2 < samples; sample += 2)
        {
            UTEST_CHECK_NOTHROW(storage.set(sample, 42));
        }
        UTEST_CHECK_EQUAL(storage.optional(), true);
        for (tensor_size_t sample = 0; sample < samples; ++ sample)
        {
            UTEST_CHECK_NOTHROW(storage.set(sample, 17));
        }
        UTEST_CHECK_EQUAL(storage.optional(), false);
    }
}

UTEST_CASE(feature_storage_scalar)
{
    for (const auto type :
    {
        feature_type::int8,
        feature_type::int16,
        feature_type::int32,
        feature_type::int64,
        feature_type::uint8,
        feature_type::uint16,
        feature_type::uint32,
        feature_type::uint64,
        feature_type::float32,
        feature_type::float64,
    })
    {
        for (const auto dims : {make_dims(1, 1, 1), make_dims(4, 2, 3)})
        {
            const auto is_scalar = ::nano::size(dims) == 1;

            feature_t feature{"scalar"};
            feature.scalar(type, dims);

            feature_storage_t storage(feature, 17);
            UTEST_CHECK_EQUAL(storage.samples(), 17);
            UTEST_CHECK_EQUAL(storage.feature(), feature);
            UTEST_CHECK_EQUAL(storage.optional(), true);

            check_set_scalar<int8_t>(storage, 0, 11, dims);
            check_set_scalar<int16_t>(storage, 1, 12, dims);
            check_set_scalar<int32_t>(storage, 2, 13, dims);
            check_set_scalar<int64_t>(storage, 5, 14, dims);
            check_set_scalar<uint8_t>(storage, 7, 21, dims);
            check_set_scalar<uint16_t>(storage, 9, 22, dims);
            check_set_scalar<uint32_t>(storage, 10, 23, dims);
            check_set_scalar<uint64_t>(storage, 11, 24, dims);
            check_set_scalar<float>(storage, 12, 32.0f, dims);
            check_set_scalar<double>(storage, 14, 42.0, dims);
            UTEST_CHECK_EQUAL(storage.optional(), true);

            if (is_scalar)
            {
                UTEST_REQUIRE_NOTHROW(storage.set(16, "57"));
            }
            else
            {
                UTEST_REQUIRE_THROW(storage.set(16, "57"), std::runtime_error);
            }
            UTEST_REQUIRE_THROW(storage.set(-1, "57"), std::runtime_error);
            UTEST_REQUIRE_THROW(storage.set(17, "57"), std::runtime_error);
            UTEST_REQUIRE_THROW(storage.set(16, "WHAT"), std::runtime_error);
            UTEST_CHECK_EQUAL(storage.optional(), true);

            tensor_mem_t<scalar_t, 4> scalars;
            tensor_mem_t<tensor_size_t, 1> slabels;
            tensor_mem_t<tensor_size_t, 2> mlabels;

            const auto samples = ::nano::arange(0, 17);
            UTEST_REQUIRE_NOTHROW(storage.get(samples, scalars));
            UTEST_REQUIRE_THROW(storage.get(samples, slabels), std::runtime_error);
            UTEST_REQUIRE_THROW(storage.get(samples, mlabels), std::runtime_error);

            UTEST_REQUIRE_EQUAL(scalars.dims(), cat_dims(17, dims));
            UTEST_CHECK_CLOSE(scalars(0, 0, 0, 0), 11.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(1, 0, 0, 0), 12.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(2, 0, 0, 0), 13.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(5, 0, 0, 0), 14.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(7, 0, 0, 0), 21.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(9, 0, 0, 0), 22.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(10, 0, 0, 0), 23.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(11, 0, 0, 0), 24.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(12, 0, 0, 0), 32.0, 1e-12);
            UTEST_CHECK_CLOSE(scalars(14, 0, 0, 0), 42.0, 1e-12);
            if (is_scalar)
            {
                UTEST_CHECK_CLOSE(scalars(16, 0, 0, 0), 57.0, 1e-12);
            }

            UTEST_CHECK(feature_storage_t::missing(scalars(3, 0, 0, 0)));
            UTEST_CHECK(feature_storage_t::missing(scalars(4, 0, 0, 0)));
            UTEST_CHECK(feature_storage_t::missing(scalars(6, 0, 0, 0)));
            UTEST_CHECK(feature_storage_t::missing(scalars(8, 0, 0, 0)));
            UTEST_CHECK(feature_storage_t::missing(scalars(13, 0, 0, 0)));
            UTEST_CHECK(feature_storage_t::missing(scalars(15, 0, 0, 0)));
            if (!is_scalar)
            {
                UTEST_CHECK(feature_storage_t::missing(scalars(16, 0, 0, 0)));
            }

            feature_scalar_stats_t stats;
            UTEST_REQUIRE_NOTHROW(stats = storage.scalar_stats(samples));
            UTEST_CHECK_EQUAL(stats.m_min.dims(), dims);
            UTEST_CHECK_EQUAL(stats.m_max.dims(), dims);
            UTEST_CHECK_EQUAL(stats.m_mean.dims(), dims);
            UTEST_CHECK_EQUAL(stats.m_stdev.dims(), dims);

            UTEST_CHECK_CLOSE(stats.m_min.min(), 11.0, 1e-12);
            UTEST_CHECK_CLOSE(stats.m_min.max(), 11.0, 1e-12);

            UTEST_CHECK_CLOSE(stats.m_max.min(), is_scalar ? 57.0 : 42.0, 1e-12);
            UTEST_CHECK_CLOSE(stats.m_max.max(), is_scalar ? 57.0 : 42.0, 1e-12);

            UTEST_CHECK_CLOSE(stats.m_mean.min(), is_scalar ? (271.0/11.0) : 21.4, 1e-12);
            UTEST_CHECK_CLOSE(stats.m_mean.max(), is_scalar ? (271.0/11.0) : 21.4, 1e-12);

            UTEST_CHECK_CLOSE(stats.m_stdev.min(), is_scalar ? 14.214589176425 : 9.822875795249, 1e-12);
            UTEST_CHECK_CLOSE(stats.m_stdev.max(), is_scalar ? 14.214589176425 : 9.822875795249, 1e-12);

            UTEST_CHECK_THROW(storage.sclass_stats(samples), std::runtime_error);
            UTEST_CHECK_THROW(storage.mclass_stats(samples), std::runtime_error);
        }
    }
}

UTEST_CASE(feature_storage_sclass)
{
    feature_t feature("sclass");
    feature.sclass(4);

    feature_storage_t storage(feature, 17);
    UTEST_CHECK_EQUAL(storage.samples(), 17);
    UTEST_CHECK_EQUAL(storage.feature(), feature);
    UTEST_CHECK_EQUAL(storage.optional(), true);

    check_set_sclass<int8_t>(storage, 0, 0);
    check_set_sclass<int16_t>(storage, 1, 1);
    check_set_sclass<int32_t>(storage, 2, 2);
    check_set_sclass<int64_t>(storage, 5, 3);
    check_set_sclass<uint8_t>(storage, 7, 2);
    check_set_sclass<uint16_t>(storage, 9, 1);
    check_set_sclass<uint32_t>(storage, 10, 0);
    check_set_sclass<uint64_t>(storage, 11, 1);
    check_set_sclass<float>(storage, 12, 2);
    check_set_sclass<double>(storage, 14, 3);
    UTEST_CHECK_EQUAL(storage.optional(), true);

    UTEST_REQUIRE_NOTHROW(storage.set(16, "3"));
    UTEST_REQUIRE_THROW(storage.set(16, "4"), std::runtime_error);
    UTEST_REQUIRE_THROW(storage.set(17, "3"), std::runtime_error);
    UTEST_REQUIRE_THROW(storage.set(16, "WHAT"), std::runtime_error);
    UTEST_CHECK_EQUAL(storage.optional(), true);

    tensor_mem_t<scalar_t, 4> scalars;
    tensor_mem_t<tensor_size_t, 1> slabels;
    tensor_mem_t<tensor_size_t, 2> mlabels;

    const auto samples = ::nano::arange(0, 17);
    UTEST_REQUIRE_NOTHROW(storage.get(samples, slabels));
    UTEST_REQUIRE_THROW(storage.get(samples, scalars), std::runtime_error);
    UTEST_REQUIRE_THROW(storage.get(samples, mlabels), std::runtime_error);

    const auto expected_slabels = tensor_mem_t<int, 1>{
        make_dims(17),
        {0, 1, 2, -1, -1, 3, -1, 2, -1, 1, 0, 1, 2, -1, 3, -1, 3}
    };
    UTEST_CHECK_EQUAL(slabels.dims(), make_dims(17));
    UTEST_CHECK_EIGEN_CLOSE(slabels.vector().cast<int>(), expected_slabels.vector(), 1e-12);

    feature_sclass_stats_t stats;
    UTEST_REQUIRE_NOTHROW(stats = storage.sclass_stats(samples));
    UTEST_REQUIRE_EQUAL(stats.m_class_counts.dims(), make_dims(4));
    UTEST_CHECK_EQUAL(stats.m_class_counts(0), 2);
    UTEST_CHECK_EQUAL(stats.m_class_counts(1), 3);
    UTEST_CHECK_EQUAL(stats.m_class_counts(2), 3);
    UTEST_CHECK_EQUAL(stats.m_class_counts(3), 3);

    UTEST_CHECK_THROW(storage.scalar_stats(samples), std::runtime_error);
    UTEST_CHECK_THROW(storage.mclass_stats(samples), std::runtime_error);
}

UTEST_CASE(feature_storage_mclass)
{
    feature_t feature("mclass");
    feature.mclass(3);

    feature_storage_t storage(feature, 17);
    UTEST_CHECK_EQUAL(storage.samples(), 17);
    UTEST_CHECK_EQUAL(storage.feature(), feature);
    UTEST_CHECK_EQUAL(storage.optional(), true);

    check_set_mclass<int8_t>(storage, 0, 0, 0, 0);
    check_set_mclass<int16_t>(storage, 1, 0, 0, 1);
    check_set_mclass<int32_t>(storage, 2, 1, 1, 0);
    check_set_mclass<int64_t>(storage, 5, 1, 1, 1);
    check_set_mclass<uint8_t>(storage, 7, 0, 1, 0);
    check_set_mclass<uint16_t>(storage, 9, 1, 0, 0);
    check_set_mclass<uint32_t>(storage, 10, 0, 0, 0);
    check_set_mclass<uint64_t>(storage, 11, 1, 1, 1);
    check_set_mclass<float>(storage, 12, 1, 0, 0);
    check_set_mclass<double>(storage, 14, 0, 1, 1);
    UTEST_CHECK_EQUAL(storage.optional(), true);

    UTEST_REQUIRE_THROW(storage.set(16, "4"), std::runtime_error);
    UTEST_REQUIRE_THROW(storage.set(17, "3"), std::runtime_error);
    UTEST_REQUIRE_THROW(storage.set(16, "WHAT"), std::runtime_error);
    UTEST_CHECK_EQUAL(storage.optional(), true);

    tensor_mem_t<scalar_t, 4> scalars;
    tensor_mem_t<tensor_size_t, 1> slabels;
    tensor_mem_t<tensor_size_t, 2> mlabels;

    const auto samples = ::nano::arange(0, 17);
    UTEST_REQUIRE_NOTHROW(storage.get(samples, mlabels));
    UTEST_REQUIRE_THROW(storage.get(samples, scalars), std::runtime_error);
    UTEST_REQUIRE_THROW(storage.get(samples, slabels), std::runtime_error);

    const auto expected_mlabels = tensor_mem_t<int, 2>{
        make_dims(17, 3),
        {
            +0, +0, +0,
            +0, +0, +1,
            +1, +1, +0,
            -1, -1, -1,
            -1, -1, -1,
            +1, +1, +1,
            -1, -1, -1,
            +0, +1, +0,
            -1, -1, -1,
            +1, +0, +0,
            +0, +0, +0,
            +1, +1, +1,
            +1, +0, +0,
            -1, -1, -1,
            +0, +1, +1,
            -1, -1, -1,
            -1, -1, -1
        }
    };
    UTEST_CHECK_EQUAL(mlabels.dims(), make_dims(17, 3));
    UTEST_CHECK_EIGEN_CLOSE(mlabels.vector().cast<int>(), expected_mlabels.vector(), 1e-12);

    feature_mclass_stats_t stats;
    UTEST_REQUIRE_NOTHROW(stats = storage.mclass_stats(samples));
    UTEST_REQUIRE_EQUAL(stats.m_class_counts.dims(), make_dims(3));
    UTEST_CHECK_EQUAL(stats.m_class_counts(0), 5);
    UTEST_CHECK_EQUAL(stats.m_class_counts(1), 5);
    UTEST_CHECK_EQUAL(stats.m_class_counts(2), 4);

    UTEST_CHECK_THROW(storage.scalar_stats(samples), std::runtime_error);
    UTEST_CHECK_THROW(storage.sclass_stats(samples), std::runtime_error);
}

UTEST_END_MODULE()
