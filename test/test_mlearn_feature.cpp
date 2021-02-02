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
static void check_set_scalar(feature_storage_t& storage, tensor_size_t sample, tvalue value)
{
    UTEST_REQUIRE_NOTHROW(storage.set(sample, value));
    /*
    UTEST_REQUIRE_NOTHROW(storage.set(sample, make_tensor<tvalue, 3>(value, make_dims(1, 1, 1))));
    UTEST_REQUIRE_NOTHROW(storage.set(samples, make_tensor<tvalue, 4>(value, make_dims(1, 1, 1, 1))));

    UTEST_REQUIRE_THROW(storage.set(sample, make_tensor<tvalue, 1>(value, make_dims(1)), std::runtime_error));
    UTEST_REQUIRE_THROW(storage.set(sample, make_tensor<tvalue, 2>(value, make_dims(1)), std::runtime_error));
    UTEST_REQUIRE_THROW(storage.set(sample, make_tensor<tvalue, 3>(value, make_dims(2, 1, 1)), std::runtime_error));

    UTEST_REQUIRE_THROW(storage.set(-1, value), std::runtime_error);
    */

    /*
    const auto samples = make_tensor<tensor_size_t, 1>(sample);


    // can set multiple samples at once
    {
    }

    // cannot set multi-label values
    {
        tensor_mem_t<tvalue, 1> values(3);
        values.zero();
        UTEST_REQUIRE_THROW(storage.set(sample, values), std::runtime_error);
    }

    // cannot set high-dimensional scalar values
    {
        tensor_mem_t<tvalue, 3> values(1, 1, 1);
        values.constant(value);
        UTEST_REQUIRE_NOTHROW(storage.set(sample, values));

        values.resize(3, 1, 1);
        values.constant(value);
        UTEST_REQUIRE_THROW(storage.set(sample, values), std::runtime_error);
    }*/
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

UTEST_CASE(feature_storage_optional)
{
    {
        const auto storage = feature_storage_t{};
        UTEST_CHECK_EQUAL(storage.samples(), 0);
        UTEST_CHECK_EQUAL(storage.optional(), false);
    }
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
    feature_t feature{"feature"};

    // TODO: make it work for all scalar feature types
    // TODO: should fail with tensor_cmap_t of incompatible dimension and rank
    // TODO: should fail with scalar if dims() > 1, 1, 1

    feature.scalar(feature_type::int16); //make_dims(1, 3, 2));

    feature_storage_t storage(feature, 17);
    UTEST_CHECK_EQUAL(storage.samples(), 17);
    UTEST_CHECK_EQUAL(storage.feature(), feature);

    check_set_scalar<int8_t>(storage, 0, 11);
    check_set_scalar<int16_t>(storage, 1, 12);
    check_set_scalar<int32_t>(storage, 2, 13);
    check_set_scalar<int64_t>(storage, 5, 14);
    check_set_scalar<uint8_t>(storage, 7, 21);
    check_set_scalar<uint16_t>(storage, 9, 22);
    check_set_scalar<uint32_t>(storage, 10, 23);
    check_set_scalar<uint64_t>(storage, 11, 24);
    check_set_scalar<float>(storage, 12, 32.0f);
    check_set_scalar<double>(storage, 14, 42.0);
    UTEST_REQUIRE_NOTHROW(storage.set(16, std::to_string(57)));

    const auto samples = ::nano::arange(0, 17);
    tensor_mem_t<scalar_t, 4> values;
    UTEST_REQUIRE_NOTHROW(storage.get(samples, values));
    UTEST_REQUIRE_EQUAL(values.dims(), make_dims(17, 1, 1, 1));

    UTEST_CHECK_CLOSE(values(0), 11.0, 1e-12);
    UTEST_CHECK_CLOSE(values(1), 12.0, 1e-12);
    UTEST_CHECK_CLOSE(values(2), 13.0, 1e-12);
    UTEST_CHECK_CLOSE(values(5), 14.0, 1e-12);
    UTEST_CHECK_CLOSE(values(7), 21.0, 1e-12);
    UTEST_CHECK_CLOSE(values(9), 22.0, 1e-12);
    UTEST_CHECK_CLOSE(values(10), 23.0, 1e-12);
    UTEST_CHECK_CLOSE(values(11), 24.0, 1e-12);
    UTEST_CHECK_CLOSE(values(12), 32.0, 1e-12);
    UTEST_CHECK_CLOSE(values(14), 42.0, 1e-12);
    UTEST_CHECK_CLOSE(values(16), 57.0, 1e-12);

    UTEST_CHECK(feature_storage_t::missing(values(3)));
    UTEST_CHECK(feature_storage_t::missing(values(4)));
    UTEST_CHECK(feature_storage_t::missing(values(6)));
    UTEST_CHECK(feature_storage_t::missing(values(8)));
    UTEST_CHECK(feature_storage_t::missing(values(13)));
    UTEST_CHECK(feature_storage_t::missing(values(15)));
    // TODO:
    //
}

UTEST_END_MODULE()
