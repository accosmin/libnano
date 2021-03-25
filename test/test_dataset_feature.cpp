#include <utest/utest.h>
#include <nano/dataset/feature.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_feature)

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
}

UTEST_CASE(missing)
{
    UTEST_CHECK(feature_t::missing(tensor_size_t(-1)));
    UTEST_CHECK(!feature_t::missing(tensor_size_t(+0)));
    UTEST_CHECK(!feature_t::missing(tensor_size_t(+1)));
    UTEST_CHECK(!feature_t::missing(tensor_size_t(+123)));

    UTEST_CHECK(feature_t::missing(feature_t::placeholder_value()));
    UTEST_CHECK(feature_t::missing(std::numeric_limits<scalar_t>::infinity()));
    UTEST_CHECK(feature_t::missing(std::numeric_limits<scalar_t>::quiet_NaN()));

    UTEST_CHECK(!feature_t::missing(-1.0));
    UTEST_CHECK(!feature_t::missing(+0.0));
    UTEST_CHECK(!feature_t::missing(+1.0));
    UTEST_CHECK(!feature_t::missing(+123.0));
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
    const auto make_feature_cont = [] (const string_t& name, feature_type type = feature_type::float32, tensor3d_dims_t dims = make_dims(1, 1, 1))
    {
        auto feature = feature_t{name}.scalar(type, dims);
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
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont("f", feature_type::float32, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont("f")), "name=f,type=float32,dims=1x1x1,labels[],mandatory");

    UTEST_CHECK_EQUAL(make_feature_cont_opt("f"), make_feature_cont_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont_opt("f"), make_feature_cont_opt("ff"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont_opt("f")), "name=f,type=float32,dims=1x1x1,labels[],optional");

    UTEST_CHECK_EQUAL(make_feature_cate("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate("f"), make_feature_cate("x"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate("f")), "name=f,type=sclass,dims=1x1x1,labels[cate0,cate1,cate2],mandatory");

    UTEST_CHECK_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("x"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("f", feature_type::mclass));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate_opt("f")), "name=f,type=sclass,dims=1x1x1,labels[cate_opt0,cate_opt1],optional");

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

UTEST_END_MODULE()
