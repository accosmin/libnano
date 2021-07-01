#include <fstream>
#include "fixture/dataset.h"
#include <nano/dataset/tabular.h>

using namespace nano;

static auto feature_cont()
{
    return feature_t{"cont"}.scalar(feature_type::float64);
}

static auto feature_cate(bool with_labels = false)
{
    auto feature = feature_t{"cate"};
    if (with_labels)
    {
        feature.sclass(strings_t{"cate0", "cate1", "cate2"});
    }
    else
    {
        feature.sclass(3);
    }
    return feature;
}

class fixture_dataset_t final : public tabular_dataset_t
{
public:

    static auto data_path() { return "test_dataset_tabular_data.csv"; }
    static auto test_path() { return "test_dataset_tabular_test.csv"; }

    static auto csvs(int data_size = 20, int test_size = 10)
    {
        return csvs_t(
        {
            csv_t{data_path()}.delim(",").header(false).expected(data_size).skip('@').placeholder("?"),
            csv_t{test_path()}.delim(",").header(true).expected(test_size).skip('@').testing(0, test_size).placeholder("?")
        });
    }

    fixture_dataset_t() :
        fixture_dataset_t(fixture_dataset_t::csvs(), features_t{})
    {
    }

    explicit fixture_dataset_t(features_t features, size_t target = string_t::npos, bool too_many_labels = false) :
        fixture_dataset_t(fixture_dataset_t::csvs(), std::move(features), target, too_many_labels)
    {
    }

    fixture_dataset_t(csvs_t csvs, features_t features, size_t target = string_t::npos, bool too_many_labels = false) :
        tabular_dataset_t(std::move(csvs), std::move(features), target),
        m_too_many_labels(too_many_labels)
    {
        std::remove(data_path());
        std::remove(test_path());

        write_data(data_path());
        write_test(test_path());
    }

    fixture_dataset_t(fixture_dataset_t&&) = default;
    fixture_dataset_t(const fixture_dataset_t&) = delete;
    fixture_dataset_t& operator=(fixture_dataset_t&&) = default;
    fixture_dataset_t& operator=(const fixture_dataset_t&) = delete;

    ~fixture_dataset_t() override
    {
        std::remove(data_path());
        std::remove(test_path());
    }

    void write_data(const char* path) const
    {
        std::ofstream os(path);
        write(os, 1, 20, false);
        UTEST_REQUIRE(os);
    }

    void write_test(const char* path) const
    {
        std::ofstream os(path);
        write(os, 21, 10, true);
        UTEST_REQUIRE(os);
    }

    void write(std::ostream& os, int begin, int size, bool header) const
    {
        if (header)
        {
            os << "cont,cate\n";
        }

        for (auto index = begin; index < begin + size; ++ index)
        {
            (index % 2 == 0) ? (os << "?,") : (os << (3.0 - 0.2 * index) << ",");
            (index % 5 == 4) ? (os << "?,") : (os << "cate" << ((index - 1) % (m_too_many_labels ? 4 : 3)) << ",");
            os << "\n";

            if (index % 7 == 0) { os << "\n"; }
            if (index % 9 == 0) { os << "@ this line should be skipped\n"; }
        }
    }

    static auto mask_cate() { return make_tensor<uint8_t>(make_dims(4), 0xEF, 0x7B, 0xDE, 0xF4); }
    static auto mask_cont() { return make_tensor<uint8_t>(make_dims(4), 0xAA, 0xAA, 0xAA, 0xA8); }

    static auto values_cate()
    {
        return make_tensor<uint8_t>(make_dims(30),
            0, 1, 2, 0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 2,
            0, 1, 2, 0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 2);
    }

    static auto values_cont()
    {
        return make_tensor<scalar_t>(make_dims(30, 1, 1, 1),
            +2.8, +0.0, +2.4, +0.0, +2.0, +0.0, +1.6, +0.0, +1.2, +0.0, +0.8, +0.0, +0.4, +0.0, +0.0,
            +0.0, -0.4, +0.0, -0.8, +0.0, -1.2, +0.0, -1.6, +0.0, -2.0, +0.0, -2.4, +0.0, -2.8, +0.0);
    }

private:

    // attributes
    bool    m_too_many_labels{false}; ///< toggle whether to write an invalid number of labels for categorical features
};

UTEST_BEGIN_MODULE(test_dataset_tabular)

UTEST_CASE(empty)
{
    auto dataset = fixture_dataset_t{};

    UTEST_CHECK_EQUAL(dataset.samples(), 0);
    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_TENSOR_EQUAL(dataset.test_samples(), indices_t{});
    UTEST_CHECK_TENSOR_EQUAL(dataset.train_samples(), indices_t{});
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);
}

UTEST_CASE(no_target_no_load)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate()}};

    UTEST_CHECK_EQUAL(dataset.samples(), 0);
    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);
    UTEST_CHECK_TENSOR_EQUAL(dataset.test_samples(), indices_t{});
    UTEST_CHECK_TENSOR_EQUAL(dataset.train_samples(), indices_t{});
}

UTEST_CASE(with_target_no_load)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate()}, 0U};

    UTEST_CHECK_EQUAL(dataset.samples(), 0);
    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);
    UTEST_CHECK_TENSOR_EQUAL(dataset.test_samples(), indices_t{});
    UTEST_CHECK_TENSOR_EQUAL(dataset.train_samples(), indices_t{});
}

UTEST_CASE(cannot_load_no_data)
{
    const auto csvs = csvs_t{};
    auto dataset = fixture_dataset_t{csvs, {feature_cont(), feature_cate()}, 0U};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_no_features)
{
    auto dataset = fixture_dataset_t{};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_invalid_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate()}, 2U};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_unsupported_mclass)
{
    const auto feature_mclass = feature_t{"feature"}.mclass(3);
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate(), feature_mclass}};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_unsupported_struct)
{
    const auto feature_struct = feature_t{"feature"}.scalar(feature_type::uint8, make_dims(3, 32, 32));
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate(), feature_struct}};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_wrong_expected_csv_length0)
{
    const auto csvs = fixture_dataset_t::csvs(21, 10);
    auto dataset = fixture_dataset_t{csvs, {feature_cont(), feature_cate()}};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_wrong_expected_csv_length1)
{
    const auto csvs = fixture_dataset_t::csvs(20, 9);
    auto dataset = fixture_dataset_t{csvs, {feature_cont(), feature_cate()}};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_too_many_labels)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate()}, string_t::npos, true};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(load_no_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate()}};

    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.features(), 2);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_cate(true));
    UTEST_CHECK_TENSOR_EQUAL(dataset.test_samples(), arange(20, 30));
    UTEST_CHECK_TENSOR_EQUAL(dataset.train_samples(), arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);

    check_inputs(dataset, 0, feature_cont(), fixture_dataset_t::values_cont(), fixture_dataset_t::mask_cont());
    check_inputs(dataset, 1, feature_cate(true), fixture_dataset_t::values_cate(), fixture_dataset_t::mask_cate());
}

UTEST_CASE(load_cate_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate()}, 1U};

    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.features(), 1);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont());
    UTEST_CHECK_TENSOR_EQUAL(dataset.test_samples(), arange(20, 30));
    UTEST_CHECK_TENSOR_EQUAL(dataset.train_samples(), arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.type(), task_type::sclassification);

    check_inputs(dataset, 0, feature_cont(), fixture_dataset_t::values_cont(), fixture_dataset_t::mask_cont());
    check_target(dataset, feature_cate(true), fixture_dataset_t::values_cate(), fixture_dataset_t::mask_cate());
}

UTEST_CASE(load_cont_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cate()}, 0U};

    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.features(), 1);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cate(true));
    UTEST_CHECK_TENSOR_EQUAL(dataset.test_samples(), arange(20, 30));
    UTEST_CHECK_TENSOR_EQUAL(dataset.train_samples(), arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.type(), task_type::regression);

    check_target(dataset, feature_cont(), fixture_dataset_t::values_cont(), fixture_dataset_t::mask_cont());
    check_inputs(dataset, 0, feature_cate(true), fixture_dataset_t::values_cate(), fixture_dataset_t::mask_cate());
}

UTEST_END_MODULE()
