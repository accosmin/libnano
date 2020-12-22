#include <fstream>
#include <utest/utest.h>
#include <nano/dataset/tabular.h>

using namespace nano;

static auto feature_cont()
{
    return feature_t{"cont"};
}

static auto feature_cont_opt()
{
    return feature_t{"cont_opt"}.placeholder("?");
}

static auto feature_cate()
{
    return feature_t{"cate"}.discrete(strings_t{"cate0", "cate1", "cate2"});
}

static auto feature_cate_opt()
{
    return feature_t{"cate_opt"}.discrete(strings_t{"cate_opt0", "cate_opt1"}).placeholder("?");
}

class fixture_dataset_t final : public tabular_dataset_t
{
public:

    static auto data_path() { return "test_dataset_tabular_data.csv"; }
    static auto test_path() { return "test_dataset_tabular_test.csv"; }

    static auto csvs(tensor_size_t data_expected = 20, tensor_size_t test_expected = 10)
    {
        return csvs_t(
        {
            csv_t{data_path()}.delim(",").header(false).expected(data_expected).skip('@'),
            csv_t{test_path()}.delim(",").header(true).expected(test_expected).skip('@').testing(0, test_expected)
        });
    }

    fixture_dataset_t() :
        fixture_dataset_t(fixture_dataset_t::csvs(), features_t{})
    {
    }

    fixture_dataset_t(features_t features, size_t target = string_t::npos) :
        fixture_dataset_t(fixture_dataset_t::csvs(), std::move(features), target)
    {
    }

    fixture_dataset_t(csvs_t csvs, features_t features, size_t target = string_t::npos) :
        tabular_dataset_t(std::move(csvs), std::move(features), target)
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

    static void write_data(const char* path)
    {
        std::ofstream os(path);
        write(os, 1, 20, false);
        UTEST_REQUIRE(os);
    }

    static void write_test(const char* path)
    {
        std::ofstream os(path);
        write(os, 21, 10, true);
        UTEST_REQUIRE(os);
    }

    static void check(const scalar_t value, int row, const int col)
    {
        ++ row;
        switch (col)
        {
        case 0:     check(value, row); break;
        case 1:     check(value, (row % 2 == 0) ? feature_t::placeholder_value() : (3.0 - 0.2 * row)); break;
        case 2:     check(value, row % 3); break;
        case 3:     check(value, (row % 4 == 0) ? feature_t::placeholder_value() : (row % 2)); break;
        default:    UTEST_REQUIRE(false);
        }
    }

    static void check(const scalar_t value, const scalar_t ground)
    {
        UTEST_CHECK_EQUAL(std::isfinite(value), std::isfinite(ground));

        if (std::isfinite(value))
        {
            UTEST_CHECK_CLOSE(value, ground, 1e-8);
        }
    }

    static void write(std::ostream& os, const int begin, const int size, const bool header)
    {
        if (header)
        {
            os << "cont,cont_opt,cate,cate_opt\n";
        }

        for (auto index = begin; index < begin + size; ++ index)
        {
            os << index << ",";
            (index % 2 == 0) ? (os << "?,") : (os << (3.0 - 0.2 * index) << ",");
            os << "cate" << (index % 3) << ",";
            (index % 4 == 0) ? (os << "?,") : (os << "cate_opt" << (index % 2) << ",");
            os << "\n";

            if (index % 7 == 0) { os << "\n"; }
            if (index % 9 == 0) { os << "@ this line should be skipped\n"; }
        }
    }
};

UTEST_BEGIN_MODULE(test_dataset_tabular)

UTEST_CASE(empty)
{
    auto dataset = fixture_dataset_t{};

    UTEST_CHECK(!dataset.target());
    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_THROW(dataset.feature(0), std::out_of_range);
}

UTEST_CASE(config_no_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}};

    UTEST_CHECK(!dataset.target());
    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_cate());
    UTEST_CHECK_EQUAL(dataset.feature(3), feature_cate_opt());
}

UTEST_CASE(config_with_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0};

    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_cate());
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_cate_opt());
    UTEST_CHECK_EQUAL(dataset.target(), feature_cont());
}

UTEST_CASE(noload_no_data)
{
    const auto csvs = csvs_t{};
    auto dataset = fixture_dataset_t{csvs, {feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_no_features)
{
    auto dataset = fixture_dataset_t{};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_few_features)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate()}, 0};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_wrong_features0)
{
    auto dataset = fixture_dataset_t{{feature_cont_opt(), feature_cont(), feature_cate(), feature_cate_opt()}, 1};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_wrong_features1)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate_opt(), feature_cate()}, 0};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_wrong_expected0)
{
    const auto csvs = fixture_dataset_t::csvs(21, 10);
    auto dataset = fixture_dataset_t{csvs, {feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_wrong_expected0)
{
    const auto csvs = fixture_dataset_t::csvs(20, 9);
    auto dataset = fixture_dataset_t{csvs, {feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_invalid_target0)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 4};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_invalid_target1)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 1};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(noload_invalid_target2)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 3};
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(load_no_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}};

    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.features(), 4);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_cate());
    UTEST_CHECK_EQUAL(dataset.feature(3), feature_cate_opt());
    UTEST_CHECK(!dataset.target());
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);

    UTEST_CHECK_EQUAL(dataset.idims(), make_dims(4, 1, 1));
    UTEST_CHECK_EQUAL(dataset.tdims(), make_dims(0, 1, 1));

    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.train_samples(), ::nano::arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.test_samples(), ::nano::arange(20, 30));

    const auto inputs = dataset.inputs(::nano::arange(10, 30));
    const auto targets = dataset.targets(::nano::arange(10, 30));

    UTEST_CHECK_EQUAL(inputs.dims(), make_dims(20, 4, 1, 1));
    UTEST_CHECK_EQUAL(targets.dims(), make_dims(20, 0, 1, 1));

    for (auto index = 0; index < 20; ++ index)
    {
        fixture_dataset_t::check(inputs(index, 0, 0, 0), index + 10, 0);
        fixture_dataset_t::check(inputs(index, 1, 0, 0), index + 10, 1);
        fixture_dataset_t::check(inputs(index, 2, 0, 0), index + 10, 2);
        fixture_dataset_t::check(inputs(index, 3, 0, 0), index + 10, 3);
    }
}

UTEST_CASE(load_with_cont_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0};

    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.features(), 3);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_cate());
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_cate_opt());
    UTEST_CHECK_EQUAL(dataset.target(), feature_cont());
    UTEST_CHECK_EQUAL(dataset.type(), task_type::regression);

    UTEST_CHECK_EQUAL(dataset.idims(), make_dims(3, 1, 1));
    UTEST_CHECK_EQUAL(dataset.tdims(), make_dims(1, 1, 1));

    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.train_samples(), ::nano::arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.test_samples(), ::nano::arange(20, 30));

    const auto inputs = dataset.inputs(::nano::arange(10, 30));
    const auto targets = dataset.targets(::nano::arange(10, 30));

    UTEST_CHECK_EQUAL(inputs.dims(), make_dims(20, 3, 1, 1));
    UTEST_CHECK_EQUAL(targets.dims(), make_dims(20, 1, 1, 1));

    for (auto index = 0; index < 20; ++ index)
    {
        fixture_dataset_t::check(targets(index, 0, 0, 0), index + 10, 0);
        fixture_dataset_t::check(inputs(index, 0, 0, 0), index + 10, 1);
        fixture_dataset_t::check(inputs(index, 1, 0, 0), index + 10, 2);
        fixture_dataset_t::check(inputs(index, 2, 0, 0), index + 10, 3);
    }
}

UTEST_CASE(load_with_cate_target)
{
    auto dataset = fixture_dataset_t{{feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 2};

    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.features(), 3);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_cate_opt());
    UTEST_CHECK_EQUAL(dataset.target(), feature_cate());
    UTEST_CHECK_EQUAL(dataset.type(), task_type::sclassification);

    UTEST_CHECK_EQUAL(dataset.idims(), make_dims(3, 1, 1));
    UTEST_CHECK_EQUAL(dataset.tdims(), make_dims(3, 1, 1));

    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.train_samples(), ::nano::arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.test_samples(), ::nano::arange(20, 30));

    const auto inputs = dataset.inputs(::nano::arange(10, 30));
    const auto targets = dataset.targets(::nano::arange(10, 30));

    UTEST_CHECK_EQUAL(inputs.dims(), make_dims(20, 3, 1, 1));
    UTEST_CHECK_EQUAL(targets.dims(), make_dims(20, 3, 1, 1));

    for (auto index = 0; index < 20; ++ index)
    {
        fixture_dataset_t::check(inputs(index, 0, 0, 0), index + 10, 0);
        fixture_dataset_t::check(inputs(index, 1, 0, 0), index + 10, 1);
        tensor_size_t category = 0;
        targets.vector(index).maxCoeff(&category);
        fixture_dataset_t::check(category, index + 10, 2);
        fixture_dataset_t::check(inputs(index, 2, 0, 0), index + 10, 3);
    }
}

UTEST_END_MODULE()
