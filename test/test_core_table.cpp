#include <utest/utest.h>
#include <nano/core/table.h>

using namespace nano;

template <typename tscalar>
std::ostream& operator<<(std::ostream& os, const std::vector<std::pair<size_t, tscalar>>& values)
{
    os << "{";
    for (const auto& cv : values)
    {
        os << "{" << cv.first << "," << cv.second << "}";
    }
    return os << "}";
}

template <typename tscalar>
std::ostream& operator<<(std::ostream& os, const std::vector<tscalar>& values)
{
    os << "{";
    for (const auto& value : values)
    {
        os << "{" << value << "}";
    }
    return os << "}";
}

UTEST_BEGIN_MODULE(test_core_table)

UTEST_CASE(make_less)
{
    const auto less = nano::make_less_from_string<int>();

    UTEST_CHECK_EQUAL(less("1", "2"), true);
    UTEST_CHECK_EQUAL(less("2", "1"), false);
    UTEST_CHECK_EQUAL(less("x", "1"), true);
    UTEST_CHECK_EQUAL(less("2", "x"), true);
}

UTEST_CASE(make_greater)
{
    const auto greater = nano::make_greater_from_string<int>();

    UTEST_CHECK_EQUAL(greater("1", "2"), false);
    UTEST_CHECK_EQUAL(greater("2", "1"), true);
    UTEST_CHECK_EQUAL(greater("x", "1"), true);
    UTEST_CHECK_EQUAL(greater("2", "x"), true);
}

UTEST_CASE(table)
{
    nano::table_t t1;
    t1.header() << "head" << "col1" << "col2";
    t1.delim();
    t1.append() << "row1" << "v11" << "v12";
    t1.append() << "row2" << "v21" << "v22";
    t1.append() << "row3" << "v21" << "v22";

    UTEST_CHECK_EQUAL(t1.rows(), 5U);
    UTEST_CHECK_EQUAL(t1.cols(), 3U);
}

UTEST_CASE(table_rows)
{
    using namespace nano;

    nano::table_t table;
    table.header() << "head" << colspan(2) << "colx" << colspan(1) << "col3";
    table.append() << "row1" << "1000" << "9000" << "4000";
    table.append() << "row2" << "3200" << colspan(2) << "2000";
    table.append() << "row3" << colspan(3) << "2500";

    UTEST_CHECK_EQUAL(table.rows(), 4U);
    UTEST_CHECK_EQUAL(table.cols(), 4U);

    UTEST_CHECK_EQUAL(table.row(0).cols(), 4U);
    UTEST_CHECK_EQUAL(table.row(1).cols(), 4U);
    UTEST_CHECK_EQUAL(table.row(2).cols(), 4U);
    UTEST_CHECK_EQUAL(table.row(3).cols(), 4U);

    UTEST_CHECK_EQUAL(table.row(0).data(0), "head"); UTEST_CHECK_EQUAL(table.row(0).mark(0), "");
    UTEST_CHECK_EQUAL(table.row(0).data(1), "colx"); UTEST_CHECK_EQUAL(table.row(0).mark(1), "");
    UTEST_CHECK_EQUAL(table.row(0).data(2), "colx"); UTEST_CHECK_EQUAL(table.row(0).mark(2), "");
    UTEST_CHECK_EQUAL(table.row(0).data(3), "col3"); UTEST_CHECK_EQUAL(table.row(0).mark(3), "");

    UTEST_CHECK_EQUAL(table.row(1).data(0), "row1"); UTEST_CHECK_EQUAL(table.row(1).mark(0), "");
    UTEST_CHECK_EQUAL(table.row(1).data(1), "1000"); UTEST_CHECK_EQUAL(table.row(1).mark(1), "");
    UTEST_CHECK_EQUAL(table.row(1).data(2), "9000"); UTEST_CHECK_EQUAL(table.row(1).mark(2), "");
    UTEST_CHECK_EQUAL(table.row(1).data(3), "4000"); UTEST_CHECK_EQUAL(table.row(1).mark(3), "");

    UTEST_CHECK_EQUAL(table.row(2).data(0), "row2"); UTEST_CHECK_EQUAL(table.row(2).mark(0), "");
    UTEST_CHECK_EQUAL(table.row(2).data(1), "3200"); UTEST_CHECK_EQUAL(table.row(2).mark(1), "");
    UTEST_CHECK_EQUAL(table.row(2).data(2), "2000"); UTEST_CHECK_EQUAL(table.row(2).mark(2), "");
    UTEST_CHECK_EQUAL(table.row(2).data(3), "2000"); UTEST_CHECK_EQUAL(table.row(2).mark(3), "");

    UTEST_CHECK_EQUAL(table.row(3).data(0), "row3"); UTEST_CHECK_EQUAL(table.row(3).mark(0), "");
    UTEST_CHECK_EQUAL(table.row(3).data(1), "2500"); UTEST_CHECK_EQUAL(table.row(3).mark(1), "");
    UTEST_CHECK_EQUAL(table.row(3).data(2), "2500"); UTEST_CHECK_EQUAL(table.row(3).mark(2), "");
    UTEST_CHECK_EQUAL(table.row(3).data(3), "2500"); UTEST_CHECK_EQUAL(table.row(3).mark(3), "");

    {
        using values_t = std::vector<std::pair<size_t, int>>;
        const auto values0 = table.row(0).collect<int>();
        const auto values1 = table.row(1).collect<int>();
        const auto values2 = table.row(2).collect<int>();
        const auto values3 = table.row(3).collect<int>();
        UTEST_CHECK_EQUAL(values0, (values_t{}));
        UTEST_CHECK_EQUAL(values1, (values_t{{1, 1000}, {2, 9000}, {3, 4000}}));
        UTEST_CHECK_EQUAL(values2, (values_t{{1, 3200}, {2, 2000}, {3, 2000}}));
        UTEST_CHECK_EQUAL(values3, (values_t{{1, 2500}, {2, 2500}, {3, 2500}}));
    }
    {
        const auto indices0 = table.row(0).select<int>([] (const auto value) { return value >= 3000; });
        const auto indices1 = table.row(1).select<int>([] (const auto value) { return value >= 3000; });
        const auto indices2 = table.row(2).select<int>([] (const auto value) { return value >= 3000; });
        const auto indices3 = table.row(3).select<int>([] (const auto value) { return value >= 3000; });
        UTEST_CHECK_EQUAL(indices0, (std::vector<size_t>{}));
        UTEST_CHECK_EQUAL(indices1, (std::vector<size_t>{2, 3}));
        UTEST_CHECK_EQUAL(indices2, (std::vector<size_t>{1}));
        UTEST_CHECK_EQUAL(indices3, (std::vector<size_t>{}));
    }
}

UTEST_CASE(table_mark)
{
    nano::table_t table;
    table.header() << "name " << "col1" << "col2" << "col3";
    table.append() << "name1" << "1000" << "9000" << "4000";
    table.append() << "name2" << "3200" << "2000" << "5000";
    table.append() << "name3" << "1500" << "7000" << "6000";

    for (size_t r = 0; r < table.rows(); ++ r)
    {
        for (size_t c = 0; c < table.cols(); ++ c)
        {
            UTEST_CHECK_EQUAL(table.row(r).mark(c), "");
        }
    }
    {
        auto tablex = table;
        tablex.mark(nano::make_marker_minimum_col<int>(), "*");
        UTEST_CHECK_EQUAL(tablex.row(1).mark(1), "*");
        UTEST_CHECK_EQUAL(tablex.row(2).mark(2), "*");
        UTEST_CHECK_EQUAL(tablex.row(3).mark(1), "*");
    }
    {
        auto tablex = table;
        tablex.mark(nano::make_marker_maximum_col<int>(), "*");
        UTEST_CHECK_EQUAL(tablex.row(1).mark(2), "*");
        UTEST_CHECK_EQUAL(tablex.row(2).mark(3), "*");
        UTEST_CHECK_EQUAL(tablex.row(3).mark(2), "*");
    }
}

UTEST_CASE(table_sort)
{
    nano::table_t table;
    table.header() << "name" << "col1" << "col2" << "col3";
    table.append() << "name" << "1000" << "9000" << "4000";
    table.append() << "name" << "3200" << "2000" << "6000";
    table.append() << "name" << "1500" << "2000" << "5000";

    {
        auto tablex = table;
        tablex.sort([] (const string_t& lhs, const string_t& rhs) { return lhs < rhs; }, {0});

        UTEST_CHECK_EQUAL(tablex.row(0).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(0).data(1), "col1");
        UTEST_CHECK_EQUAL(tablex.row(0).data(2), "col2");
        UTEST_CHECK_EQUAL(tablex.row(0).data(3), "col3");

        UTEST_CHECK_EQUAL(tablex.row(1).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(1).data(1), "1500");
        UTEST_CHECK_EQUAL(tablex.row(1).data(2), "2000");
        UTEST_CHECK_EQUAL(tablex.row(1).data(3), "5000");

        UTEST_CHECK_EQUAL(tablex.row(2).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(2).data(1), "3200");
        UTEST_CHECK_EQUAL(tablex.row(2).data(2), "2000");
        UTEST_CHECK_EQUAL(tablex.row(2).data(3), "6000");

        UTEST_CHECK_EQUAL(tablex.row(3).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(3).data(1), "1000");
        UTEST_CHECK_EQUAL(tablex.row(3).data(2), "9000");
        UTEST_CHECK_EQUAL(tablex.row(3).data(3), "4000");
    }
    {
        auto tablex = table;
        tablex.sort(nano::make_less_from_string<int>(), {2, 3});

        UTEST_CHECK_EQUAL(tablex.row(0).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(0).data(1), "col1");
        UTEST_CHECK_EQUAL(tablex.row(0).data(2), "col2");
        UTEST_CHECK_EQUAL(tablex.row(0).data(3), "col3");

        UTEST_CHECK_EQUAL(tablex.row(1).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(1).data(1), "1500");
        UTEST_CHECK_EQUAL(tablex.row(1).data(2), "2000");
        UTEST_CHECK_EQUAL(tablex.row(1).data(3), "5000");

        UTEST_CHECK_EQUAL(tablex.row(2).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(2).data(1), "3200");
        UTEST_CHECK_EQUAL(tablex.row(2).data(2), "2000");
        UTEST_CHECK_EQUAL(tablex.row(2).data(3), "6000");

        UTEST_CHECK_EQUAL(tablex.row(3).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(3).data(1), "1000");
        UTEST_CHECK_EQUAL(tablex.row(3).data(2), "9000");
        UTEST_CHECK_EQUAL(tablex.row(3).data(3), "4000");
    }
    {
        auto tablex = table;
        tablex.sort(nano::make_greater_from_string<int>(), {1});

        UTEST_CHECK_EQUAL(tablex.row(0).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(0).data(1), "col1");
        UTEST_CHECK_EQUAL(tablex.row(0).data(2), "col2");
        UTEST_CHECK_EQUAL(tablex.row(0).data(3), "col3");

        UTEST_CHECK_EQUAL(tablex.row(1).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(1).data(1), "3200");
        UTEST_CHECK_EQUAL(tablex.row(1).data(2), "2000");
        UTEST_CHECK_EQUAL(tablex.row(1).data(3), "6000");

        UTEST_CHECK_EQUAL(tablex.row(2).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(2).data(1), "1500");
        UTEST_CHECK_EQUAL(tablex.row(2).data(2), "2000");
        UTEST_CHECK_EQUAL(tablex.row(2).data(3), "5000");

        UTEST_CHECK_EQUAL(tablex.row(3).data(0), "name");
        UTEST_CHECK_EQUAL(tablex.row(3).data(1), "1000");
        UTEST_CHECK_EQUAL(tablex.row(3).data(2), "9000");
        UTEST_CHECK_EQUAL(tablex.row(3).data(3), "4000");
    }
}

UTEST_CASE(table_stream_single_line)
{
    nano::table_t table;
    table.header() << "head" << "col1" << "col2";
    table.delim();
    table.append() << "row1" << "v11" << "v12";
    table.append() << nano::colspan(2) << "row2+v21" << "v22";
    table.append() << nano::colspan(3) << "row3+v31+v32";

    std::stringstream stream;
    stream << table;
    UTEST_CHECK_EQUAL(stream.str(),
        "|------|------|------|\n"\
        "| head | col1 | col2 |\n"\
        "|------|------|------|\n"\
        "| row1 | v11  | v12  |\n"\
        "| row2+v21    | v22  |\n"\
        "| row3+v31+v32       |\n"\
        "|------|------|------|\n"
    );
}

UTEST_END_MODULE()
