#include <nano/tensor.h>
#include <nano/core/table.h>
#include <nano/core/chrono.h>
#include <nano/core/logger.h>
#include <nano/core/cmdline.h>

namespace
{
    using namespace nano;

    template <typename tscalar>
    auto make_scalar()
    {
        auto rng = make_rng();
        return make_udist<tscalar>(-1, +1)(rng);
    }

    template <typename tscalar>
    auto make_vector(const tensor_size_t dims)
    {
        tensor_vector_t<tscalar> x(dims);
        x.setRandom();
        return x;
    }

    template <typename tscalar>
    auto make_matrix(const tensor_size_t rows, const tensor_size_t cols)
    {
        tensor_matrix_t<tscalar> x(rows, cols);
        x.setRandom();
        return x;
    }

    template <typename toperator>
    void store(row_t& row, const tensor_size_t flops, const toperator& op)
    {
        const auto trials = size_t(10);
        const auto duration = nano::measure<picoseconds_t>(op, trials);
        row << nano::gflops(flops, duration);
    }

    struct copy1_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto x = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 2 * dims, [&] () { std::memcpy(z.data(), x.data(), sizeof(tscalar) * static_cast<size_t>(dims)); });
        }
    };

    struct copy2_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto x = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 2 * dims, [&] () { std::copy(x.data(), x.data() + dims, z.data()); });
        }
    };

    struct copy3_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto x = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 2 * dims, [&] () { z = x; });
        }
    };

    struct blas11_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto c = make_scalar<tscalar>();
            auto x = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, dims, [&] () { z = x.array() + c; });
        }
    };

    struct blas12_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto x = make_vector<tscalar>(dims);
            auto y = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, dims, [&] () { z = x.array() + y.array(); });
        }
    };

    struct blas13_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto a = make_scalar<tscalar>();
            auto c = make_scalar<tscalar>();
            auto x = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 2 * dims, [&] () { z = x.array() * a + c; });
        }
    };

    struct blas14_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto a = make_scalar<tscalar>();
            auto x = make_vector<tscalar>(dims);
            auto y = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 2 * dims, [&] () { z = x.array() * a + y.array(); });
        }
    };

    struct blas15_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto a = make_scalar<tscalar>();
            auto b = make_scalar<tscalar>();
            auto x = make_vector<tscalar>(dims);
            auto y = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 3 * dims, [&] () { z = x.array() * a + y.array() * b; });
        }
    };

    struct blas16_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto a = make_scalar<tscalar>();
            auto b = make_scalar<tscalar>();
            auto c = make_scalar<tscalar>();
            auto x = make_vector<tscalar>(dims);
            auto y = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 4 * dims, [&] () { z = x.array() * a + y.array() * b + c; });
        }
    };

    struct blas21_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto A = make_matrix<tscalar>(dims, dims);
            auto x = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 2 * dims * dims, [&] () { z = A * x; });
        }
    };

    struct blas22_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto A = make_matrix<tscalar>(dims, dims);
            auto x = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);
            auto c = make_scalar<tscalar>();

            store(row, 2 * dims * dims + dims, [&] () { z = (A * x).array() + c; });
        }
    };

    struct blas23_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto A = make_matrix<tscalar>(dims, dims);
            auto x = make_vector<tscalar>(dims);
            auto y = make_vector<tscalar>(dims);
            auto z = make_vector<tscalar>(dims);

            store(row, 2 * dims * dims + dims, [&] () { z = A * x + y; });
        }
    };

    struct blas24_t
    {
        template <typename tscalar> static void measure(const tensor_size_t dims, row_t& row)
        {
            auto Z = make_matrix<tscalar>(dims, dims);
            auto C = make_matrix<tscalar>(dims, dims);
            auto x = make_vector<tscalar>(dims);
            auto y = make_vector<tscalar>(dims);

            store(row, 2 * dims * dims + dims, [&] () { Z.noalias() = x * y.transpose() + C; });
        }
    };

    struct blas31_t
    {
        template <typename tscalar> static auto measure(const tensor_size_t dims, row_t& row)
        {
            auto A = make_matrix<tscalar>(dims, dims);
            auto B = make_matrix<tscalar>(dims, dims);
            auto Z = make_matrix<tscalar>(dims, dims);

            store(row, 2 * dims * dims * dims, [&] () { Z.noalias() = A * B; });
        }
    };

    struct blas32_t
    {
        template <typename tscalar> static auto measure(const tensor_size_t dims, row_t& row)
        {
            auto A = make_matrix<tscalar>(dims, dims);
            auto B = make_matrix<tscalar>(dims, dims);
            auto C = make_matrix<tscalar>(dims, dims);
            auto Z = make_matrix<tscalar>(dims, dims);

            store(row, 2 * dims * dims * dims + dims * dims, [&] () { Z.noalias() = A * B + C; });
        }
    };

    struct blas33_t
    {
        template <typename tscalar> static auto measure(const tensor_size_t dims, row_t& row)
        {
            auto A = make_matrix<tscalar>(dims, dims);
            auto B = make_matrix<tscalar>(dims, dims);
            auto C = make_matrix<tscalar>(dims, dims);
            auto Z = make_matrix<tscalar>(dims, dims);

            store(row, 2 * dims * dims * dims + dims * dims, [&] () { Z.noalias() = A * B.transpose() + C; });
        }
    };

    template <typename top>
    void foreach_dims(const tensor_size_t min, const tensor_size_t max, const top& op)
    {
        for (tensor_size_t dim = min; dim <= max; dim *= 2)
        {
            op(dim);
        }
    }

    template <typename tscalar, typename top>
    void foreach_dims_row(const tensor_size_t min, const tensor_size_t max, row_t& row, const top& op)
    {
        for (tensor_size_t dim = min; dim <= max; dim *= 2)
        {
            op.template measure<tscalar>(dim, row);
        }
    }

    void header1(const tensor_size_t min, const tensor_size_t max, const char* section_name, table_t& table)
    {
        auto& row = table.header();
        row << " "
            << nano::colspan(static_cast<size_t>(std::log2(static_cast<double>(max) / static_cast<double>(min))) + 1U)
            << nano::alignment::center << section_name;
        table.delim();
    }

    void header2(const tensor_size_t min, const tensor_size_t max, const char* operation_name, table_t& table)
    {
        auto& row = table.header();
        row << operation_name;
        foreach_dims(min, max, [&] (const tensor_size_t dims)
        {
            const auto kilo = tensor_size_t(1024);
            const auto mega = kilo * kilo;
            const auto value = (dims < kilo) ? dims : (dims < mega ? (dims / kilo) : (dims / mega));
            const auto units = (dims < kilo) ? string_t("") : (dims < mega ? string_t("K") : string_t("M"));
            row << nano::scat(value, units);
        });
        table.delim();
    }

    template <typename tscalar>
    void copy(const tensor_size_t min, const tensor_size_t max, table_t& table)
    {
        foreach_dims_row<tscalar>(min, max, table.append() << "z = x (std::memcpy)", copy1_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = x (std::copy)", copy2_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = x (Eigen)", copy3_t{});
    }

    template <typename tscalar>
    void blas1(const tensor_size_t min, const tensor_size_t max, table_t& table)
    {
        foreach_dims_row<tscalar>(min, max, table.append() << "z = x + c", blas11_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = x + y", blas12_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = ax + c", blas13_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = ax + y", blas14_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = ax + by", blas15_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = ax + by + c", blas16_t{});
    }

    template <typename tscalar>
    void blas2(const tensor_size_t min, const tensor_size_t max, table_t& table)
    {
        foreach_dims_row<tscalar>(min, max, table.append() << "z = Ax", blas21_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = Ax + c", blas22_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "z = Ax + y", blas23_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "Z = xy^t + C", blas24_t{});
    }

    template <typename tscalar>
    void blas3(const tensor_size_t min, const tensor_size_t max, table_t& table)
    {
        foreach_dims_row<tscalar>(min, max, table.append() << "Z = AB", blas31_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "Z = AB + C", blas32_t{});
        foreach_dims_row<tscalar>(min, max, table.append() << "Z = AB^t + C", blas33_t{});
    }
}

static int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark linear algebra operations using Eigen");
    cmdline.add("", "min-dims",     "minimum number of dimensions [1, 1024]", "16");
    cmdline.add("", "max-dims",     "maximum number of dimensions [--min-dims, 4096]", "1024");
    cmdline.add("", "copy",         "benchmark copy operations (vector to vector)");
    cmdline.add("", "blas1",        "benchmark level1 BLAS operations (vector-vector)");
    cmdline.add("", "blas2",        "benchmark level2 BLAS operations (matrix-vector)");
    cmdline.add("", "blas3",        "benchmark level3 BLAS operations (matrix-matrix)");

    cmdline.process(argc, argv);

    // check arguments and options
    const auto min_dims = std::clamp(cmdline.get<tensor_size_t>("min-dims"), tensor_size_t(1), tensor_size_t(1024));
    const auto max_dims = std::clamp(cmdline.get<tensor_size_t>("max-dims"), min_dims, tensor_size_t(4096));
    const auto copy = cmdline.has("copy");
    const auto blas1 = cmdline.has("blas1");
    const auto blas2 = cmdline.has("blas2");
    const auto blas3 = cmdline.has("blas3");

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    if (!blas1 && !blas2 && !blas3 && !copy)
    {
        cmdline.usage();
        return EXIT_FAILURE;
    }

    table_t table;

    if (copy)
    {
        const auto min = 1024 * min_dims;
        const auto max = 1024 * max_dims;

        header1(min, max, "vector dimension [GB/s]", table);
        header2(min, max, "operation (float)", table); ::copy<float>(min, max, table); table.delim();
        header2(min, max, "operation (double)", table); ::copy<double>(min, max, table); table.delim();
        header2(min, max, "operation (long double)", table); ::copy<long double>(min, max, table);
    }
    if (blas1)
    {
        const auto min = 1024 * min_dims;
        const auto max = 1024 * max_dims;

        if (copy)
        {
            table.delim();
        }
        header1(min, max, "vector dimension [GFLOPS]", table);
        header2(min, max, "operation (float)", table); ::blas1<float>(min, max, table); table.delim();
        header2(min, max, "operation (double)", table); ::blas1<double>(min, max, table); table.delim();
        header2(min, max, "operation (long double)", table); ::blas1<long double>(min, max, table);
    }
    if (blas2)
    {
        const auto min = min_dims;
        const auto max = max_dims;

        if (copy || blas1)
        {
            table.delim();
        }
        header1(min, max, "vector dimension [GFLOPS]", table);
        header2(min, max, "operation (float)", table); ::blas2<float>(min, max, table); table.delim();
        header2(min, max, "operation (double)", table); ::blas2<double>(min, max, table); table.delim();
        header2(min, max, "operation (long double)", table); ::blas2<long double>(min, max, table);
    }
    if (blas3)
    {
        const auto min = min_dims;
        const auto max = max_dims;

        if (copy || blas1 || blas2)
        {
            table.delim();
        }
        header1(min, max, "matrix dimension [GFLOPS]", table);
        header2(min, max, "operation (float)", table); ::blas3<float>(min, max, table); table.delim();
        header2(min, max, "operation (double)", table); ::blas3<double>(min, max, table); table.delim();
        header2(min, max, "operation (long double)", table); ::blas3<long double>(min, max, table);
    }

    std::cout << table;
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
