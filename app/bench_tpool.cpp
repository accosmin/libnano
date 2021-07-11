#include <iomanip>
#include <nano/tensor.h>
#include <nano/core/tpool.h>
#include <nano/core/table.h>
#include <nano/core/chrono.h>
#include <nano/core/logger.h>
#include <nano/core/cmdline.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace nano;

struct exp_t
{
    static const char* name() { return "exp"; }

    template <typename tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return (-targets * outputs).exp().sum();
    }
};

struct log_t
{
    static const char* name() { return "log"; }

    template <typename tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return ((-targets * outputs).exp() + 1).log().sum();
    }
};

struct mse_t
{
    static const char* name() { return "mse"; }

    template <typename tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return (targets - outputs).square().sum();
    }
};

template <typename toperator>
static scalar_t sti(const tensor_size_t i, const matrix_t& targets, const matrix_t& outputs)
{
    assert(targets.rows() == outputs.rows());
    assert(targets.cols() == outputs.cols());
    assert(0 <= i && i < targets.rows());

    return toperator::get(targets.row(i).array(), outputs.row(i).array());
}

template <typename toperator>
static scalar_t reduce_st(const matrix_t& targets, const matrix_t& outputs)
{
    scalar_t value = 0;
    for (tensor_size_t i = 0, size = targets.rows(); i < size; ++ i)
    {
        value += sti<toperator>(i, targets, outputs);
    }
    return value / static_cast<scalar_t>(targets.rows());
}

template <typename toperator>
static scalar_t reduce_mt(const matrix_t& targets, const matrix_t& outputs)
{
    vector_t values = vector_t::Zero(static_cast<tensor_size_t>(tpool_t::size()));
    nano::loopi(targets.rows(), [&] (tensor_size_t i, size_t t)
    {
        values(static_cast<tensor_size_t>(t)) += sti<toperator>(i, targets, outputs);
    });

    return values.sum() / static_cast<scalar_t>(targets.rows());
}

#if defined(_OPENMP)
template <typename toperator>
static scalar_t reduce_op(const matrix_t& targets, const matrix_t& outputs)
{
    vector_t values = vector_t::Zero(omp_get_max_threads());
    const auto size = targets.rows();

    #pragma omp parallel for
    for (tensor_size_t i = 0; i < size; ++ i)
    {
        const auto t = omp_get_thread_num();
        values(t) += sti<toperator>(i, targets, outputs);
    }

    return values.sum() / static_cast<scalar_t>(targets.rows());
}
#endif

static bool close(const scalar_t v1, const scalar_t v2, const char* name, const scalar_t epsilon)
{
    if (std::fabs(v1 - v2) > epsilon)
    {
        std::cerr << "mis-matching sum (" << name << "): delta=" << std::fabs(v1 - v2) << ")!" << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}

template <typename toperator>
static bool evaluate(const tensor_size_t min_size, const tensor_size_t max_size, table_t& table)
{
    std::vector<scalar_t> single_deltas;
    std::vector<scalar_t> single_values;
    std::vector<matrix_t> single_targets;
    std::vector<matrix_t> single_outputs;

    // single-thread
    auto& row1 = table.append();
    row1 << scat("reduce-", toperator::name()) << "single";
    for (tensor_size_t size = min_size; size <= max_size; size *= 2)
    {
        matrix_t targets = matrix_t::Constant(size, 10, -1);
        matrix_t outputs = matrix_t::Random(size, 10);
        for (tensor_size_t i = 0; i < size; ++ i)
        {
            targets(i, i % 10) = +1;
        }

        scalar_t value = 0;
        const auto delta = measure<nanoseconds_t>([&] { value = reduce_st<toperator>(targets, outputs); }, 16);
        row1 << "1.00";

        single_deltas.push_back(static_cast<scalar_t>(delta.count()));
        single_values.push_back(value);
        single_targets.push_back(targets);
        single_outputs.push_back(outputs);
    }

    // multi-threaded (using the thread pool)
    auto& row2 = table.append();
    row2 << scat("reduce-", toperator::name()) << scat("tpool(x", tpool_t::size(), ")");
    for (size_t i = 0; i < single_deltas.size(); ++ i)
    {
        const auto deltaST = single_deltas[i];
        const auto valueST = single_values[i];
        const auto& targets = single_targets[i];
        const auto& outputs = single_outputs[i];

        scalar_t valueMT = 0;
        const auto deltaMT = measure<nanoseconds_t>([&] { valueMT = reduce_mt<toperator>(targets, outputs); }, 16);
        row2 << scat(std::setprecision(2), std::fixed, deltaST / static_cast<double>(deltaMT.count()));
        if (!close(valueST, valueMT, "tpool", epsilon1<scalar_t>())) { return false; }
    }

#ifdef _OPENMP
    // multi-threaded (using OpenMP)
    auto& row3 = table.append();
    row3 << scat("reduce-", toperator::name()) << "openmp";
    for (size_t i = 0; i < single_deltas.size(); ++ i)
    {
        const auto deltaST = single_deltas[i];
        const auto valueST = single_values[i];
        const auto& targets = single_targets[i];
        const auto& outputs = single_outputs[i];

        scalar_t valueMT = 0;
        const auto deltaMT = measure<nanoseconds_t>([&] { valueMT = reduce_op<toperator>(targets, outputs); }, 16);
        row3 << scat(std::setprecision(2), std::fixed, deltaST / static_cast<double>(deltaMT.count()));
        if (!close(valueST, valueMT, "openmp", epsilon1<scalar_t>())) { return false; }
    }
#endif

    // OK
    return true;
}

static int unsafe_main(int argc, const char *argv[])
{
    // parse the command line
    cmdline_t cmdline("benchmark thread pool");
    cmdline.add("", "min-size",     "minimum problem size (in kilo)", 1);
    cmdline.add("", "max-size",     "maximum problem size (in kilo)", 1024);

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto kilo = tensor_size_t(1024), mega = kilo * kilo, giga = mega * kilo;
    const auto cmd_min_size = std::clamp(kilo * cmdline.get<tensor_size_t>("min-size"), kilo, mega);
    const auto cmd_max_size = std::clamp(kilo * cmdline.get<tensor_size_t>("max-size"), cmd_min_size, giga);

    table_t table;
    auto& header = table.header();
    header << "problem" << "method";
    for (auto size = cmd_min_size; size <= cmd_max_size; size *= 2)
    {
        header << scat(size / kilo, "K");
    }
    table.delim();

    // benchmark for different problem sizes and processing chunk sizes
    if (!evaluate<exp_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }
    table.delim();
    if (!evaluate<log_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }
    table.delim();
    if (!evaluate<mse_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }

    // print results
    std::cout << table;

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
