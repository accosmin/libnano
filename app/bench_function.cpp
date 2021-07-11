#include <iomanip>
#include <nano/function.h>
#include <nano/core/table.h>
#include <nano/core/stats.h>
#include <nano/core/chrono.h>
#include <nano/core/logger.h>
#include <nano/core/cmdline.h>

using namespace nano;

static void eval_func(const function_t& function, table_t& table)
{
    const auto dims = function.size();
    const vector_t x = vector_t::Zero(dims);
    vector_t g = vector_t::Zero(dims);

    const size_t trials = 16;

    volatile scalar_t fx = 0;
    const auto fval_time = measure<nanoseconds_t>([&] ()
    {
        fx += function.vgrad(x);
    }, trials).count();

    volatile scalar_t gx = 0;
    const auto grad_time = measure<nanoseconds_t>([&] ()
    {
        function.vgrad(x, &g);
        gx += g.template lpNorm<Eigen::Infinity>();
    }, trials).count();

    scalar_t grad_accuracy = 0;
    for (size_t i = 0; i < trials; ++ i)
    {
        grad_accuracy += function.grad_accuracy(vector_t::Random(dims));
    }

    auto& row = table.append();
    row << function.name() << fval_time << grad_time
        << scat(std::setprecision(12), std::fixed, grad_accuracy / static_cast<scalar_t>(trials));
}

static int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("benchmark optimization test functions");
    cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "1024");
    cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "1024");
    cmdline.add("", "functions",    "use this regex to select the functions to benchmark", ".+");

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
    const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
    const auto functions = std::regex(cmdline.get<string_t>("functions"));

    table_t table;
    table.header() << "function" << "f(x)[ns]" << "f(x,g)[ns]" << "grad accuracy";
    table.delim();

    tensor_size_t prev_size = min_dims;
    for (const auto& function : get_functions(min_dims, max_dims, convexity::unknown, functions))
    {
        if (function->size() != prev_size)
        {
            table.delim();
            prev_size = function->size();
        }
        eval_func(*function, table);
    }

    std::cout << table;

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
