#include <nano/core/table.h>
#include <nano/core/chrono.h>
#include <nano/core/logger.h>
#include <nano/core/cmdline.h>
#include <nano/linear/model.h>
#include <nano/linear/function.h>
#include <nano/dataset/imclass.h>

using namespace nano;

/*
static auto get_imclass(const string_t& id)
{
    const auto start = nano::timer_t{};

    auto dataset = imclass_dataset_t::all().get(id);
    critical(!dataset, scat("invalid dataset '", id, "'"));
    critical(!dataset->load(), scat("failed to load dataset '", id, "'"));

    log_info() << ">>> loading done in " << start.elapsed() << ".";
    return dataset;
}

static auto get_loss(const string_t& id)
{
    auto loss = loss_t::all().get(id);
    critical(!loss, scat("invalid loss '", id, "'"));
    return loss;
}

static auto get_solver(const string_t& id, const scalar_t epsilon, const int max_iterations)
{
    auto solver = solver_t::all().get(id);
    critical(!solver, scat("invalid solver '", id, "'"));
    solver->epsilon(epsilon);
    solver->max_iterations(max_iterations);
    solver->logger([&] (const solver_state_t& state)
    {
        std::cout << std::fixed << std::setprecision(6) << "\tdescent: " << state << ".\n";
        return true;
    });
    return solver;
}

static auto tune_batch(const dataset_t& dataset)
{
    const auto min_batch = 8;
    const auto max_batch = 1024;
    log_info() << "tuning the batch size per thread in the range [" << min_batch << ", " << max_batch << "]...";

    std::vector<scalar_t> batch_millis;
    table_t table;
    {
        auto& header = table.header();
        header << colspan(1) << "" << alignment::center
            << colspan(static_cast<int>(std::log2(max_batch) - std::log2(min_batch) + 1))
            << scat("batch size [ms/", max_batch, "samples]");
    }
    table.delim();
    {
        auto& header = table.header();
        header << "normalization";
        for (tensor_size_t batch = min_batch; batch <= max_batch; batch *= 2)
        {
            header << batch;
            batch_millis.push_back(0);
        }
    }

    const auto loss = get_loss("s-classnll");
    const auto samples = dataset.train_samples();
    auto function = linear_function_t{*loss, dataset, samples};

    const auto op_bench = [&] (row_t& row, const scalar_t l1reg, const scalar_t l2reg, const scalar_t vAreg, const auto& op)
    {
        for (tensor_size_t batch = min_batch, ibatch = 0; batch <= max_batch; batch *= 2, ++ ibatch)
        {
            function.l1reg(l1reg);
            function.l2reg(l2reg);
            function.vAreg(vAreg);
            function.batch(batch);

            const auto trials = size_t(10);
            const auto duration = ::nano::measure<microseconds_t>(op, trials);
            const auto millis = 1e-3 * static_cast<scalar_t>(duration.count()) / 2;
            batch_millis[ibatch] += millis;
            row << scat(std::fixed, std::setprecision(1), millis);
        }
    };

    const auto op_bench_value = [&] (row_t& row, const scalar_t l1reg, const scalar_t l2reg, const scalar_t vAreg)
    {
        volatile scalar_t value = 0;
        const vector_t x = vector_t::Random(function.size());
        op_bench(row, l1reg, l2reg, vAreg, [&] () { value = function.vgrad(x); });
    };

    const auto op_bench_vgrad = [&] (row_t& row, const scalar_t l1reg, const scalar_t l2reg, const scalar_t vAreg)
    {
        volatile scalar_t value = 0;
        vector_t gx(function.size());
        const vector_t x = vector_t::Random(function.size());
        op_bench(row, l1reg, l2reg, vAreg, [&] () { value = function.vgrad(x, &gx); });
    };

    table.delim();
    op_bench_value(table.append() << "none", 0e+0, 0e+0, 0e+0);
    op_bench_value(table.append() << "lasso", 1e+0, 0e+0, 0e+0);
    op_bench_value(table.append() << "ridge", 0e+0, 1e+0, 0e+0);
    op_bench_value(table.append() << "variance", 0e+0, 0e+0, 1e+0);

    table.delim();
    op_bench_vgrad(table.append() << "none", 0e+0, 0e+0, 0e+0);
    op_bench_vgrad(table.append() << "lasso", 1e+0, 0e+0, 0e+0);
    op_bench_vgrad(table.append() << "ridge", 0e+0, 1e+0, 0e+0);
    op_bench_vgrad(table.append() << "variance", 0e+0, 0e+0, 1e+0);

    table.delim();
    {
        auto& row = table.append() << "average";
        for (const auto& millis : batch_millis)
        {
            row << scat(std::fixed, std::setprecision(1), millis / 8.0);
        }
    }

    std::cout << table;

    const auto minimum = std::min_element(batch_millis.begin(), batch_millis.end());
    return min_batch * static_cast<tensor_size_t>(1U << static_cast<uint32_t>(std::distance(batch_millis.begin(), minimum)));
}
*/

static int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("report statistics on training linear models on image classification datasets");
    cmdline.add("", "imclass",          "regex to select image classification datasets", ".+");
    cmdline.add("", "solver",           "regex to select the solvers to benchmark", "lbfgs");
    cmdline.add("", "loss",             "regex to select the loss functions to benchmark", "s-classnll");
    cmdline.add("", "normalization",    "regex to select the feature scaling methods to benchmark", ".+");
    cmdline.add("", "regularization",   "regex to select the regularization methods to benchmark", ".+");
    cmdline.add("", "epsilon",          "convergence criterion (solver)", 1e-3);
    cmdline.add("", "max-iterations",   "maximum number of iterations (solver)", 1000);
    cmdline.add("", "tune-trials",      "maximum number of trials per tuning step of the regularization factor", 7);
    cmdline.add("", "tune-steps",       "number of tuning steps of the regularization factor", 2);
    cmdline.add("", "no-training",      "don't train the linear models (e.g. check dataset loading)");

    // todo: option to save trained models
    // todo: option to save training history to csv
    // todo: wrapper script to generate plots?!

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    /*
    const auto folds = cmdline.get<size_t>("folds");
    const auto epsilon = cmdline.get<scalar_t>("epsilon");
    const auto tune_steps = cmdline.get<int>("tune-steps");
    const auto tune_trials = cmdline.get<int>("tune-trials");
    const auto max_iterations = cmdline.get<int>("max-iterations");
    const auto train_percentage = cmdline.get<int>("train-percentage");
    const auto normalizations = enum_values<normalization>(std::regex(cmdline.get<string_t>("normalization")));
    const auto regularizations = enum_values<regularization>(std::regex(cmdline.get<string_t>("regularization")));

    // for each image classification dataset...
    for (const auto& id_imclass : imclass_dataset_t::all().ids(std::regex(cmdline.get<string_t>("imclass"))))
    {
        const auto dataset = get_imclass(id_imclass, folds, train_percentage);

        // tune the batch size wrt the processing time
        const auto batch = tune_batch(*dataset);
        log_info() << ">>> optimum batch size per thread is " << batch << " (samples).";

        if (cmdline.has("no-training"))
        {
            continue;
        }

        // train linear models
        table_t table;
        auto& header = table.append();
        header << "solver" << "loss" << "normalization" << "regularization" << "test error" << "eval[ms]" << "train[ms]";

        // for each numerical optimization method...
        for (const auto& id_solver : solver_t::all().ids(std::regex(cmdline.get<string_t>("solver"))))
        {
            const auto solver = get_solver(id_solver, epsilon, max_iterations);

            // for each loss function...
            for (const auto& id_loss : loss_t::all().ids(std::regex(cmdline.get<string_t>("loss"))))
            {
                const auto loss = get_loss(id_loss);

                table.delim();

                // for each feature scaling method...
                for (const auto normalization : normalizations)
                {
                    // for each regularization method
                    auto model = linear_model_t{};
                    for (const auto regularization : regularizations)
                    {
                        model.batch(batch);
                        model.tune_steps(tune_steps);
                        model.tune_trials(tune_trials);
                        model.normalization(normalization);
                        model.regularization(regularization);
                        const auto training = model.train(*loss, *dataset, *solver);

                        stats_t te_errors, eval_times, train_times;
                        for (const auto& tfold : training)
                        {
                            te_errors(tfold.te_error());
                            eval_times(0); // FIXME: tfold.m_eval_time.count());
                            train_times(0); // FIXME: tfold.m_train_time.count());
                        }

                        auto& row = table.append();
                        row << id_solver << id_loss << scat(normalization) << scat(regularization)
                            << scat(std::setprecision(2), std::fixed, te_errors.median())
                            << std::lround(eval_times.median())
                            << std::lround(train_times.median());
                    }
                }
            }
        }

        std::cout << table;
    }*/

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
