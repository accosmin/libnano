#include <iomanip>
#include <nano/core/table.h>
#include <nano/core/chrono.h>
#include <nano/core/logger.h>
#include <nano/core/cmdline.h>
#include <nano/core/tokenizer.h>
#include <nano/mlearn/enums.h>
#include <nano/gboost/model.h>
#include <nano/model/grid_search.h>
#include <nano/gboost/wlearner_table.h>
#include <nano/gboost/wlearner_stump.h>

using namespace nano;

template <typename tobject>
void info_factory(const string_t& name, const factory_t<tobject>& factory, const string_t& regex)
{
    table_t table;
    table.header() << name << "description";
    table.delim();
    for (const auto& id : factory.ids(std::regex(regex)))
    {
        table.append() << id << factory.description(id);
    }
    std::cout << table;
}

static void info_dataset(const rdataset_t& dataset, tensor_size_t max_features = 7)
{
    const auto op_append = [&] (table_t& table, const char* type, const feature_t& feature)
    {
        table.append() << type << feature.name()
            << (feature.discrete() ? scat("discrete x", feature.labels().size()) : "continuous")
            << (feature.optional() ? "optional" : "not optional");
    };

    table_t table;
    table.append() << "samples" << colspan(3) << dataset->samples();
    table.delim();
    if (dataset->features() > (max_features * 2))
    {
        for (tensor_size_t size = max_features, i = 0; i < size; ++ i)
        {
            op_append(table, "input", dataset->feature(i));
        }
        table.append() << "..." << "..." << "..." << "...";
        for (tensor_size_t size = dataset->features(), i = size - max_features; i < size; ++ i)
        {
            op_append(table, "input", dataset->feature(i));
        }
    }
    else
    {
        for (tensor_size_t size = dataset->features(), i = 0; i < size; ++ i)
        {
            op_append(table, "input", dataset->feature(i));
        }
    }
    table.delim();
    op_append(table, "target", dataset->target());
    std::cout << table;
}

static auto make_loss_id(const dataset_t& dataset, const cmdline_t& cmdline)
{
    string_t id;

    if (cmdline.has("loss"))
    {
        id = cmdline.get<string_t>("loss");
    }
    else
    {
        switch (dataset.type())
        {
        case task_type::unsupervised:
            critical(false, "unsupervised datasets are not supported!");
            break;

        case task_type::sclassification:
            id = "s-logistic";
            break;

        case task_type::mclassification:
            id = "m-logistic";
            break;

        default:
            id = "squared";
            break;
        }
        log_info() << "using loss <" << id << ">...";
    }

    return id;
}

static auto make_loss(const string_t& id)
{
    auto loss = loss_t::all().get(id);
    critical(!loss, scat("invalid loss '", id, "'"));
    return loss;
}

static auto make_dataset(const string_t& id)
{
    const auto start = nano::timer_t{};

    auto dataset = dataset_t::all().get(id);
    critical(!dataset, scat("invalid dataset '", id, "'"));
    dataset->load();

    log_info() << ">>> loading done in " << start.elapsed() << ".";
    return dataset;
}

static auto make_solver(const cmdline_t& cmdline)
{
    const auto id = cmdline.get<string_t>("solver");
    const auto epsilon = cmdline.get<scalar_t>("solver-epsilon");
    const auto max_iterations = cmdline.get<int>("solver-maxiter");

    auto solver = solver_t::all().get(id);
    critical(!solver, scat("invalid solver '", id, "'"));
    solver->epsilon(epsilon);
    solver->max_iterations(max_iterations);
    /*solver->logger([&] (const solver_state_t& state)
    {
        std::cout << std::fixed << std::setprecision(6) << "\tdescent: " << state << ".\n";
        return true;
    });*/
    return solver;
}

template <typename tscalar>
auto split(const string_t& config)
{
    std::vector<tscalar> values;
    for (auto tokenizer = tokenizer_t{config, ", \t\n\r"}; tokenizer; ++ tokenizer)
    {
        values.push_back(from_string<tscalar>(tokenizer.get()));
    }
    return values;
}

rmodel_t make_gridsearch(const gboost_model_t& model, const cmdline_t& cmdline, param_grid_t&& param_grid)
{
    auto gs = grid_search_model_t{model, param_grid};
    gs.folds(cmdline.get<int>("gridsearch-folds"));
    gs.max_trials(cmdline.get<int>("gridsearch-max-trials"));
    return gs.clone();
}

static auto make_boosters(const cmdline_t& cmdline)
{
    const auto wtable = wlearner_table_t{};
    const auto wstump = wlearner_stump_t{};

    auto model = gboost_model_t{};
    model.add(wtable);
    model.add(wstump);

    model.vAreg(0.0);
    model.shrinkage(1.0);
    model.subsample(1.00);
    model.batch(cmdline.get<int>("gboost-batch"));
    model.rounds(cmdline.get<int>("gboost-rounds"));
    model.epsilon(cmdline.get<scalar_t>("gboost-epsilon"));
    model.wscale(cmdline.get<::nano::wscale>("gboost-wscale"));

    const auto vAregs = split<scalar_t>(cmdline.get<string_t>("gboost-vAreg"));
    const auto shrinkages = split<scalar_t>(cmdline.get<string_t>("gboost-shrinkage"));
    const auto subsamples = split<scalar_t>(cmdline.get<string_t>("gboost-subsample"));

    std::map<string_t, rmodel_t> boosters;

    // no regularization
    boosters.emplace("default", model.clone());

    // tune shrinkage
    if (shrinkages.size() > 1U)
    {
        boosters.emplace("+shrinkage", make_gridsearch(model, cmdline, {{"gboost::shrinkage", shrinkages}}));
    }

    // tune subsampling
    if (subsamples.size() > 1U)
    {
        boosters.emplace("+subsample", make_gridsearch(model, cmdline, {{"gboost::subsample", subsamples}}));
    }

    // tune variance regularization
    if (vAregs.size() > 1U)
    {
        boosters.emplace("+vAreg", make_gridsearch(model, cmdline, {{"gboost::vAreg", vAregs}}));
    }

    return boosters;
}

static int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("report statistics regarding training gradient boosting models on various builtin datasets");
    cmdline.add("", "dataset",          "regex to select datasets, use --help-dataset to list the available options", "iris|wine");
    cmdline.add("", "loss",             "loss function, use --help-loss to list the available options, otherwise an appropriate default will be used");
    cmdline.add("", "solver",           "solver, use --help-solver to list the available options", "lbfgs");
    cmdline.add("", "solver-epsilon",   "solver: convergence criterion's threshold", 1e-6);
    cmdline.add("", "solver-maxiter",   "solver: maximum number of iterations", 100);
    cmdline.add("", "gboost-batch",     "gboost: number of samples to process at once", 32);
    cmdline.add("", "gboost-rounds",    "gboost: maximum number of boosting rounds", 100);
    cmdline.add("", "gboost-epsilon",   "gboost: error threshold to stop training at", 1e-6);
    cmdline.add("", "gboost-vAreg",     "gboost: comma-separated variance regularization factors [0,+inf)", "1e-3,1e-2,1e-1,1e+0");
    cmdline.add("", "gboost-shrinkage", "gboost: comma-separated shrinkage factors (0,1]", "0.1,0.2,0.5,0.9,1.0");
    cmdline.add("", "gboost-subsample", "gboost: comma-separated sub-sampling percentages (0,1]", "0.1,0.2,0.5,0.9,1.0");
    cmdline.add("", "gboost-wscale",    scat("gboost: weak learner scale [", enum_values<wscale>(), "]"), wscale::gboost);
    cmdline.add("", "importance",       scat("feature importance [", enum_values<importance>(), "]"), importance::shuffle);
    cmdline.add("", "folds",            "number of folds for k-fold evaluation (outer loop)", 10);
    cmdline.add("", "repetitions",      "number of repetitions for k-fold evaluation (outer loop)", 1);
    cmdline.add("", "gridsearch-folds", "number of folds for grid-search tuning (inner loop)", 10);
    cmdline.add("", "gridsearch-max-trials", "maximum number of trials for grid-search tuning (inner loop)", 100);
    cmdline.add("", "no-training",      "don't train the models (e.g. check dataset loading)");
    cmdline.add("", "show-config",      "display the parameter values for all the evaluated models");
    cmdline.add("", "help-loss",        "regex to select the builtin loss functions to display", ".+");
    cmdline.add("", "help-solver",      "regex to select the builtin solvers to display", ".+");
    cmdline.add("", "help-dataset",     "regex to select the builtin datasets to display", ".+");

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }
    if (cmdline.has("help-loss"))
    {
        info_factory("loss", loss_t::all(), cmdline.get<string_t>("help-loss"));
        return EXIT_SUCCESS;
    }
    if (cmdline.has("help-solver"))
    {
        info_factory("solver", solver_t::all(), cmdline.get<string_t>("help-solver"));
        return EXIT_SUCCESS;
    }
    if (cmdline.has("help-dataset"))
    {
        info_factory("dataset", dataset_t::all(), cmdline.get<string_t>("help-dataset"));
        return EXIT_SUCCESS;
    }

    table_t table;
    table.header() << "dataset" << "loss" << "model" << "time" << "train error" << "valid error";
    table.delim();

    const auto dataset_ids = dataset_t::all().ids(std::regex(cmdline.get<string_t>("dataset")));
    critical(dataset_ids.empty(), scat("invalid datasets '", cmdline.get<string_t>("dataset"), "'"));
    for (const auto& dataset_id : dataset_ids)
    {
        const auto dataset = make_dataset(dataset_id);
        info_dataset(dataset);

        if (cmdline.has("no-training"))
        {
            continue;
        }

        // TODO: vary feature importance
        // TODO: flavours to evaluate: wscale, w/o vAreg, w/o shrinkage, w/o subsample
        // TODO: report statistics: training+validation mean&variance, most important features, best hyper-parameters per fold, training time
        // TODO: t-test to rank models
        // TODO: find setup that returns stable results across various runs!!!
        // TODO: print report with best configs

        const auto folds = cmdline.get<int>("folds");
        const auto repetitions = cmdline.get<int>("repetitions");
        //const auto importance = ::nano::from_string<::nano::importance>(cmdline.get<string_t>("importance"));

        const auto loss_id = make_loss_id(*dataset, cmdline);

        const auto loss = make_loss(loss_id);
        const auto solver = make_solver(cmdline);

        for (const auto& [model_name, model] : make_boosters(cmdline))
        {
            const auto start = nano::timer_t{};
            const auto result = kfold(*model, *loss, *dataset, dataset->train_samples(), *solver, folds, repetitions);
            const auto elapsed = start.elapsed();

            stats_t train_stats(begin(result.m_train_errors), end(result.m_train_errors));
            stats_t valid_stats(begin(result.m_valid_errors), end(result.m_valid_errors));

            auto& row = table.append();
            row << dataset_id << loss_id << model_name << elapsed
                << scat(std::setprecision(4), std::fixed, train_stats)
                << scat(std::setprecision(4), std::fixed, valid_stats);

            if (cmdline.has("show-config"))
            {
                table.delim();
                for (size_t i = 0; i < result.m_models.size(); ++ i)
                {
                    auto& row = table.append();

                    row << nano::colspan(5) << result.m_models[i]->config()
                        << result.m_valid_errors(static_cast<tensor_size_t>(i));
                }
                table.delim();
            }

            // TODO: report median, stdev error
        }
    }

    std::cout << table << std::endl;

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
