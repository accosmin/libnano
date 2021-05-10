#include <iomanip>
#include <nano/table.h>
#include <nano/tpool.h>
#include <nano/logger.h>
#include <nano/solver.h>
#include <nano/cmdline.h>
#include <nano/core/stats.h>
#include <nano/core/chrono.h>
#include <nano/core/numeric.h>

using namespace nano;

struct solver_stat_t
{
    void update(const solver_state_t& state)
    {
        m_crits(state.convergence_criterion());
        m_fails(state.m_status != solver_state_t::status::converged ? 1 : 0);
        m_iters(static_cast<scalar_t>(state.m_iterations));
        m_errors(state.m_status == solver_state_t::status::failed ? 1 : 0);
        m_maxits(state.m_status == solver_state_t::status::max_iters ? 1 : 0);
        m_fcalls(static_cast<scalar_t>(state.m_fcalls));
        m_gcalls(static_cast<scalar_t>(state.m_gcalls));
        m_costs(static_cast<scalar_t>(state.m_fcalls + 2 * state.m_gcalls));
    }

    stats_t     m_crits;            ///< convergence criterion
    stats_t     m_fails;            ///< #convergence failures
    stats_t     m_iters;            ///< #optimization iterations
    stats_t     m_errors;           ///< #internal errors (e.g. line-search failed)
    stats_t     m_maxits;           ///< #maximum iterations reached
    stats_t     m_fcalls;           ///< #function value calls
    stats_t     m_gcalls;           ///< #gradient calls
    stats_t     m_costs;            ///< computation cost as a function of function value and gradient calls
    int64_t     m_milliseconds{0};  ///< total number of milliseconds
};

using solver_config_stats_t = std::map<
    std::tuple<string_t, string_t, string_t>,
    solver_stat_t>;

static void show_table(const string_t& table_name, const solver_config_stats_t& stats)
{
    assert(!stats.empty());

    // show global statistics
    table_t table;
    table.header()
        << table_name
        << "lsearch0"
        << "lsearchk"
        << "gnorm"
        << "#fails"
        << "#iters"
        << "#errors"
        << "#maxits"
        << "#fcalls"
        << "#gcalls"
        << "cost"
        << "[ms]";
    table.delim();

    for (const auto& it : stats)
    {
        const auto& stat = it.second;

        if (stat.m_fcalls)
        {
            table.append()
            << std::get<0>(it.first) << std::get<1>(it.first) << std::get<2>(it.first)
            << stat.m_crits.avg()
            << static_cast<size_t>(stat.m_fails.sum1())
            << static_cast<size_t>(stat.m_iters.avg())
            << static_cast<size_t>(stat.m_errors.sum1())
            << static_cast<size_t>(stat.m_maxits.sum1())
            << static_cast<size_t>(stat.m_fcalls.avg())
            << static_cast<size_t>(stat.m_gcalls.avg())
            << static_cast<size_t>(stat.m_costs.avg())
            << stat.m_milliseconds;
        }
    }

    table.sort(nano::make_less_from_string<scalar_t>(), {4, 10});
    std::cout << table;
}

static auto log_solver(const function_t& function, const rsolver_t& solver, const string_t& solver_id,
    const vector_t& x0)
{
    std::cout << std::fixed << std::setprecision(10);
    std::cout << function.name()
        << " solver[" << solver_id
        << "],lsearch0[" << solver->lsearch0_id()
        << "],lsearchk[" << solver->lsearchk_id() << "]" << std::endl;

    solver->logger([&] (const solver_state_t& state)
    {
        std::cout
            << "descent: i=" << state.m_iterations << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << "[" << state.m_status << "]" << ",calls=" << state.m_fcalls << "/" << state.m_gcalls << "." << std::endl;
        return true;
    });

    solver->lsearch0_logger([&] (const solver_state_t& state0, const scalar_t t)
    {
        std::cout
            << "\tlsearch(0): t=" << state0.t << ",f=" << state0.f << ",g=" << state0.convergence_criterion()
            << ",t=" << t << "." << std::endl;
    });

    solver->lsearchk_logger([&] (const solver_state_t& state0, const solver_state_t& state)
    {
        std::cout
            << "\tlsearch(t): t=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << ",armijo=" << state.has_armijo(state0, solver->c1())
            << ",wolfe=" << state.has_wolfe(state0, solver->c2())
            << ",swolfe=" << state.has_strong_wolfe(state0, solver->c2()) << "." << std::endl;
    });

    auto state = solver->minimize(function, x0);
    std::cout << std::flush;

    // NB: need to reset the logger for the next batch of tests!
    solver->logger({});
    solver->lsearch0_logger({});
    solver->lsearchk_logger({});

    return state;
}

static void check_solver(const function_t& function, const rsolver_t& solver,
    const string_t& solver_id, const std::vector<vector_t>& x0s,
    solver_config_stats_t& fstats, solver_config_stats_t& gstats,
    const bool log_failures)
{
    const auto timer = nano::timer_t{};

    std::vector<solver_state_t> states(x0s.size());
    loopi(x0s.size(), [&] (const size_t i, const size_t)
    {
        states[i] = solver->minimize(function, x0s[i]);
    });

    const auto milliseconds = timer.milliseconds().count();

    for (size_t i = 0; i < x0s.size(); ++ i)
    {
        // log in full detail the optimization trajectory if it fails
        if (log_failures &&
            states[i].m_status != solver_state_t::status::max_iters &&
            states[i].m_status != solver_state_t::status::converged)
        {
            const auto state = log_solver(function, solver, solver_id, x0s[i]);
            assert(state.m_status == states[i].m_status);
        }
    }

    const auto key = std::make_tuple(solver_id, solver->lsearch0_id(), solver->lsearchk_id());
    auto& fstat = fstats[key];
    auto& gstat = gstats[key];

    for (const auto& state : states)
    {
        fstat.update(state);
        gstat.update(state);
    }
    fstat.m_milliseconds += milliseconds;
    gstat.m_milliseconds += milliseconds;
}

static void check_function(const function_t& function,
    const std::vector<std::pair<string_t, rsolver_t>>& id_solvers,
    const size_t trials, solver_config_stats_t& gstats, const bool log_failures)
{
    // generate fixed random trials
    std::vector<vector_t> x0s(trials);
    std::generate(x0s.begin(), x0s.end(), [&] () { return vector_t::Random(function.size()); });

    // per-problem statistics
    solver_config_stats_t fstats;

    // evaluate all possible combinations (solver & line-search)
    for (const auto& id_solver : id_solvers)
    {
        const auto& solver_id = id_solver.first;
        const auto& solver = id_solver.second;

        check_solver(function, solver, solver_id, x0s, fstats, gstats, log_failures);
    }

    // show per-problem statistics
    show_table(align(function.name(), 36), fstats);
}

static int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark solvers");
    cmdline.add("", "solver",           "regex to select the line-search solvers to benchmark", ".+");
    cmdline.add("", "function",         "regex to select the functions to benchmark", ".+");
    cmdline.add("", "min-dims",         "minimum number of dimensions for each test function (if feasible)", "100");
    cmdline.add("", "max-dims",         "maximum number of dimensions for each test function (if feasible)", "1000");
    cmdline.add("", "trials",           "number of random trials for each test function", "100");
    cmdline.add("", "max-iterations",   "maximum number of iterations", "1000");
    cmdline.add("", "epsilon",          "convergence criterion", 1e-6);
    cmdline.add("", "convex",           "use only convex test functions");
    cmdline.add("", "c1",               "use this c1 value (see Armijo-Goldstein line-search step condition)");
    cmdline.add("", "c2",               "use this c2 value (see Wolfe line-search step condition)");
    cmdline.add("", "lsearch0",         "use this regex to select the line-search initialization methods");
    cmdline.add("", "lsearchk",         "use this regex to select the line-search strategies");
    cmdline.add("", "log-failures",     "log the optimization trajectory for the runs that fail");

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
    const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
    const auto trials = cmdline.get<size_t>("trials");
    const auto max_iterations = cmdline.get<int>("max-iterations");
    const auto epsilon = cmdline.get<scalar_t>("epsilon");
    const auto convex = cmdline.has("convex") ? convexity::yes : convexity::unknown;
    const auto log_failures = cmdline.has("log-failures");

    const auto fregex = std::regex(cmdline.get<string_t>("function"));
    const auto sregex = std::regex(cmdline.get<string_t>("solver"));

    const auto lsearch0s = cmdline.has("lsearch0") ?
        lsearch0_t::all().ids(std::regex(cmdline.get<string_t>("lsearch0"))) :
        strings_t{""};

    const auto lsearchks = cmdline.has("lsearchk") ?
        lsearchk_t::all().ids(std::regex(cmdline.get<string_t>("lsearchk"))) :
        strings_t{""};

    // construct the list of solver configurations to evaluate
    std::vector<std::pair<string_t, rsolver_t>> solvers;
    const auto add_solver = [&] (const string_t& solver_id, const string_t& lsearch0, const string_t& lsearchk)
    {
        auto solver = solver_t::all().get(solver_id);

        solver->epsilon(epsilon);
        solver->max_iterations(max_iterations);
        if (!lsearch0.empty()) { solver->lsearch0(lsearch0); }
        if (!lsearchk.empty()) { solver->lsearchk(lsearchk); }
        solver->tolerance(
            cmdline.has("c1") ? cmdline.get<scalar_t>("c1") : solver->c1(),
            cmdline.has("c2") ? cmdline.get<scalar_t>("c2") : solver->c2());

        solvers.emplace_back(solver_id, std::move(solver));
    };

    for (const auto& id : solver_t::all().ids(sregex))
    {
        for (const auto& lsearch0 : lsearch0s)
        {
            for (const auto& lsearchk : lsearchks)
            {
                add_solver(id, lsearch0, lsearchk);
            }
        }
    }

    // benchmark
    solver_config_stats_t gstats;
    for (const auto& function : get_functions(min_dims, max_dims, convex, fregex))
    {
        check_function(*function, solvers, trials, gstats, log_failures);
    }

    show_table(align("solver", 36), gstats);

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
