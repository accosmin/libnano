#include <iostream>
#include <nano/table.h>
#include <nano/logger.h>
#include <nano/cmdline.h>
#include <nano/solver.h>
#include <nano/version.h>

using namespace nano;

namespace
{
    template <typename tobject>
    void print(const string_t& name, const factory_t<tobject>& factory)
    {
        table_t table;
        table.header() << name << "description" << "configuration";
        table.delim();
        for (const auto& id : factory.ids())
        {
            const auto json = factory.get(id)->config();
            table.append() << id << factory.description(id) << json.dump();
        }
        std::cout << table;
    }
}

static int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("display the registered objects");
    cmdline.add("", "lsearch-init",         "methods to estimate the initial line-search step length");
    cmdline.add("", "lsearch-strategy",     "line-search methods");
    cmdline.add("", "solver",               "numerical optimization methods");
    cmdline.add("", "version",              "library version");
    cmdline.add("", "git-hash",             "git commit hash");
    cmdline.add("", "system",               "system: all available information");
    cmdline.add("", "sys-logical-cpus",     "system: number of logical cpus");
    cmdline.add("", "sys-physical-cpus",    "system: number of physical cpus");
    cmdline.add("", "sys-memsize",          "system: memory size in GB");

    cmdline.process(argc, argv);

    const auto has_lsearch_init = cmdline.has("lsearch-init");
    const auto has_lsearch_strategy = cmdline.has("lsearch-strategy");
    const auto has_solver = cmdline.has("solver");
    const auto has_system = cmdline.has("system");
    const auto has_sys_logical = cmdline.has("sys-logical-cpus");
    const auto has_sys_physical = cmdline.has("sys-physical-cpus");
    const auto has_sys_memsize = cmdline.has("sys-memsize");
    const auto has_version = cmdline.has("version");
    const auto has_git_hash = cmdline.has("git-hash");

    if (!has_lsearch_init &&
        !has_lsearch_strategy &&
        !has_solver &&
        !has_system &&
        !has_sys_logical &&
        !has_sys_physical &&
        !has_sys_memsize &&
        !has_version &&
        !has_git_hash)
    {
        cmdline.usage();
        return EXIT_FAILURE;
    }

    // check arguments and options
    if (has_lsearch_init)
    {
        print("lsearch-init", lsearch_init_t::all());
    }
    if (has_lsearch_strategy)
    {
        print("lsearch-strategy", lsearch_strategy_t::all());
    }
    if (has_solver)
    {
        print("solver", solver_t::all());
    }
    if (has_system || has_sys_physical)
    {
        std::cout << "physical CPUs..." << nano::physical_cpus() << std::endl;
    }
    if (has_system || has_sys_logical)
    {
        std::cout << "logical CPUs...." << nano::logical_cpus() << std::endl;
    }
    if (has_system || has_sys_memsize)
    {
        std::cout << "memsize........." << nano::memsize_gb() << "GB" << std::endl;
    }
    if (has_version)
    {
        std::cout << nano::major_version << "." << nano::minor_version << std::endl;
    }
    if (has_git_hash)
    {
        std::cout << nano::git_commit_hash << std::endl;
    }

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
