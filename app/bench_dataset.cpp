#include <nano/logger.h>
#include <nano/cmdline.h>
#include <nano/dataset.h>

using namespace nano;

static auto benchmark(const string_t& dataset_id)
{
    const auto dataset = dataset_t::all().get(dataset_id);
    dataset->load();

}

static int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark loading datasets and generating features");
    cmdline.add("", "dataset",          "regex to select the datasets to benchmark", ".+");

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto dregex = std::regex(cmdline.get<string_t>("dataset"));

    // benchmark
    for (const auto& id : dataset_t::all().ids(dregex))
    {
        benchmark(id);
    }

    // TODO: benchmark loading
    // TODO: benchmark feature generation (identity, histogram, gradient, quadratic)

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
