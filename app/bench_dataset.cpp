#include <nano/dataset.h>
#include <nano/core/chrono.h>
#include <nano/core/logger.h>
#include <nano/core/cmdline.h>
#include <nano/generator/elemwise_gradient.h>
#include <nano/generator/elemwise_identity.h>
#include <nano/generator/elemwise_histogram.h>
#include <nano/generator/pairwise_scalar2scalar.h>
#include <nano/generator/pairwise_scalar2sclass.h>

using namespace nano;

static auto make_identity(const dataset_t& dataset)
{
    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<sclass_identity_t>>();
    generator.add<elemwise_generator_t<mclass_identity_t>>();
    generator.add<elemwise_generator_t<scalar_identity_t>>();
    generator.add<elemwise_generator_t<struct_identity_t>>();
    return generator;
}

static auto make_gradient(const dataset_t& dataset)
{
    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<elemwise_gradient_t>>();
    return generator;
}

static auto make_product(const dataset_t& dataset, struct2scalar s2s = struct2scalar::on)
{
    auto generator = dataset_generator_t{dataset};
    generator.add<pairwise_generator_t<pairwise_scalar2scalar_t<product_t>>>(s2s);
    return generator;
}

/*static auto make_product_sign_class(const dataset_t& dataset, struct2scalar s2s = struct2scalar::on)
{
    auto generator = dataset_generator_t{dataset};
    generator.add<pairwise_generator_t<pairwise_scalar2sclass_t<product_sign_class_t>>>(s2s);
    return generator;
}*/

static auto make_ratio_histogram(const dataset_t& dataset,
    struct2scalar s2s = struct2scalar::on, tensor_size_t bins = 10)
{
    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<ratio_histogram_medians_t>>(s2s, bins);
    return generator;
}

static auto make_percentile_histogram(const dataset_t& dataset,
    struct2scalar s2s = struct2scalar::on, tensor_size_t bins = 10)
{
    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<percentile_histogram_medians_t>>(s2s, bins);
    return generator;
}

static auto benchmark(const string_t& generator_id, dataset_generator_t&& generator)
{
    const auto samples = arange(0, generator.dataset().samples());

    auto timer = ::nano::timer_t{};

    timer.reset();
    generator.fit(samples, execution::par);
    log_info() << "generator [" << generator_id << "] fitted in <" << timer.elapsed() << ">.";

    const auto batch = tensor_size_t{128};

    timer.reset();
    tensor2d_t flatten;
    for (tensor_size_t begin = 0; begin < samples.size(); )
    {
        const auto end = std::min(begin + batch, samples.size());
        generator.flatten(samples.slice(begin, end), flatten);
        (void)flatten;
        begin = end;
    }
    log_info() << "generator [" << generator_id << "] flatten in <" << timer.elapsed() << ">.";

    timer.reset();
    const auto flatten_stats = generator.flatten_stats(samples, execution::par, batch);
    (void)flatten_stats;
    log_info() << "generator [" << generator_id << "] flatten statistics in <" << timer.elapsed() << ">.";

    timer.reset();
    tensor4d_t targets;
    for (tensor_size_t begin = 0; begin < samples.size(); )
    {
        const auto end = std::min(begin + batch, samples.size());
        generator.targets(samples.slice(begin, end), targets);
        (void)targets;
        begin = end;
    }
    log_info() << "generator [" << generator_id << "] targets in <" << timer.elapsed() << ">.";

    timer.reset();
    const auto targets_stats = generator.targets_stats(samples, execution::par, batch);
    (void)targets_stats;
    log_info() << "generator [" << generator_id << "] targets statistics in <" << timer.elapsed() << ">.";

    timer.reset();
    const auto select_stats = generator.select_stats(execution::par);
    (void)select_stats;
    log_info() << "generator [" << generator_id << "] select statistics in <" << timer.elapsed() << ">.";

    // TODO: generic utility to loop through features by type or through flatten features by batch
}

static auto benchmark(
    const string_t& dataset_id,
    bool gen_identity, bool gen_gradient, bool gen_product, bool gen_ratio_histogram, bool gen_percentile_histogram,
    struct2scalar s2s, tensor_size_t bins)
{
    const auto rdataset = dataset_t::all().get(dataset_id);
    auto& dataset = *rdataset;

    const auto timer = ::nano::timer_t{};
    dataset.load();
    const auto elapsed = timer.elapsed();
    log_info() << string_t(80, '=');
    log_info() << "dataset [" << dataset_id << "] loaded in <" << elapsed << ">.";
    log_info() << "  type=" << dataset.type();
    log_info() << "  samples=" << dataset.samples();
    log_info() << "  features=" << dataset.features();
    log_info() << string_t(80, '=');

    if (gen_identity)
    {
        benchmark("identity", make_identity(dataset));
    }
    if (gen_gradient)
    {
        benchmark("gradient", make_gradient(dataset));
    }
    if (gen_product)
    {
        benchmark("product", make_product(dataset, s2s));
    }
    //benchmark("product sign class", make_product_sign_class(dataset));
    if (gen_ratio_histogram)
    {
        benchmark("ratio histogram", make_ratio_histogram(dataset, s2s, bins));
    }
    if (gen_percentile_histogram)
    {
        benchmark("percentile histogram", make_percentile_histogram(dataset, s2s, bins));
    }
}

static int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark loading datasets and generating features");
    cmdline.add("", "dataset",          "regex to select the datasets to benchmark", ".+");
    cmdline.add("", "gen-identity",     "enable feature generation: identity transformation");
    cmdline.add("", "gen-gradient",     "enable feature generation: 2D gradient-like transformations");
    cmdline.add("", "gen-product",      "enable feature generation: pairwise product");
    cmdline.add("", "gen-ratio-hist",   "enable feature generation: ratio-based histograms");
    cmdline.add("", "gen-perc-hist",    "enable feature generation: percentile-based histogram");
    cmdline.add("", "struct2scalar",    "use structured features as scalar features element wise");
    cmdline.add("", "bins",             "number of bins to use for the histogram-based features", 10);

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto dregex = std::regex(cmdline.get<string_t>("dataset"));

    const auto gen_product = cmdline.has("gen-product");
    const auto gen_identity = cmdline.has("gen-identity");
    const auto gen_gradient = cmdline.has("gen-gradient");
    const auto gen_ratio_histogram = cmdline.has("gen-ratio-hist");
    const auto gen_percentile_histogram = cmdline.has("gen-perc-hist");
    const auto s2s = cmdline.has("struct2scalar") ? struct2scalar::on : struct2scalar::off;
    const auto bins = cmdline.get<tensor_size_t>("bins");

    // benchmark
    for (const auto& id : dataset_t::all().ids(dregex))
    {
        benchmark(
            id,
            gen_identity, gen_gradient, gen_product, gen_ratio_histogram, gen_percentile_histogram,
            s2s, bins);
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
