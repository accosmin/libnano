#include <nano/tokenizer.h>
#include <nano/core/logger.h>
#include <nano/dataset/tabular.h>

using namespace nano;

tabular_dataset_t::tabular_dataset_t(csvs_t csvs, features_t features) :
    tabular_dataset_t(std::move(csvs), std::move(features), string_t::npos)
{
}

tabular_dataset_t::tabular_dataset_t(csvs_t csvs, features_t features, size_t target) :
    m_csvs(std::move(csvs)),
    m_features(std::move(features)),
    m_target(target)
{
}

void tabular_dataset_t::load()
{
    // check features
    critical(
        m_features.empty(),
        "tabular dataset: need to set at least one feature!");

    for (const auto& feature : m_features)
    {
        critical(
            feature.type() == feature_type::mclass,
            "tabular dataset: multi-label features are not supported (", feature, ")!");

        critical(
            ::nano::size(feature.dims()) > 1,
            "tabular dataset: structured features are not supported (", feature, ")!");
    }

    critical(
        m_target != string_t::npos &&
        m_target >= m_features.size(),
        "tabular dataset: the target feature index (", m_target,
        ") is not valid, expecting in the [0, ", m_features.size(), ") range!");

    // allocate storage
    tensor_size_t samples = 0;
    for (const auto& csv : m_csvs)
    {
        csv.parse([&] (const string_t&, const tensor_size_t)
        {
            ++ samples;
            return true;
        });
    }

    critical(
        samples == 0,
        "tabular dataset: no data to read, check paths!");

    resize(samples, m_features, m_target);

    // load data
    tensor_size_t sample = 0;
    for (const auto& csv : m_csvs)
    {
        log_info() << "tabular dataset: reading " << csv.m_path << "...";

        const auto old_sample = sample;
        csv.parse([&] (const string_t& line, tensor_size_t line_index)
        {
            this->parse(csv, line, line_index, sample ++);
            return true;
        });

        const auto samples_read = sample - old_sample;
        critical(
            csv.m_expected > 0 && samples_read != csv.m_expected,
            "tabular dataset: read ", samples_read, ", expecting ", csv.m_expected, " samples!");

        dataset_t::testing(make_range(
            old_sample + csv.m_testing.begin(),
            old_sample + csv.m_testing.end()));

        log_info() << "tabular dataset: read " << sample << " samples!";
    }

    critical(
        sample != samples,
        "tabular dataset: read ", sample, " samples, expecting ", samples, "!");
}

void tabular_dataset_t::parse(const csv_t& csv, const string_t& line, tensor_size_t line_index, tensor_size_t sample)
{
    critical(
        sample >= samples(),
        "tabular dataset: too many samples, expecting ", samples(),  "!");

    for (auto tokenizer = tokenizer_t{line, csv.m_delim.c_str()}; tokenizer; ++ tokenizer)
    {
        critical(
            tokenizer.count() > m_features.size(),
            "tabular dataset: invalid line [", line, "]@", csv.m_path, ":", line_index,
            ", got ", tokenizer.count(), " tokens, expecting ", m_features.size(), "!");

        const auto f = tokenizer.count() - 1;
        const auto token = tokenizer.get();
        const auto& feature = m_features[f];

        if (token != csv.m_placeholder)
        {
            try
            {
                set(sample, static_cast<tensor_size_t>(f), token);
            }
            catch (std::exception& e)
            {
                critical0(
                    "tabular dataset: invalid line [", line, "]@", csv.m_path, ":", line_index,
                    ", invalid token [", token, "] for feature (", feature, ")!");
            }
        }
    }
}
