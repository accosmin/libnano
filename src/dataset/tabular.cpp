#include <nano/logger.h>
#include <nano/tokenizer.h>
#include <nano/mlearn/class.h>
#include <nano/dataset/tabular.h>

using namespace nano;

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

    critical(
        m_target != string_t::npos &&
        m_target >= m_features.size(),
        scat("tabular dataset: the target feature (", m_target,
             ") is not valid, expecting in the [0, ", m_features.size(), ") range!"));

    critical(
        m_target < m_features.size() && m_features[m_target].optional(),
        scat("tabular dataset: the target feature (", m_target, ") cannot be optional!"));

    // allocate storage
    tensor_size_t data_size = 0;
    for (const auto& csv : m_csvs)
    {
        csv.parse([&] (const string_t&, const tensor_size_t)
        {
            ++ data_size;
            return true;
        });
    }

    tensor_size_t n_inputs = 0, n_targets = 0;
    for (size_t f = 0; f < m_features.size(); ++ f)
    {
        const auto& feature = m_features[f];
        if (f == m_target)
        {
            n_targets += feature.discrete() ?
                static_cast<tensor_size_t>(feature.labels().size()):
                tensor_size_t(1);
        }
        else
        {
            n_inputs ++;
        }
    }

    critical(
        data_size == 0,
        "tabular dataset: no data to read, check paths!");

    resize(make_dims(data_size, n_inputs, 1, 1), make_dims(data_size, n_targets, 1, 1));

    // load data
    tensor_size_t row = 0;
    for (const auto& csv : m_csvs)
    {
        log_info() << "tabular dataset: reading " << csv.m_path << "...";

        const auto old_row = row;
        critical(
            !csv.parse([&] (const string_t& line, const tensor_size_t line_index)
            {
                return this->parse(csv, line, line_index, row ++);
            }),
            "failed to read file!");

        const auto samples_read = row - old_row;
        critical(
            csv.m_expected > 0 && samples_read != csv.m_expected,
            scat("tabular dataset: read ", samples_read, ", expecting ", csv.m_expected, " samples!"));

        dataset_t::testing(make_range(
            old_row + csv.m_testing.begin(),
            old_row + csv.m_testing.end()));

        log_info() << "tabular dataset: read " << row << " samples!";
    }

    critical(
        row != data_size,
        scat("tabular dataset: read ", row, " samples, expecting ", data_size, "!"));
}

void tabular_dataset_t::store(const tensor_size_t row, const size_t col, const scalar_t value)
{
    if (col != m_target)
    {
        input(row)(static_cast<tensor_size_t>((col > m_target) ? (col - 1) : col), 0, 0) = value;
    }
    else
    {
        assert(m_target < m_features.size());
        assert(all_targets().dims() == make_dims(all_targets().size<0>(), 1, 1, 1));

        target(row)(0, 0, 0) = value;
    }
}

void tabular_dataset_t::store(const tensor_size_t row, const size_t col, const tensor_size_t category)
{
    if (col != m_target)
    {
        input(row)(static_cast<tensor_size_t>((col > m_target) ? (col - 1) : col), 0, 0) = static_cast<scalar_t>(category);
    }
    else
    {
        const auto& feature = m_features[col];
        const auto labels_size = static_cast<tensor_size_t>(feature.labels().size());

        assert(feature.discrete());
        assert(category < labels_size);
        assert(m_target < m_features.size());
        assert(all_targets().dims() == make_dims(all_targets().size<0>(), labels_size, 1, 1));

        target(row) = class_target(labels_size, category);
    }
}

bool tabular_dataset_t::parse(const csv_t& csv, const string_t& line, tensor_size_t line_index, tensor_size_t row)
{
    if (row >= all_inputs().size<0>())
    {
        log_error() << "tabular dataset: too many samples, expecting " << all_inputs().size<0>() << "!";
        return false;
    }

    for (auto tokenizer = tokenizer_t{line, csv.m_delim.c_str()}; tokenizer; ++ tokenizer)
    {
        if (tokenizer.count() > m_features.size())
        {
            log_error() << "tabular dataset: invalid line " << csv.m_path << ":" << line_index
                << ", expecting " << m_features.size() << " tokens!";
            return false;
        }

        const auto f = tokenizer.count() - 1;
        const auto token = tokenizer.get();
        auto& feature = m_features[f];

        if (feature.optional() && token == csv.m_placeholder)
        {
            assert(f != m_target);
            store(row, f, feature_t::placeholder_value());
        }
        else if (!feature.discrete())
        {
            try
            {
                store(row, f, from_string<scalar_t>(token));
            }
            catch (std::exception& e)
            {
                log_error() << "tabular dataset: invalid line " << csv.m_path << ":" << line_index
                    << ", expecting arithmetic token [" << token << "] for feature [" << feature.name() << "]!";
                return false;
            }
        }
        else
        {
            const auto ilabel = feature.set_label(token);
            if (ilabel == string_t::npos)
            {
                log_error() << "tabular dataset: invalid line " << csv.m_path << ":" << line_index
                    << ", invalid label [" << token << "] for feature [" << feature.name() << "]!";
                return false;
            }

            store(row, f, static_cast<tensor_size_t>(ilabel));
        }
    }

    return true;
}

feature_t tabular_dataset_t::feature(tensor_size_t index) const
{
    auto findex = static_cast<size_t>(index);
    if (findex >= m_target)
    {
        ++ findex;
    }
    return m_features.at(findex);
}

feature_t tabular_dataset_t::target() const
{
    return (m_target < m_features.size()) ? m_features[m_target] : feature_t{};
}
