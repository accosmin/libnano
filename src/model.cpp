#include <mutex>
#include <nano/mlearn/kfold.h>
#include <nano/gboost/model.h>
#include <nano/linear/model.h>
#include <nano/model/grid_search.h>

using namespace nano;

namespace nano
{
    template <typename tscalar>
    std::ostream& write(std::ostream& stream, const param1_t<tscalar>& param)
    {
        if (::nano::write(stream, param.name()) &&
            ::nano::write(stream, param.get()) &&
            ::nano::write(stream, param.min()) &&
            ::nano::write(stream, param.max()) &&
            ::nano::write(stream, static_cast<uint32_t>(param.minLE())) &&
            ::nano::write(stream, static_cast<uint32_t>(param.maxLE())))
        {
        }
        return stream;
    }

    std::ostream& write(std::ostream& stream, const eparam1_t& param)
    {
        if (::nano::write(stream, param.name()) &&
            ::nano::write(stream, param.get()))
        {
        }
        return stream;
    }

    template <typename tscalar>
    std::istream& read(std::istream& stream, param1_t<tscalar>& param)
    {
        string_t name;
        tscalar value, min, max;
        uint32_t minLE = 0U, maxLE = 0U;

        if (::nano::read(stream, name) &&
            ::nano::read(stream, value) &&
            ::nano::read(stream, min) &&
            ::nano::read(stream, max) &&
            ::nano::read(stream, minLE) &&
            ::nano::read(stream, maxLE))
        {
            param = param1_t<tscalar>{
                name, min, (minLE != 0U) ? LEorLT{LE} : LEorLT{LT},
                value, (maxLE != 0U) ? LEorLT{LE} : LEorLT{LT}, max};
        }
        return stream;
    }

    std::istream& read(std::istream& stream, eparam1_t& param)
    {
        string_t name;
        int64_t value = 0;

        if (::nano::read(stream, name) &&
            ::nano::read(stream, value))
        {
            param = eparam1_t{name, value};
        }
        return stream;
    }
}

model_param_t::model_param_t(eparam1_t param) :
    m_storage(std::move(param))
{
}

model_param_t::model_param_t(iparam1_t param) :
    m_storage(std::move(param))
{
}

model_param_t::model_param_t(sparam1_t param) :
    m_storage(std::move(param))
{
}

void model_param_t::set(int32_t value)
{
    set(static_cast<int64_t>(value));
}

void model_param_t::set(int64_t value)
{
    if (is_ivalue())
    {
        iparam().set(value);
    }
    else if (is_svalue())
    {
        sparam().set(static_cast<scalar_t>(value));
    }
    else
    {
        critical0("model parameter (", name(), "): cannot set enumeration with integer (", value, ")!");
    }
}

void model_param_t::set(scalar_t value)
{
    if (is_svalue())
    {
        sparam().set(value);
    }
    else
    {
        critical0("model parameter (", name(), "): cannot set not-scalar with scalar (", value, ")!");
    }
}

int64_t model_param_t::ivalue() const
{
    return iparam().get();
}

scalar_t model_param_t::svalue() const
{
    return sparam().get();
}

bool model_param_t::is_evalue() const
{
    return std::holds_alternative<eparam1_t>(m_storage);
}

bool model_param_t::is_ivalue() const
{
    return std::holds_alternative<iparam1_t>(m_storage);
}

bool model_param_t::is_svalue() const
{
    return std::holds_alternative<sparam1_t>(m_storage);
}

const string_t& model_param_t::name() const
{
    if (is_evalue())
    {
        return eparam().name();
    }
    else if (is_ivalue())
    {
        return iparam().name();
    }
    else
    {
        return sparam().name();
    }
}

void model_param_t::read(std::istream& stream)
{
    int32_t type = -1;
    critical(
        !::nano::read(stream, type),
        "model parameter: failed to read from stream!");

    switch (type)
    {
    case 0:
        {
            eparam1_t param;
            critical(
                !::nano::read(stream, param),
                "model parameter: failed to read from stream!");
            m_storage = param;
        }
        break;

    case 1:
        {
            iparam1_t param;
            critical(
                !::nano::read(stream, param),
                "model parameter: failed to read from stream!");
            m_storage = param;
        }
        break;

    case 2:
        {
            sparam1_t param;
            critical(
                !::nano::read(stream, param),
                "model parameter: failed to read from stream!");
            m_storage = param;
        }
        break;

    default:
        critical0("model parameter: failed to read from stream (type=", type, ")!");
        break;
    }
}

void model_param_t::write(std::ostream& stream) const
{
    int32_t type = 0;
    if (is_evalue())
    {
        type = 0;
        critical(
            !::nano::write(stream, type) ||
            !::nano::write(stream, eparam()),
            "model parameter (", name(), "): failed to write to stream!");
    }
    else if (is_ivalue())
    {
        type = 1;
        critical(
            !::nano::write(stream, type) ||
            !::nano::write(stream, iparam()),
            "model parameter (", name(), "): failed to write to stream!");
    }
    else
    {
        type = 2;
        critical(
            !::nano::write(stream, type) ||
            !::nano::write(stream, sparam()),
            "model parameter (", name(), "): failed to write to stream!");
    }
}

eparam1_t& model_param_t::eparam()
{
    critical(!is_evalue(), "model parameter: expecting enumeration!");
    return std::get<eparam1_t>(m_storage);
}

const eparam1_t& model_param_t::eparam() const
{
    critical(!is_evalue(), "model parameter: expecting enumeration!");
    return std::get<eparam1_t>(m_storage);
}

iparam1_t& model_param_t::iparam()
{
    critical(!is_ivalue(), "model parameter: expecting integer!");
    return std::get<iparam1_t>(m_storage);
}

const iparam1_t& model_param_t::iparam() const
{
    critical(!is_ivalue(), "model parameter: expecting integer!");
    return std::get<iparam1_t>(m_storage);
}

sparam1_t& model_param_t::sparam()
{
    critical(!is_svalue(), "model parameter: expecting scalar!");
    return std::get<sparam1_t>(m_storage);
}

const sparam1_t& model_param_t::sparam() const
{
    critical(!is_svalue(), "model parameter: expecting scalar!");
    return std::get<sparam1_t>(m_storage);
}

std::ostream& ::nano::operator<<(std::ostream& stream, const model_param_t& param)
{
    stream << param.name() << "=";
    if (param.is_svalue())
    {
        return stream << param.sparam().get();
    }
    else if (param.is_ivalue())
    {
        return stream << param.iparam().get();
    }
    else
    {
        return stream << param.eparam().get();
    }
}

void model_config_t::add(string_t name, int32_t value)
{
    add(std::move(name), static_cast<int64_t>(value));
}

void model_config_t::add(string_t name, int64_t value)
{
    m_values.emplace_back(std::move(name), value);
}

void model_config_t::add(string_t name, scalar_t value)
{
    m_values.emplace_back(std::move(name), value);
}

void model_config_t::evaluate(scalar_t error)
{
    m_error = error;
}

std::ostream& ::nano::operator<<(std::ostream& stream, const model_config_t& config)
{
    const auto& values = config.values();
    for (size_t i = 0; i < values.size(); ++ i)
    {
        const auto& value = values[i];
        stream << value.first << "=";
        if (std::holds_alternative<int64_t>(value.second))
        {
            stream << std::get<int64_t>(value.second);
        }
        else
        {
            stream << std::get<scalar_t>(value.second);
        }

        if (i + 1 < values.size())
        {
            stream << ',';
        }
    }
    return stream;
}

model_factory_t& model_t::all()
{
    static model_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [&] ()
    {
        manager.add_by_type<gboost_model_t>();
        manager.add_by_type<linear_model_t>();
        manager.add_by_type<grid_search_model_t>();
    });

    return manager;
}

void model_t::read(std::istream& stream)
{
    serializable_t::read(stream);

    critical(
        !::nano::read(stream, m_params),
        "model: failed to read from stream!");
}

void model_t::write(std::ostream& stream) const
{
    serializable_t::write(stream);

    critical(
        !::nano::write(stream, m_params),
        "model: failed to write to stream!");
}

void model_t::set(const model_config_t& config)
{
    for (const auto& value : config.values())
    {
        if (std::holds_alternative<int64_t>(value.second))
        {
            set(value.first, std::get<int64_t>(value.second));
        }
        else
        {
            set(value.first, std::get<scalar_t>(value.second));
        }
    }
}

model_config_t model_t::config() const
{
    model_config_t config;
    for (const auto& param : m_params)
    {
        if (param.is_evalue())
        {
            config.add(param.eparam().name(), param.eparam().get());
        }
        else if (param.is_ivalue())
        {
            config.add(param.iparam().name(), param.iparam().get());
        }
        else
        {
            config.add(param.sparam().name(), param.sparam().get());
        }
    }

    return config;
}

model_param_t& model_t::find(const string_t& name)
{
    const auto it = std::find_if(m_params.begin(), m_params.end(), [&] (const model_param_t& param)
    {
        return param.name() == name;
    });

    critical(it == m_params.end(), "model: cannot find parameter by name (", name, ")!");
    return *it;
}

const model_param_t& model_t::find(const string_t& name) const
{
    const auto it = std::find_if(m_params.begin(), m_params.end(), [&] (const model_param_t& param)
    {
        return param.name() == name;
    });

    critical(it == m_params.end(), "model: cannot find parameter by name (", name, ")!");
    return *it;
}

tensor1d_t model_t::evaluate(const loss_t& loss, const dataset_t& dataset, const indices_t& samples) const
{
    const auto outputs = predict(dataset, samples);

    tensor1d_t errors(samples.size());
    loopr(samples.size(), tensor_size_t{1024}, [&] (tensor_size_t begin, tensor_size_t end, size_t)
    {
        const auto range = make_range(begin, end);
        const auto targets = dataset.targets(samples.slice(range));
        loss.error(targets, outputs.slice(range), errors.slice(range));
    });

    return errors;
}

kfold_result_t::kfold_result_t(tensor_size_t folds) :
    m_train_errors(folds),
    m_valid_errors(folds),
    m_models(static_cast<size_t>(folds))
{
}

kfold_result_t nano::kfold(const model_t& model_,
    const loss_t& loss, const dataset_t& dataset, const indices_t& samples, const solver_t& solver,
    tensor_size_t folds, tensor_size_t repetitions)
{
    const auto min_folds = 3;
    const auto min_repetitions = 1;

    critical(
        folds < min_folds,
        "kfold: the number of folds (", folds, ") should be greater than ", min_folds, "!");

    critical(
        repetitions < min_repetitions,
        "kfold: the number of repetitions (", repetitions, ") should be greater than ", min_repetitions, "!");

    kfold_result_t result{folds * repetitions};
    for (tensor_size_t repetition = 0, index = 0; repetition < repetitions; ++ repetition)
    {
        const auto kfold = kfold_t{samples, folds};

        for (tensor_size_t fold = 0; fold < folds; ++ fold, ++ index)
        {
            const auto [train_samples, valid_samples] = kfold.split(fold);

            auto model = model_.clone();
            result.m_train_errors(index) = model->fit(loss, dataset, train_samples, solver);
            result.m_valid_errors(index) = model->evaluate(loss, dataset, valid_samples).mean();
            result.m_models[static_cast<size_t>(index)] = std::move(model);
        }
    }

    return result;
}
