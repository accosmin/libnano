#pragma once

#include <nano/loss.h>
#include <nano/solver.h>
#include <nano/stream.h>
#include <nano/dataset.h>

namespace nano
{
    class model_t;
    using model_factory_t = factory_t<model_t>;
    using rmodel_t = model_factory_t::trobject;
    using rmodels_t = std::vector<rmodel_t>;

    ///
    /// \brief safely stores a parameter for a machine learning model.
    ///
    ///     the parameter can be an integer, a scalar or an enumeration and
    ///     it can be serialized to and from binary streams.
    ///
    class NANO_PUBLIC model_param_t final : public serializable_t
    {
    public:

        using storage_t = std::variant<eparam1_t, iparam1_t, sparam1_t>;

        ///
        /// \brief default constructor
        ///
        model_param_t() = default;

        ///
        /// \brief constructor
        ///
        explicit model_param_t(eparam1_t);
        explicit model_param_t(iparam1_t);
        explicit model_param_t(sparam1_t);

        ///
        /// \brief change the parameter's value.
        ///
        void set(int32_t);
        void set(int64_t);
        void set(scalar_t);

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        void set(tenum value)
        {
            eparam().set(value);
        }

        ///
        /// \brief retrieve the current parameter's value.
        ///
        int64_t ivalue() const;
        scalar_t svalue() const;

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        tenum evalue() const
        {
            return eparam().as<tenum>();
        }

        ///
        /// \brief returns true if the parameter is an enumeration, an integer or a scalar.
        ///
        bool is_evalue() const;
        bool is_ivalue() const;
        bool is_svalue() const;

        ///
        /// \brief returns the parameter's name if initialized, otherwise throws an exception.
        ///
        const string_t& name() const;

        ///
        /// \brief @see serializable_t
        ///
        void read(std::istream&) override;

        ///
        /// \brief @see serializable_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief returns the stored parameters.
        ///
        const eparam1_t& eparam() const;
        const iparam1_t& iparam() const;
        const sparam1_t& sparam() const;

    private:

        eparam1_t& eparam();
        iparam1_t& iparam();
        sparam1_t& sparam();

        // attributes
        storage_t       m_storage;      ///<
    };

    using model_params_t = std::vector<model_param_t>;

    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const model_param_t&);

    ///
    /// \brief stores values for a set of parameters given by name, optionally with:
    ///     - the validation error (e.g. from k-fold cross-validation).
    ///
    class NANO_PUBLIC model_config_t
    {
    public:

        using xvalue_t = std::pair<string_t, std::variant<int64_t, scalar_t>>;
        using xvalues_t = std::vector<xvalue_t>;

        ///
        /// \brief default constructor
        ///
        model_config_t() = default;

        ///
        /// \brief store parameter's value.
        ///
        void add(string_t name, int32_t value);
        void add(string_t name, int64_t value);
        void add(string_t name, scalar_t value);

        ///
        /// \brief store validation error.
        ///
        void evaluate(scalar_t error);

        ///
        /// \brief access functions.
        ///
        auto error() const { return m_error; }
        const auto& values() const { return m_values; }

    private:

        // attributes
        xvalues_t       m_values;           ///<
        scalar_t        m_error{std::numeric_limits<scalar_t>::quiet_NaN()};    ///<
    };

    using model_configs_t = std::vector<model_config_t>;

    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const model_config_t&);

    ///
    /// \brief interface for machine learning models.
    ///
    /// the minimum set of operations are:
    ///     - training (mutable) which fits the model on the given dataset,
    ///     - prediction (constant) which evaluates the trained model on the given dataset,
    ///     - saving/reading to/from binary streams.
    ///
    class NANO_PUBLIC model_t : public serializable_t
    {
    public:

        ///
        /// \brief returns the available implementations.
        ///
        static model_factory_t& all();

        ///
        /// \brief default constructor.
        ///
        model_t() = default;

        ///
        /// \brief @see serializable_t
        ///
        void read(std::istream&) override;

        ///
        /// \brief @see serializable_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief clone the object.
        ///
        virtual rmodel_t clone() const = 0;

        ///
        /// \brief fit the model using the given samples and the current set of (hyper-)parameters
        ///     and returns the average error of the given samples.
        ///
        virtual scalar_t fit(const loss_t&, const dataset_t&, const indices_t&, const solver_t&) = 0;

        ///
        /// \brief evaluate the trained model and returns the error for each of the given samples.
        ///
        tensor1d_t evaluate(const loss_t&, const dataset_t&, const indices_t&) const;

        ///
        /// \brief evaluate the trained model and returns the predictions for each of the given samples.
        ///
        virtual tensor4d_t predict(const dataset_t&, const indices_t&) const = 0;

        ///
        /// \brief register new parameters.
        ///
        void register_param(eparam1_t param) { m_params.emplace_back(std::move(param)); }
        void register_param(iparam1_t param) { m_params.emplace_back(std::move(param)); }
        void register_param(sparam1_t param) { m_params.emplace_back(std::move(param)); }

        ///
        /// \brief set parameter values by name.
        ///
        void set(const model_config_t&);
        void set(const string_t& name, int32_t value) { find(name).set(value); }
        void set(const string_t& name, int64_t value) { find(name).set(value); }
        void set(const string_t& name, scalar_t value) { find(name).set(value); }

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        void set(const string_t& name, tenum value) { find(name).set(value); }

        ///
        /// \brief retrieve parameter values by name.
        ///
        int64_t ivalue(const string_t& name) const { return find(name).ivalue(); }
        scalar_t svalue(const string_t& name) const { return find(name).svalue(); }

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        tenum evalue(const string_t& name) const { return find(name).evalue<tenum>(); }

        ///
        /// \brief returns all stored parameters.
        ///
        const auto& params() const { return m_params; }
        model_config_t config() const;

    private:

        model_param_t& find(const string_t& name);
        const model_param_t& find(const string_t& name) const;

        // attributes
        model_params_t  m_params;   ///<
    };

    using imodel_t = identifiable_t<model_t>;
    using imodels_t = std::vector<imodel_t>;

    ///
    /// \brief gather the results of k-fold cross-validation.
    ///
    struct kfold_result_t
    {
        kfold_result_t() = default;
        explicit kfold_result_t(tensor_size_t folds);

        tensor1d_t      m_train_errors; ///<
        tensor1d_t      m_valid_errors; ///<
        rmodels_t       m_models;       ///<
    };

    ///
    /// \brief (repeated) k-fold cross-validation
    ///     using the given model as currently setup in terms of (hyper-)parameters.
    ///
    NANO_PUBLIC kfold_result_t kfold(
        const model_t&, const loss_t&, const dataset_t&, const indices_t&, const solver_t&,
        tensor_size_t folds = 5, tensor_size_t repetitions = 1);
}
