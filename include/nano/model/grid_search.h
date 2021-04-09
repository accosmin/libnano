#pragma once

#include <nano/model.h>

namespace nano
{
    class grid_search_model_t;

    template <>
    struct factory_traits_t<grid_search_model_t>
    {
        static string_t id() { return "grid-search"; }
        static string_t description() { return "tune the given model using variants of grid-search"; }
    };

    ///
    /// \brief list of values to evaluate for a parameter.
    ///
    using param_values_t = std::pair<
        string_t,                               ///< parameter name
        std::variant<
            std::vector<int64_t>,               ///< integer (discrete) parameter values or
            std::vector<scalar_t>               ///< scalar (continuous) parameter values to evaluate
        >
    >;

    using param_grid_t = std::vector<param_values_t>;

    ///
    /// \brief tune the given model by evaluating the combinations of the given hyper-parameters:
    ///     - either exhaustively, if the total number of combinations is smaller than ```max_trials```,
    ///     - or by randomly sampling ```max_trials``` of them, otherwise.
    ///
    /// the tuning is performed in two steps:
    ///     - evaluate all hyper-parameter configuration using k-fold cross validation and then
    ///     - train the model with the best hyper-parameters using the whole training dataset.
    ///
    /// NB: all the evaluated hyper-parameter configurations are available for further analysis.
    /// NB: this implementation is simular to the inner loop of typical nested k-fold cross-validation.
    ///
    class NANO_PUBLIC grid_search_model_t final : public model_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        grid_search_model_t() = default;

        ///
        /// \brief constructor
        ///
        grid_search_model_t(const string_t& model_id, param_grid_t);

        ///
        /// \brief constructor
        ///
        grid_search_model_t(string_t model_id, rmodel_t, param_grid_t);

        ///
        /// \brief constructor
        ///
        template
        <
            typename tmodel,
            std::enable_if_t<std::is_base_of_v<model_t, tmodel>, bool> = true
        >
        grid_search_model_t(const tmodel& model, const param_grid_t& grid) :
            grid_search_model_t(factory_traits_t<tmodel>::id(), model.clone(), grid)
        {
        }

        ///
        /// \brief @see serializable_t
        ///
        void read(std::istream&) override;

        ///
        /// \brief @see serializable_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief @see model_t
        ///
        rmodel_t clone() const override;

        ///
        /// \brief @see model_t
        ///
        scalar_t fit(const loss_t&, const dataset_t&, const indices_t&, const solver_t&) override;

        ///
        /// \brief @see model_t
        ///
        tensor4d_t predict(const dataset_t&, const indices_t&) const override;

        ///
        /// \brief return the evaluated hyper-parameter configurations with the associated cross-validation error.
        ///
        const auto& configs() const { return m_configs; }

        ///
        /// \brief return the model with the best hyper-parameter configuration.
        ///
        const auto& model() const { return m_imodel.get(); }

        ///
        /// \brief configure the model.
        ///
        void folds(int64_t folds) { set("grid-search::folds", folds); }
        void max_trials(int64_t trials) { set("grid-search::max_trials", trials); }

        ///
        /// \brief access functions.
        ///
        auto folds() const { return ivalue("grid-search::folds"); }
        auto max_trials() const { return ivalue("grid-search::max_trials"); }

    private:

        using count_config_t = tensor_mem_t<size_t, 1>;

        count_config_t make_counts() const;
        model_config_t make_config(const count_config_t&) const;

        // attributes
        imodel_t        m_imodel;               ///< model to tune and evaluate
        param_grid_t    m_grid;                 ///< hyper-parameter values to evaluate
        model_configs_t m_configs;              ///< evaluated hyper-parameter configurations
    };
}
