#pragma once

#include <nano/loss.h>
#include <nano/solver.h>
#include <nano/stream.h>
#include <nano/dataset.h>
#include <nano/mlearn/train.h>
#include <nano/gboost/wlearner.h>

namespace nano
{
    //
    // TODO: estimate feature importance and stability across folds.
    // TODO: a better splitting criterion for decision trees
    //       may be to check if the score of current node is smaller than of the parent!

    ///
    /// \brief Gradient Boosting model.
    ///
    /// some important features:
    ///     - weak learners are selected from a configurable pool of prototypes and thus the final model
    ///         can mix different types of weak learners (e.g. like discrete stumps with real-valued tables).
    ///     - support for both discrete and real-valued weak learners.
    ///     - support for variance-based regularization (like EBBoost or VadaBoost)
    ///         with automatic tuning on the validation dataset if enabled.
    ///     - builtin early stopping if the validation error doesn't decrease in a configurable number of boosting rounds.
    ///     - support for serialization of its parameters and the selected weak learners.
    ///     - training and evaluation is performed using all available threads.
    ///     - a model is trained from scratch for each fold and
    ///         the final predictions are the average of the predictions of all these models (like in model averaging).
    ///     - the bias computation and the scaling of the weak learners can be solved
    ///         using any of the available builtin line-search-based solvers (e.g. lBFGS, CGD, CG_DESCENT).
    ///
    /// missing features:
    ///     - shrinkage (useful for improving generalization):
    ///         - not implemented because it simplifies implementation and
    ///         - the builtin alternative of the averaging the trained models for each fold is arguably superior.
    ///
    /// see "The Elements of Statistical Learning", by Trevor Hastie, Robert Tibshirani, Jerome Friedman
    /// see "Greedy Function Approximation: A Gradient Boosting Machine", by Jerome Friedman
    /// see "Stochastic Gradient Boosting", by Jerome Friedman
    ///
    /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
    /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
    ///
    class NANO_PUBLIC gboost_model_t : public serializable_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        gboost_model_t() = default;

        ///
        /// \brief @see serializable_t
        ///
        void read(std::istream&) override;

        ///
        ///
        /// \brief @see serializable_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief register a prototype weak learner to choose from by its ID in the associated factory.
        ///
        void add(const string_t& id);

        ///
        /// \brief register a prototype weak learner to choose from.
        ///
        template
        <
            typename twlearner,
            typename = typename std::enable_if<std::is_base_of<wlearner_t, twlearner>::value>::type
        >
        void add(const twlearner& wlearner)
        {
            const auto id = factory_traits_t<twlearner>::id();
            auto rwlearner = std::make_unique<twlearner>(wlearner);
            add(id, std::move(rwlearner));
        }

        ///
        /// \brief train the linear model on the given samples.
        ///
        [[nodiscard]] train_result_t train(const loss_t&, const dataset_t&, const solver_t&);

        ///
        /// \brief returns the selected features with their associated statistics (e.g. importance).
        ///
        [[nodiscard]] feature_infos_t features(const loss_t&, const dataset_t&, tensor_size_t trials = 10) const;

        ///
        /// \brief compute the predictions for all samples in the given fold.
        ///
        void predict(const dataset_t&, fold_t, tensor4d_t& outputs) const;
        void predict(const dataset_t&, fold_t, tensor4d_map_t&& outputs) const;

        ///
        /// \brief change parameters
        ///
        void rounds(int rounds) { m_rounds = rounds; }
        void subsample(int s) { m_subsample = s; }
        void patience(int patience) { m_patience = patience; }
        void batch(tensor_size_t batch) { m_batch = batch; }
        void tune_steps(int steps) { m_tune_steps = steps; }
        void tune_trials(int trials) { m_tune_trials = trials; }

        void scale(::nano::wscale scale) { m_scale = scale; }
        void regularization(::nano::regularization regularization) { m_regularization = regularization; }

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto batch() const { return m_batch.get(); }
        [[nodiscard]] auto rounds() const { return m_rounds.get(); }
        [[nodiscard]] auto patience() const { return m_patience.get(); }
        [[nodiscard]] auto subsample() const { return m_subsample.get(); }
        [[nodiscard]] auto tune_steps() const { return m_tune_steps.get(); }
        [[nodiscard]] auto tune_trials() const { return m_tune_trials.get(); }

        [[nodiscard]] auto scale() const { return m_scale; }
        [[nodiscard]] auto regularization() const { return m_regularization; }

        [[nodiscard]] const auto& models() const { return m_models; }

    private:

        struct model_t
        {
            tensor1d_t      m_bias;
            iwlearners_t    m_protos;
        };
        using models_t = std::vector<model_t>;

        void add(string_t id, rwlearner_t&&);

        [[nodiscard]] indices_t make_indices(tensor_size_t samples) const;

        [[nodiscard]] train_status done(
            tensor_size_t round, scalar_t vAreg, const tensor1d_t&, const tensor1d_t&,
            const solver_state_t&, const indices_t& features, train_curve_t&) const;

        [[nodiscard]] std::tuple<scalar_t, model_t, tensor4d_t> train(
            const loss_t&, const dataset_t&, size_t fold, const solver_t&, scalar_t vAreg, train_curve_t&) const;

        // attributes
        models_t        m_models;                                               ///< model per fold
        iwlearners_t    m_protos;                                               ///< weak learners to choose from
        iparam1_t       m_batch{"gboost::batch", 1, LE, 32, LE, 4096};          ///< #samples to use at once to predict
        iparam1_t       m_rounds{"gboost::rounds", 1, LE, 1000, LE, 10000};     ///< maximum number of boosting rounds
        iparam1_t       m_patience{"gboost::patience", 1, LE, 100, LE, 1000};   ///< #rounds to wait to detect overfitting
        iparam1_t       m_subsample{"gboost::subsample", 10, LE, 100, LE, 100}; ///< subsampling percentage
        iparam1_t       m_tune_trials{"gboost::tune_trials", 4, LE, 7, LE, 10}; ///< tuning parameters
        iparam1_t       m_tune_steps{"gboost::tune_steps", 1, LE, 2, LE, 10};   ///< tuning parameters

        ::nano::wscale  m_scale{::nano::wscale::tboost};                        ///<
        ::nano::regularization  m_regularization{::nano::regularization::none}; ///<
    };
}
