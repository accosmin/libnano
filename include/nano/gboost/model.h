#pragma once

#include <nano/model.h>
#include <nano/gboost/wlearner.h>

namespace nano
{
    class gboost_model_t;

    template <>
    struct factory_traits_t<gboost_model_t>
    {
        static string_t id() { return "gboost"; }
        static string_t description() { return "GradientBoosting model (and variants: VadaBoost)"; }
    };

    using iwlearner_t = identifiable_t<wlearner_t>;
    using iwlearners_t = std::vector<iwlearner_t>;

    ///
    /// \brief (Stochastic) Gradient Boosting model.
    ///
    /// some important features:
    ///     - weak learners are selected from a configurable pool of prototypes and thus the final model
    ///         can mix different types of weak learners (e.g. like stumps with look-up-tables).
    ///     - support for variance-based regularization (like EBBoost or VadaBoost).
    ///     - builtin early stopping if the validation error doesn't decrease in a configurable number of boosting rounds.
    ///     - support for serialization of its parameters and the selected weak learners.
    ///     - training and evaluation is performed using all available threads.
    ///     - the bias computation and the scaling of the weak learners can be solved
    ///         using any of the available builtin line-search-based solvers (e.g. lBFGS, CGD, CG_DESCENT).
    ///     - support for estimating the importance of the selected features.
    ///
    /// see "The Elements of Statistical Learning", by Trevor Hastie, Robert Tibshirani, Jerome Friedman
    /// see "Greedy Function Approximation: A Gradient Boosting Machine", by Jerome Friedman
    /// see "Stochastic Gradient Boosting", by Jerome Friedman
    ///
    /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
    /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
    ///
    class NANO_PUBLIC gboost_model_t final : public model_t
    {
    public:

        ///
        /// \brief constructor
        ///
        gboost_model_t();

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
            std::enable_if_t<std::is_base_of_v<wlearner_t, twlearner>, bool> = true
        >
        void add(const twlearner& wlearner)
        {
            const auto id = factory_traits_t<twlearner>::id();
            auto rwlearner = std::make_unique<twlearner>(wlearner);
            add(id, std::move(rwlearner));
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
        /// \brief returns the selected features, optionally with their associated importance.
        ///
        feature_infos_t features() const;
        feature_infos_t features(
            const loss_t&, const dataset_t&, const indices_t&, const solver_t&,
            importance = importance::shuffle, tensor_size_t trials = 10) const;

        ///
        /// \brief configure the model.
        ///
        void batch(int64_t batch) { set("gboost::batch", batch); }
        void vAreg(scalar_t vAreg) { set("gboost::vAreg", vAreg); }
        void rounds(int64_t rounds) { set("gboost::rounds", rounds); }
        void epsilon(scalar_t epsilon) { set("gboost::epsilon", epsilon); }
        void wscale(::nano::wscale wscale) { set("gboost::wscale", wscale); }
        void subsample(scalar_t subsample) { set("gboost::subsample", subsample); }
        void shrinkage(scalar_t shrinkage) { set("gboost::shrinkage", shrinkage); }

        ///
        /// \brief access functions.
        ///
        auto batch() const { return ivalue("gboost::batch"); }
        auto vAreg() const { return svalue("gboost::vAreg"); }
        auto rounds() const { return ivalue("gboost::rounds"); }
        auto epsilon() const { return svalue("gboost::epsilon"); }
        auto shrinkage() const { return svalue("gboost::shrinkage"); }
        auto subsample() const { return svalue("gboost::subsample"); }
        auto wscale() const { return evalue<::nano::wscale>("gboost::wscale"); }

    private:

        void add(string_t id, rwlearner_t&&);
        void scale(const cluster_t&, const indices_t&, const vector_t&, tensor4d_t&) const;
        bool done(tensor_size_t round, const tensor1d_t&, const solver_state_t&, const indices_t&) const;

        indices_t make_indices(const indices_t&) const;
        cluster_t make_cluster(const dataset_t&, const indices_t&, const wlearner_t&) const;
        tensor1d_t evaluate(const dataset_t&, const indices_t&, const loss_t&, const tensor4d_t&) const;

        // attributes
        tensor1d_t      m_bias;                 ///< fitted bias
        iwlearners_t    m_protos;               ///< weak learners to choose from (prototypes)
        iwlearners_t    m_iwlearners;           ///< fitted weak learners chosen from the prototypes
    };
}
