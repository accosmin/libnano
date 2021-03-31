#pragma once

#include <nano/model.h>

namespace nano
{
    class linear_model_t;

    template <>
    struct factory_traits_t<linear_model_t>
    {
        static string_t id() { return "linear"; }
        static string_t description() { return "linear regression model (and variants: Ridge, Lasso, ElasticNet)"; }
    };

    ///
    /// \brief a linear model is an affine transformation of the flatten input features x:
    ///     y(x) = weights * x + bias.
    ///
    /// NB: the model can be regularized using the following methods:
    ///     - lasso (the L1-norm of the weights) by tuning ::l1reg() accordingly,
    ///     - ridge (the L2-norm of the weights) by tuning ::l2reg() accordingly,
    ///     - elastic net (the L1-norm and the L2-norm of the weights) by tuning ::l1reg() and ::l2reg() accordingly,
    ///     - variance (of the loss values across samples) by tuning ::vAreg() accordingly.
    ///
    /// NB: the inputs should be normalized during training to speed-up convergence (@see nano::feature_scaling).
    ///
    /// see "Regression Shrinkage and Selection via the lasso", by R. Tibshirani
    /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
    /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
    ///
    class NANO_PUBLIC linear_model_t final : public model_t
    {
    public:

        ///
        /// \brief constructor
        ///
        linear_model_t();

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
        /// \brief configure the model.
        ///
        void batch(int64_t batch) { set("linear::batch", batch); }
        void l1reg(scalar_t l1reg) { set("linear::l1reg", l1reg); }
        void l2reg(scalar_t l2reg) { set("linear::l2reg", l2reg); }
        void vAreg(scalar_t vAreg) { set("linear::vAreg", vAreg); }
        void scaling(feature_scaling scaling) { set("linear::scaling", scaling); }

        auto batch() const { return ivalue("linear::batch"); }
        auto l1reg() const { return svalue("linear::l1reg"); }
        auto l2reg() const { return svalue("linear::l2reg"); }
        auto vAreg() const { return svalue("linear::vAreg"); }
        auto scaling() const { return evalue<feature_scaling>("linear::scaling"); }

        const auto& bias() const { return m_bias; }
        const auto& weights() const { return m_weights; }

    private:

        // attributes
        tensor1d_t      m_bias;             ///< bias vector (#outputs)
        tensor2d_t      m_weights;          ///< weight matrix (#inputs, #outputs)
    };
}
