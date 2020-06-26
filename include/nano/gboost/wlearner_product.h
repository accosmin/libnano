#pragma once

#include <nano/gboost/wlearner.h>

namespace nano
{
    class wlearner_product_t;

    template <>
    struct factory_traits_t<wlearner_product_t>
    {
        static string_t id() { return "product"; }
        static string_t description() { return "feature-wise product weak learner"; }
    };

    ///
    /// \brief a product weak learner is performing an element-wise multiplication of
    ///     a fixed number of weak learners.
    ///
    /// NB: each weak learner term is fit greedily:
    ///     - initialize product(x) = 1
    ///     - for weak learner term k up to K
    ///         - fit term_k(x) to minimize L2(residual(x) - product(x) * term_k(x)
    ///         - product(x) = product(x) * term_k(x)
    ///
    /// NB: the missing feature values are skipped during fitting and force the prediction to be zero at evaluation.
    ///
    class NANO_PUBLIC wlearner_product_t final : public wlearner_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        wlearner_product_t() = default;

        ///
        /// \brief @see wlearner_t
        ///
        void read(std::istream&) override;

        ///
        /// \brief @see wlearner_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] std::ostream& print(std::ostream&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] tensor3d_dim_t odim() const override;

        ///
        /// \brief @see wlearner_t
        ///
        void scale(const vector_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        void predict(const dataset_t&, fold_t, tensor_range_t, tensor4d_map_t&& outputs) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] scalar_t fit(const dataset_t&, fold_t, const tensor4d_t& gradients, const indices_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] cluster_t split(const dataset_t&, fold_t, const indices_t&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] indices_t features() const override;

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto feature() const { return m_feature; }
        [[nodiscard]] const auto& tables() const { return m_tables; }

    private:

        void compatible(const dataset_t&) const;

        // attributes
        tensor_size_t   m_feature{-1};  ///< index of the selected feature
        tensor4d_t      m_tables;       ///< (2, #outputs) - weights + bias
    };
}
