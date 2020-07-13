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
        wlearner_product_t();

        ///
        /// \brief default destructor
        ///
        ~wlearner_product_t();

        ///
        /// \brief enable moving and copying
        ///
        wlearner_product_t(const wlearner_product_t&);
        wlearner_product_t(wlearner_product_t&&) noexcept;
        wlearner_product_t& operator=(const wlearner_product_t&);
        wlearner_product_t& operator=(wlearner_product_t&&) noexcept;

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
        [[nodiscard]] rwlearner_t clone() const override;

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
        [[nodiscard]] scalar_t fit(const dataset_t&, fold_t,
            const tensor4d_t&, const indices_t&, const tensor4d_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] cluster_t split(const dataset_t&, fold_t, const indices_t&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] indices_t features() const override;

        ///
        /// \brief change the number of terms in the product.
        ///
        void degree(int degree);

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto degree() const { return m_degree.get(); }
        [[nodiscard]] const auto& protos() const { return m_protos; }
        [[nodiscard]] const auto& terms() const { return m_terms; }

    private:

        void compatible(const dataset_t&) const;
        void add(string_t id, rwlearner_t&& prototype);

        // attributes
        iwlearners_t    m_terms;                                        ///< chosen weak learners (terms) in the product
        iwlearners_t    m_protos;                                       ///< weak learners to choose from
        iparam1_t       m_degree{"product::degree", 1, LE, 3, LE, 20};  ///< maximum number of terms in the product
    };
}
