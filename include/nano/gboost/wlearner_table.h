#pragma once

#include <nano/gboost/wlearner_feature1.h>

namespace nano
{
    class wlearner_table_t;

    template <>
    struct factory_traits_t<wlearner_table_t>
    {
        static string_t id() { return "table"; }
        static string_t description() { return "look-up-table weak learner"; }
    };

    ///
    ///
    /// \brief a (look-up) table is a weak learner that returns the output indexed by the selected feature:
    ///     table(x) =
    ///     {
    ///         tables[int(x(feature))], if x(feature) is given
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where feature is the selected discrete feature.
    ///
    /// NB: the continuous features and the missing feature values are skipped during fiting.
    ///
    class NANO_PUBLIC wlearner_table_t final : public wlearner_feature1_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        wlearner_table_t();

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] rwlearner_t clone() const override;

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

    private:

        [[nodiscard]] auto n_fvalues() const { return tables().size<0>(); }
    };
}
