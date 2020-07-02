#pragma once

#include <nano/gboost/wlearner_feature1.h>

namespace nano
{
    ///
    /// \brief functions to map a scalar feature value.
    ///
    struct fun1_lin_t
    {
        static auto get(scalar_t x)
        {
            return x;
        }
    };

    struct fun1_sin_t
    {
        static auto get(scalar_t x)
        {
            return std::sin(x);
        }
    };

    struct fun1_cos_t
    {
        static auto get(scalar_t x)
        {
            return std::cos(x);
        }
    };

    struct fun1_log_t
    {
        static auto get(scalar_t x)
        {
            static const auto epsilon = std::sqrt(std::numeric_limits<scalar_t>::epsilon());
            return std::log(epsilon + x * x);
        }
    };

    ///
    /// \brief this weak learner is performing an element-wise transformation of the form:
    ///     affine1(x) =
    ///     {
    ///         weights[0] * fun1(x(feature)) + weights[1], if x(feature) is given
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where `feature` is the selected continuous feature.
    ///
    /// NB: the discrete features and the missing feature values are skipped during fiting.
    ///
    template <typename tfun1>
    class wlearner_affine_t final : public wlearner_feature1_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        wlearner_affine_t();

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
    };

    using wlearner_cos1_t = wlearner_affine_t<fun1_cos_t>;
    using wlearner_lin1_t = wlearner_affine_t<fun1_lin_t>;
    using wlearner_log1_t = wlearner_affine_t<fun1_log_t>;
    using wlearner_sin1_t = wlearner_affine_t<fun1_sin_t>;

    template <> struct factory_traits_t<wlearner_cos1_t>
    {
        static string_t id() { return "cos1"; }
        static string_t description() { return "affine feature-wise weak learner: h(x) = a * cos(x[feature]) + b"; }
    };

    template <> struct factory_traits_t<wlearner_lin1_t>
    {
        static string_t id() { return "lin1"; }
        static string_t description() { return "affine feature-wise weak learner: h(x) = a * x[feature] + b"; }
    };

    template <> struct factory_traits_t<wlearner_log1_t>
    {
        static string_t id() { return "log1"; }
        static string_t description() { return "affine feature-wise weak learner: h(x) = a * log(eps + x[feature]^2) + b"; }
    };

    template <> struct factory_traits_t<wlearner_sin1_t>
    {
        static string_t id() { return "sin1"; }
        static string_t description() { return "affine feature-wise weak learner: h(x) = a * sin(x[feature]) + b"; }
    };
}
