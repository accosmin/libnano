#pragma once

#include <cassert>
#include <nano/loss.h>
#include <nano/mlearn/class.h>

namespace nano
{
    ///
    /// \brief un-structured loss function: the 3D structure of a sample is flatten
    ///     and all dimensions are considered the same in computing the loss.
    ///
    /// NB: the multi-label classification problem is handled by summing or averaging:
    ///     - the associated binary classification loss value per output
    ///     - the associated 0-1 loss error per output
    ///
    /// see the following resources regarding loss functions for classification:
    ///
    /// (1): "On the design of robust classifiers for computer vision",
    ///      2010, by H. Masnadi-Shirazi, V. Mahadevan, N. Vasconcelos
    ///
    /// (2): "On the design of loss functions for classification: theory, robustness to outliers, and SavageBoost",
    ///      2008, by H. Masnadi-Shirazi, N. Vasconcelos
    ///
    template <typename top>
    class array_loss_t final : public loss_t
    {
    public:

        ///
        /// \brief @see loss_t
        ///
        void error(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor1d_map_t errors) const override
        {
            assert(targets.dims() == outputs.dims());
            assert(errors.size() == targets.size<0>());

            for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++ i)
            {
                errors(i) = top::error(targets.array(i), outputs.array(i));
            }
        }

        ///
        /// \brief @see loss_t
        ///
        void value(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor1d_map_t values) const override
        {
            assert(targets.dims() == outputs.dims());
            assert(values.size() == targets.size<0>());

            for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++ i)
            {
                values(i) = top::value(targets.array(i), outputs.array(i));
            }
        }

        ///
        /// \brief @see loss_t
        ///
        void vgrad(const tensor4d_cmap_t& targets, const tensor4d_cmap_t& outputs, tensor4d_map_t vgrads) const override
        {
            assert(targets.dims() == vgrads.dims());
            assert(targets.dims() == outputs.dims());

            for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++ i)
            {
                top::vgrad(targets.array(i), outputs.array(i), vgrads.array(i));
            }
        }
    };

    namespace detail
    {
        ///
        /// \brief generic (multivariate) regression loss that upper-bounds
        ///     the L1-distance between target and output.
        ///
        struct absdiff_t
        {
            template <typename tarray>
            static auto error(const tarray& target, const tarray& output)
            {
                return (target - output).abs().sum();
            }
        };

        ///
        /// \brief multi-class classification loss that predicts the labels with positive output.
        ///
        struct mclass_t
        {
            template <typename tarray>
            static auto error(const tarray& target, const tarray& output)
            {
                const auto edges = target * output;
                const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
                return static_cast<scalar_t>((edges < epsilon).count());
            }
        };

        ///
        /// \brief single-class classification loss that predicts the label with the highest score.
        ///
        struct sclass_t
        {
            template <typename tarray>
            static auto error(const tarray& target, const tarray& output)
            {
                if (target.size() > 1)
                {
                    tensor_size_t idx = -1;
                    output.array().maxCoeff(&idx);

                    return static_cast<scalar_t>(is_pos_target(target(idx)) ? 0 : 1);
                }
                else
                {
                    const auto edges = target.array() * output.array();
                    const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
                    return static_cast<scalar_t>((edges < epsilon).count());
                }
            }
        };

        ///
        /// \brief class negative log-likelihood loss (also called cross-entropy loss).
        ///
        template <typename terror>
        struct classnll_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                tensor_size_t imax = 0;
                const auto omax = output.maxCoeff(&imax);

                scalar_t value = 0, posum = 0;
                for (tensor_size_t i = 0, size = target.size(); i < size; ++ i)
                {
                    value += std::exp(output(i) - omax);
                    if (is_pos_target(target(i)))
                    {
                        posum += output(i);
                    }
                }
                return std::log(value) - posum + omax;
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                tensor_size_t imax = 0;
                const auto omax = output.maxCoeff(&imax);

                scalar_t value = 0;
                for (tensor_size_t i = 0, size = target.size(); i < size; ++ i)
                {
                    value += (vgrad(i) = std::exp(output(i) - omax));
                }
                for (tensor_size_t i = 0, size = target.size(); i < size; ++ i)
                {
                    vgrad(i) /= value;
                    if (is_pos_target(target(i)))
                    {
                        vgrad(i) -= 1.0;
                    }
                }
            }
        };

        ///
        /// \brief multi-class exponential loss.
        ///
        template <typename terror>
        struct exponential_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return (-target * output).exp().sum();
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                vgrad = -target * (-target * output).exp();
            }
        };

        ///
        /// \brief multi-class logistic loss.
        ///
        template <typename terror>
        struct logistic_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                scalar_t value = 0.0;
                for (tensor_size_t i = 0, size = target.size(); i < size; ++ i)
                {
                    const auto x = -target(i) * output(i);
                    value += (x < 1.0) ? std::log1p(std::exp(x)) : (x + std::log1p(std::exp(-x)));
                }
                return value;
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                for (tensor_size_t i = 0, size = target.size(); i < size; ++ i)
                {
                    const auto x = -target(i) * output(i);
                    const auto g = (x < 1.0) ? (std::exp(x) / (1.0 + std::exp(x))) : (1.0 / (1.0 + std::exp(-x)));
                    vgrad(i) = -target(i) * g;
                }
            }
        };

        ///
        /// \brief multi-class hinge loss.
        ///
        template <typename terror>
        struct hinge_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return (1 - target * output).max(0).sum();
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                vgrad = -target * ((1 - target * output).sign() + 1) * 0.5;
            }
        };

        ///
        /// \brief multi-class savage loss.
        ///
        template <typename terror>
        struct savage_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return (1 / (1 + (target * output).exp()).square()).sum();
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                vgrad = -2 * target / ((1 + (target * output).exp()).square() * (1 + (-target * output).exp()));
            }
        };

        ///
        /// \brief multi-class tangent loss.
        ///
        template <typename terror>
        struct tangent_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return (2 * (target * output).atan() - 1).square().sum();
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                vgrad = 4 * target * (2 * (target * output).atan() - 1) / (1 + (target * output).square());
            }
        };

        ///
        /// \brief absolute-difference loss.
        ///
        template <typename terror>
        struct absolute_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return (output - target).abs().sum();
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                vgrad = (output - target).sign();
            }
        };

        ///
        /// \brief squared-difference loss.
        ///
        template <typename terror>
        struct squared_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return scalar_t(0.5) * (output - target).square().sum();
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                vgrad = output - target;
            }
        };

        ///
        /// \brief robust-to-noise Cauchy loss.
        ///
        template <typename terror>
        struct cauchy_t : public terror
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return scalar_t(0.5) * ((target - output).square() + 1).log().sum();
            }

            template <typename tarray, typename tgarray>
            static void vgrad(const tarray& target, const tarray& output, tgarray&& vgrad)
            {
                vgrad = (output - target) / (1 + (output - target).square());
            }
        };
    }

    using cauchy_loss_t = array_loss_t<detail::cauchy_t<detail::absdiff_t>>;
    using squared_loss_t = array_loss_t<detail::squared_t<detail::absdiff_t>>;
    using absolute_loss_t = array_loss_t<detail::absolute_t<detail::absdiff_t>>;

    using shinge_loss_t = array_loss_t<detail::hinge_t<detail::sclass_t>>;
    using ssavage_loss_t = array_loss_t<detail::savage_t<detail::sclass_t>>;
    using stangent_loss_t = array_loss_t<detail::tangent_t<detail::sclass_t>>;
    using sclassnll_loss_t = array_loss_t<detail::classnll_t<detail::sclass_t>>;
    using slogistic_loss_t = array_loss_t<detail::logistic_t<detail::sclass_t>>;
    using sexponential_loss_t = array_loss_t<detail::exponential_t<detail::sclass_t>>;

    using mhinge_loss_t = array_loss_t<detail::hinge_t<detail::mclass_t>>;
    using msavage_loss_t = array_loss_t<detail::savage_t<detail::mclass_t>>;
    using mtangent_loss_t = array_loss_t<detail::tangent_t<detail::mclass_t>>;
    using mlogistic_loss_t = array_loss_t<detail::logistic_t<detail::mclass_t>>;
    using mexponential_loss_t = array_loss_t<detail::exponential_t<detail::mclass_t>>;
}
