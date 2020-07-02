#pragma once

#include <nano/stream.h>
#include <nano/dataset.h>
#include <nano/factory.h>
#include <nano/mlearn/cluster.h>

namespace nano
{
    class wlearner_t;
    using wlearner_factory_t = factory_t<wlearner_t>;
    using rwlearner_t = wlearner_factory_t::trobject;
    using wlearners_t = std::vector<rwlearner_t>;

    ///
    /// \brief a weak learner is a machine learning model:
    ///     - parametrized by either a single feature or a small subset of features,
    ///     - easy to fit to the given residuals (aka the solution can be found analytically),
    ///     - with rather low accuracy that can be boosted by assemblying many of them using e.g. GradientBoosting.
    ///
    class NANO_PUBLIC wlearner_t : public serializable_t
    {
    public:

        ///
        /// \brief returns the available implementations.
        ///
        static wlearner_factory_t& all();

        ///
        /// \brief default constructor
        ///
        wlearner_t() = default;

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
        /// \brief clone the object.
        ///
        [[nodiscard]] virtual rwlearner_t clone() const = 0;

        ///
        /// \brief returns the dimensions of the outputs/predictions.
        ///
        [[nodiscard]] virtual tensor3d_dim_t odim() const = 0;

        ///
        /// \brief compute the predictions for the given range of samples in the given fold.
        ///
        virtual void predict(const dataset_t&, fold_t, tensor_range_t, tensor4d_map_t&& outputs) const = 0;

        ///
        /// \brief split the given samples in the fold using the currently selected features.
        ///
        [[nodiscard]] virtual cluster_t split(const dataset_t&, fold_t, const indices_t&) const = 0;

        ///
        /// \brief select the feature or the features and estimate their associated parameters
        ///     that matches the best the given residuals/gradients in terms of the L2-norm
        ///     using the given sample indices:
        ///
        ///     argmin_h mean(L2-norm(-gradients(i), scales(i) * h(i)), i in indices)
        ///
        ///     where h is the weak learner.
        ///
        [[nodiscard]] virtual scalar_t fit(const dataset_t&, fold_t,
            const tensor4d_t& gradients, const indices_t&, const tensor4d_t& scales) = 0;

        ///
        /// \brief adjust the weak learner's parameters to obtain linearly scaled predictions.
        ///
        /// NB: the scaling vector can be either:
        ///     - one dimensional, thus the same scale if applied to all splits or
        ///     - of the same dimension as the number of splits.
        ///
        virtual void scale(const vector_t&) = 0;

        ///
        /// \brief returns the selected features.
        ///
        [[nodiscard]] virtual indices_t features() const = 0;

        ///
        /// \brief change the batch size (aka number of samples to process at a time).
        ///
        /// NB: this may require tuning for optimum speed.
        ///
        void batch(int batch);

        ///
        /// \brief change the weak learner type.
        ///
        void type(wlearner);

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto type() const { return m_type; }
        [[nodiscard]] auto batch() const { return m_batch.get(); }

    protected:

        static void check(const indices_t&);
        void check(tensor_range_t, const tensor4d_map_t& outputs) const;

        template <typename toperator>
        static void for_each(tensor_range_t range, const indices_t& indices, const toperator& op)
        {
            assert(range.size());

            const auto *const iend = ::nano::end(indices);
            const auto *const ibegin = ::nano::begin(indices);

            std::for_each(
                std::lower_bound(ibegin, iend, range.begin()),
                std::lower_bound(ibegin, iend, range.end()),
                [&] (const tensor_size_t i) { op(i); });
        }

        static void scale(tensor4d_t& tables, const vector_t& scale);

    private:

        // attributes
        wlearner        m_type{wlearner::real}; ///<
        iparam1_t       m_batch{"wlearner::batch", 1, LE, 32, LE, 1024};///< batch size
    };
}
