#pragma once

#include <nano/arch.h>
#include <nano/tpool.h>
#include <nano/factory.h>
#include <nano/mlearn/feature.h>
#include <nano/mlearn/elemwise.h>

namespace nano
{
    class dataset_t;
    using dataset_factory_t = factory_t<dataset_t>;
    using rdataset_t = dataset_factory_t::trobject;

    ///
    /// \brief machine learning dataset consisting of a collection of samples.
    ///
    /// NB: each sample consists of:
    ///     - a fixed number of (input) feature values and
    ///     - optionally a target if a supervised ML task.
    ///
    class NANO_PUBLIC dataset_t
    {
    public:

        ///
        /// \brief returns the available implementations.
        ///
        static dataset_factory_t& all();

        ///
        /// \brief default constructor
        ///
        dataset_t() = default;

        ///
        /// \brief disable copying
        ///
        dataset_t(const dataset_t&) = default;
        dataset_t& operator=(const dataset_t&) = default;

        ///
        /// \brief enable moving
        ///
        dataset_t(dataset_t&&) noexcept = default;
        dataset_t& operator=(dataset_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        virtual ~dataset_t() = default;

        ///
        /// \brief load dataset in memory.
        ///
        /// NB: any error is considered critical and an exception will be triggered.
        ///
        virtual void load() = 0;

        ///
        /// \brief returns the total number of samples.
        ///
        virtual tensor_size_t samples() const = 0;

        ///
        /// \brief returns the dimensions of a sample.
        ///
        virtual tensor3d_dims_t idims() const = 0;

        ///
        /// \brief returns the dimensions of the target (if provided).
        ///
        virtual tensor3d_dims_t tdims() const = 0;

        ///
        /// \brief returns the feature description of the target (if provided).
        ///
        virtual feature_t target() const = 0;

        ///
        /// \brief returns the toal number of input features.
        ///
        tensor_size_t features() const { return ::nano::size(idims()); }

        ///
        /// \brief returns the feature description of a given feature index.
        ///
        /// NB: the feature index must be in the range [0, features()).
        ///
        virtual feature_t feature(tensor_size_t index) const = 0;

        ///
        /// \brief returns the samples that can be used for training.
        ///
        indices_t train_samples() const { return make_train_samples(); }

        ///
        /// \brief returns the samples that should only be used for testing.
        ///
        /// NB: assumes a fixed set of test samples.
        ///
        indices_t test_samples() const { return make_test_samples(); }

        ///
        /// \brief returns the feature values of the given samples.
        ///
        virtual tensor4d_t inputs(const indices_cmap_t& samples) const = 0;

        ///
        /// \brief returns the feature values of the given samples.
        ///
        virtual tensor1d_t inputs(const indices_cmap_t& samples, tensor_size_t feature) const = 0;

        ///
        /// \brief returns the feature values of the given samples.
        ///
        virtual tensor2d_t inputs(const indices_cmap_t& samples, const indices_t& features) const = 0;

        ///
        /// \brief returns the targets of the given samples.
        ///
        virtual tensor4d_t targets(const indices_cmap_t& samples) const = 0;

        ///
        /// \brief returns the element-wise statistics for all inputs of the given fold.
        ///
        /// NB: e.g. this is useful for normalizing the continuous inputs to zero mean and unit variance.
        ///
        elemwise_stats_t istats(const indices_cmap_t& samples, tensor_size_t batch) const
        {
            std::vector<elemwise_stats_t> stats(tpool_t::size(), elemwise_stats_t{idims()});
            loopr(samples.size(), batch, [&] (tensor_size_t begin, tensor_size_t end, size_t tnum)
            {
                const auto range = make_range(begin, end);
                stats[tnum].update(inputs(samples.slice(range)));
            });

            std::for_each(++ stats.begin(), stats.end(), [&] (const elemwise_stats_t& tstats)
            {
                stats[0].update(tstats);
            });
            return stats[0].done(samples.size());
        }

        ///
        /// \brief set all the samples for training.
        ///
        void no_testing()
        {
            m_testing.resize(samples());
            m_testing.zero();
        }

        ///
        /// \brief set the given range of samples for testing.
        ///
        /// NB: this accumulates the previous range of samples set for testing.
        ///
        void testing(tensor_range_t range)
        {
            if (m_testing.size() != samples())
            {
                m_testing.resize(samples());
                m_testing.zero();
            }

            assert(range.begin() >= 0 && range.end() <= m_testing.size());
            m_testing.vector().segment(range.begin(), range.size()).setConstant(1);
        }

        ///
        /// \brief automatically detect the appropriate machine learning task given the target feature.
        ///
        task_type type() const;

    private:

        indices_t make_train_samples() const
        {
            const auto samples = this->samples();
            const auto has_testing = m_testing.size() == samples;
            return has_testing ? filter(samples - m_testing.vector().sum(), samples, 0) : arange(0, samples);
        }

        indices_t make_test_samples() const
        {
            const auto samples = this->samples();
            const auto has_testing = m_testing.size() == samples;
            return has_testing ? filter(m_testing.vector().sum(), samples, 1) : indices_t{};
        }

        indices_t filter(tensor_size_t count, tensor_size_t samples, tensor_size_t condition) const
        {
            indices_t indices(count);
            for (tensor_size_t sample = 0, index = 0; sample < samples; ++ sample)
            {
                if (m_testing(sample) == condition)
                {
                    assert(index < indices.size());
                    indices(index ++) = sample;
                }
            }
            return indices;
        }

        // attributes
        indices_t       m_testing;      ///< (#samples,) - mark sample for testing, if != 0
    };
}
