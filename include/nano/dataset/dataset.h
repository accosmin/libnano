#pragma once

#include <nano/arch.h>
#include <nano/factory.h>

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
        /// \brief returns the appropriate machine learning task (by inspecting the target feature).
        ///
        virtual task_type type() const = 0;

        ///
        /// \brief returns the total number of samples.
        ///
        virtual tensor_size_t samples() const = 0;

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
        /// \brief returns an iterator over the given samples.
        ///
        virtual rfeature_dataset_iterator_t feature_iterator(indices_t samples) const = 0;
        virtual rflatten_dataset_iterator_t flatten_iterator(indices_t samples) const = 0;

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
