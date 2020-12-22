#pragma once

#include <nano/tensor.h>
#include <nano/numeric.h>
#include <nano/mlearn/enums.h>

namespace nano
{
    ///
    /// \brief element-wise statistics, e.g. for 3D inputs/features.
    ///
    class elemwise_stats_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        elemwise_stats_t() = default;

        ///
        /// \brief constructor
        ///
        explicit elemwise_stats_t(const tensor3d_dims_t dims) :
            m_min(dims),
            m_max(dims),
            m_mean(dims),
            m_stdev(dims)
        {
            m_mean.constant(0);
            m_stdev.constant(0);
            m_min.constant(std::numeric_limits<scalar_t>::max());
            m_max.constant(std::numeric_limits<scalar_t>::lowest());
        }

        ///
        /// \brief set the statistics from an external source.
        ///
        template <template <typename, size_t> class tstorage>
        void set(
            const tensor_t<tstorage, scalar_t, 3>& min, const tensor_t<tstorage, scalar_t, 3>& max,
            const tensor_t<tstorage, scalar_t, 3>& mean, const tensor_t<tstorage, scalar_t, 3>& stdev)
        {
            assert(max.dims() == min.dims());
            assert(mean.dims() == min.dims());
            assert(stdev.dims() == min.dims());

            m_min = min;
            m_max = max;
            m_mean = mean;
            m_stdev = stdev;
        }

        ///
        /// \brief update statistics with the given samples.
        ///
        template <template <typename, size_t> class tstorage>
        void update(const tensor_t<tstorage, scalar_t, 4>& inputs)
        {
            const auto samples = this->samples(inputs);
            for (tensor_size_t s = 0; s < samples; ++ s)
            {
                m_min.array() = inputs.array(s).min(m_min.array());
                m_max.array() = inputs.array(s).max(m_max.array());
            }

            const auto isize = nano::size(m_min.dims());
            const auto imatrix = inputs.reshape(samples, isize).matrix();

            m_mean.array() += imatrix.array().colwise().sum();
            m_stdev.array() += imatrix.array().square().colwise().sum();
        }

        ///
        /// \brief update statistics with the given object.
        ///
        void update(const elemwise_stats_t& other)
        {
            m_mean.array() += other.m_mean.array();
            m_stdev.array() += other.m_stdev.array();
            m_min.array() = other.m_min.array().min(m_min.array());
            m_max.array() = other.m_max.array().max(m_max.array());
        }

        ///
        /// \brief scale the mean and the stdev once the updates are done.
        ///
        auto& done(tensor_size_t total)
        {
            auto mean = m_mean.array();
            auto stdev = m_stdev.array();

            mean /= static_cast<scalar_t>(total);
            // cppcheck-suppress unreadVariable
            stdev = ((stdev - total * mean.array().square()) / std::max(total - 1, tensor_size_t(1))).sqrt();
            return *this;
        }

        ///
        /// \brief scale element-wise the given 4D tensor,
        ///     where the first dimension is the sample index and the rest being the elements/features to normalize.
        ///
        template <template <typename, size_t> class tstorage>
        void scale(normalization norm, tensor_t<tstorage, scalar_t, 4>& inputs) const
        {
            const auto samples = this->samples(inputs);
            const auto epsilon = epsilon2<scalar_t>();

            // FIXME: write these operations as single Eigen calls!
            switch (norm)
            {
            case normalization::none:
                break;

            case normalization::mean:
                for (tensor_size_t s = 0; s < samples; ++ s)
                {
                    inputs.array(s) = (inputs.array(s) - m_mean.array()) / (m_max.array() - m_min.array()).max(epsilon);
                }
                break;

            case normalization::minmax:
                for (tensor_size_t s = 0; s < samples; ++ s)
                {
                    inputs.array(s) = (inputs.array(s) - m_min.array()) / (m_max.array() - m_min.array()).max(epsilon);
                }
                break;

            case normalization::standard:
                for (tensor_size_t s = 0; s < samples; ++ s)
                {
                    inputs.array(s) = (inputs.array(s) - m_mean.array()) / (m_stdev.array()).max(epsilon);
                }
                break;

            default:
                throw std::runtime_error("unhandled normalization method when scaling inputs");
                break;
            }
        }

        ///
        /// \brief adjust the weights and the bias of a linear transformation to work with the
        ///     the un-scaled/un-normalized inputs.
        ///
        template <template <typename, size_t> class tstorage>
        void upscale(
            normalization norm, tensor_t<tstorage, scalar_t, 2>& weights, tensor_t<tstorage, scalar_t, 1>& bias) const
        {
            const auto epsilon = epsilon2<scalar_t>();

            auto w = weights.matrix();
            auto b = bias.vector();

            // FIXME: rewrite these operations to be more intelligible!
            switch (norm)
            {
            case ::nano::normalization::none:
                break;

            case ::nano::normalization::mean:
                // cppcheck-suppress unreadVariable
                b -= w.transpose() * (m_mean.array() / (m_max.array() - m_min.array()).max(epsilon)).matrix();
                w.array().colwise() /= (m_max.array() - m_min.array()).max(epsilon);
                break;

            case ::nano::normalization::minmax:
                // cppcheck-suppress unreadVariable
                b -= w.transpose() * (m_min.array() / (m_max.array() - m_min.array()).max(epsilon)).matrix();
                w.array().colwise() /= (m_max.array() - m_min.array()).max(epsilon);
                break;

            case ::nano::normalization::standard:
                // cppcheck-suppress unreadVariable
                b -= w.transpose() * (m_mean.array() / m_stdev.array().max(epsilon)).matrix();
                w.array().colwise() /= m_stdev.array().max(epsilon);
                break;

            default:
                assert(false);
                break;
            }
        }

        ///
        /// \brief access functions
        ///
        const auto& min() const { return m_min; }
        const auto& max() const { return m_max; }
        const auto& mean() const { return m_mean; }
        const auto& stdev() const { return m_stdev; }

    private:

        template <template <typename, size_t> class tstorage>
        auto samples(const tensor_t<tstorage, scalar_t, 4>& inputs) const
        {
            const auto samples = inputs.template size<0>();
            assert(nano::cat_dims(samples, m_min.dims()) == inputs.dims());
            assert(nano::cat_dims(samples, m_max.dims()) == inputs.dims());
            assert(nano::cat_dims(samples, m_mean.dims()) == inputs.dims());
            assert(nano::cat_dims(samples, m_stdev.dims()) == inputs.dims());
            return samples;
        }

        // attributes
        tensor3d_t  m_min;      ///< minimum
        tensor3d_t  m_max;      ///< maximum
        tensor3d_t  m_mean;     ///< average
        tensor3d_t  m_stdev;    ///< standard deviation
    };
}
