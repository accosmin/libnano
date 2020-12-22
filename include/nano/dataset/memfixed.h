#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief in-memory dataset consisting of fixed-size inputs with optional targets.
    ///
    /// NB: the internal storage type can be different than scalar_t,
    ///     for example the most efficient way of storing RGB or grayscale image datasets
    ///     is to use one byte per color channel and pixel.
    ///
    /// NB: the customization point (in the derived classes) consists
    ///     of generating/loading the inputs and the targets.
    ///
    template <typename tscalar>
    class memfixed_dataset_t : public dataset_t
    {
    public:

        using dataset_t::target;

        ///
        /// \brief default constructor
        ///
        memfixed_dataset_t() = default;

        ///
        /// \brief @see dataset_t
        ///
        feature_t feature(const tensor_size_t index) const override
        {
            const auto idims = this->idims();
            assert(index >= 0 && ::nano::size(idims));

            const auto modx = std::get<1>(idims) * std::get<2>(idims);
            const auto mody = std::get<2>(idims);
            const auto x = index / modx;
            const auto y = (index - x * modx) / mody;
            const auto z = index - x * modx - y * mody;

            feature_t feature{scat("feature_", x, "_", y, "_", z)};
            return feature;
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor_size_t samples() const override
        {
            return m_inputs.template size<0>();
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor4d_t inputs(const indices_cmap_t& samples) const override
        {
            return m_inputs.template indexed<scalar_t>(samples);
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor1d_t inputs(const indices_cmap_t& samples, tensor_size_t feature) const override
        {
            assert(feature >= 0 && feature < features());

            const auto imatrix = m_inputs.reshape(this->samples(), features()).matrix();

            tensor1d_t fvalues(samples.size());
            for (tensor_size_t i = 0, size = samples.size(); i < size; ++ i)
            {
                fvalues(i) = imatrix(samples(i), feature);
            }

            return fvalues;
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor2d_t inputs(const indices_cmap_t& samples, const indices_t& features) const override
        {
            assert(features.min() >= 0 && features.max() < this->features());

            const auto imatrix = m_inputs.reshape(this->samples(), this->features()).matrix();

            tensor2d_t fvalues(samples.size(), features.size());
            for (tensor_size_t i = 0, size = samples.size(); i < size; ++ i)
            {
                for (tensor_size_t f = 0; f < features.size(); ++ f)
                {
                    fvalues(i, f) = imatrix(samples(i), features(f));
                }
            }

            return fvalues;
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor4d_t targets(const indices_cmap_t& samples) const override
        {
            return m_targets.indexed<scalar_t>(samples);
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor3d_dims_t idims() const override
        {
            return make_dims(m_inputs.template size<1>(), m_inputs.template size<2>(), m_inputs.template size<3>());
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor3d_dims_t tdims() const override
        {
            return make_dims(m_targets.size<1>(), m_targets.size<2>(), m_targets.size<3>());
        }

        ///
        /// \brief returns the inputs and targets as they are stored.
        ///
        const auto& all_inputs() const { return m_inputs; }
        const auto& all_targets() const { return m_targets; }

        ///
        /// \brief returns the mutable input and target sample.
        ///
        auto input(tensor_size_t sample) { return m_inputs.tensor(sample); }
        auto target(tensor_size_t sample) { return m_targets.tensor(sample); }

        ///
        /// \brief returns the constant input and target sample.
        ///
        auto input(tensor_size_t sample) const { return m_inputs.tensor(sample); }
        auto target(tensor_size_t sample) const { return m_targets.tensor(sample); }

    protected:

        ///
        /// \brief allocate input and target tensors
        ///
        void resize(const tensor4d_dims_t& idims, const tensor4d_dims_t& tdims)
        {
            assert(std::get<0>(idims) == std::get<0>(tdims));

            m_inputs.resize(idims);
            m_targets.resize(tdims);
        }

    private:

        // attributes
        tensor_mem_t<tscalar, 4>    m_inputs;       ///< (total number of samples, #idim1, #idim2, #idim3)
        tensor_mem_t<scalar_t, 4>   m_targets;      ///< (total number of samples, #tdim1, #tdim2, #tdim3)
    };
}
