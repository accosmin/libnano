#pragma once

#include <nano/memfixed.h>

namespace nano
{
    ///
    /// \brief synthetic dataset:
    ///     the targets is a random affine transformation of the inputs.
    ///
    /// NB: only the inputs modulo a given constant are taken into account,
    ///     e.g. to test feature selection.
    ///
    /// NB: uniformly-distributed noise is added to targets if noise() > 0.
    ///
    class synthetic_affine_dataset_t final : public memfixed_dataset_t<scalar_t>
    {
    public:

        using memfixed_dataset_t::samples;

        ///
        /// \brief default constructor
        ///
        synthetic_affine_dataset_t() = default;

        ///
        /// \brief @see memfixed_dataset_t
        ///
        bool load() override
        {
            // create fixed random bias and weights
            m_bias = vector_t::Random(size(m_odim));
            m_weights = matrix_t::Zero(size(m_odim), size(m_idim));
            for (tensor_size_t i = 0, isize = m_weights.cols(); i < isize; i += m_modulo)
            {
                m_weights.col(i).setRandom();
            }

            // create samples: target = weights * input + bias + noise
            resize(cat_dims(m_samples, m_idim), cat_dims(m_samples, m_odim));
            for (tensor_size_t s = 0; s < m_samples; ++ s)
            {
                input(s).random();
                target(s).vector() = m_weights * input(s).vector() + m_bias + m_noise * vector_t::Random(m_bias.size());
            }

            // create folds
            for (size_t f = 0; f < folds(); ++ f)
            {
                this->split(f) = split_t{
                    nano::split3(m_samples, train_percentage(), (100 - train_percentage()) / 2)
                };
            }

            return true;
        }

        ///
        /// \brief @see memfixed_dataset_t
        ///
        feature_t tfeature() const override
        {
            return feature_t{"Wx+b"};
        }

        ///
        /// \brief change parameters
        ///
        void noise(const scalar_t noise) { m_noise = noise; }
        void idim(const tensor3d_dim_t idim) { m_idim = idim; }
        void odim(const tensor3d_dim_t odim) { m_odim = odim; }
        void modulo(const tensor_size_t modulo) { m_modulo = modulo; }
        void samples(const tensor_size_t samples) { m_samples = samples; }

        ///
        /// \brief access functions
        ///
        auto noise() const { return m_noise; }
        auto modulo() const { return m_modulo; }
        const auto& idim() const { return m_idim; }
        const auto& odim() const { return m_odim; }
        const auto& bias() const { return m_bias; }
        const auto& weights() const { return m_weights; }

    private:

        // attributes
        scalar_t            m_noise{0};         ///< noise level (relative to the [-1,+1] uniform distribution)
        tensor_size_t       m_modulo{1};        ///<
        tensor_size_t       m_samples{1000};    ///< total number of samples to generate (train + validation + test)
        tensor3d_dim_t      m_idim{{10, 1, 1}}; ///< dimension of an input sample
        tensor3d_dim_t      m_odim{{3, 1, 1}};  ///< dimension of a target/output sample
        matrix_t            m_weights;          ///< 2D weight matrix that maps the input to the output
        vector_t            m_bias;             ///< 1D bias vector that offsets the output
    };
}
