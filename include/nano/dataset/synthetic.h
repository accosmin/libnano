#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief synthetic dataset:
    ///     the targets is a random affine transformation of the flatten input features.
    ///
    /// NB: uniformly-distributed noise is added to targets if noise() > 0.
    /// NB: every (sample index + feature index) % modulo() feature value is missing.
    ///
    class NANO_PUBLIC synthetic_affine_dataset_t : public dataset_t
    {
    public:

        using dataset_t::features;
        using dataset_t::samples;

        ///
        /// \brief default constructor
        ///
        synthetic_affine_dataset_t();

        ///
        /// \brief @see dataset_t
        ///
        void load() override;

        ///
        /// \brief change parameters
        ///
        void noise(scalar_t noise) { m_noise = noise; }
        void modulo(tensor_size_t modulo) { m_modulo = modulo; }
        void samples(tensor_size_t samples) { m_samples = samples; }
        void targets(tensor_size_t targets) { m_targets = targets; }
        void features(tensor_size_t features) { m_features = features; }

        ///
        /// \brief access functions
        ///
        auto noise() const { return m_noise; }
        const auto& bias() const { return m_bias; }
        const auto& weights() const { return m_weights; }

    private:

        // attributes
        scalar_t            m_noise{0};         ///< noise level (relative to the [-1,+1] uniform distribution)
        tensor_size_t       m_modulo{71};       ///<
        tensor_size_t       m_targets{3};       ///< number of targets
        tensor_size_t       m_features{10};     ///< total number of features to generate, of various types
        tensor_size_t       m_samples{1000};    ///< total number of samples to generate (train + validation + test)
        tensor2d_t          m_weights;          ///< 2D weight matrix that maps the input to the output
        tensor1d_t          m_bias;             ///< 1D bias vector that offsets the output
    };
}
