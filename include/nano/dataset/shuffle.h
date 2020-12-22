#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief wrapper over a dataset to shuffle (across samples) a given feature.
    ///
    /// NB: this is useful for estimating the importance of a feature by measuring
    ///     the difference in accuracy when the associated feature values are shuffled (across samples).
    ///
    class shuffle_dataset_t : public dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        shuffle_dataset_t(const dataset_t& source, tensor_size_t feature2shuffle) :
            m_source(source),
            m_feature2shuffle(feature2shuffle)
        {
            assert(feature2shuffle >= 0 && feature2shuffle < m_source.features());
        }

        ///
        /// \brief @see dataset_t
        ///
        void load() override
        {
        }

        ///
        /// \brief @see dataset_t
        ///
        feature_t feature(const tensor_size_t index) const override
        {
            return m_source.feature(index);
        }

        ///
        /// \brief @see dataset_t
        ///
        feature_t target() const override
        {
            return m_source.target();
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor_size_t samples() const override
        {
            return m_source.samples();
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor4d_t inputs(const indices_cmap_t& samples) const override
        {
            auto inputs = m_source.inputs(samples);
            shuffle(inputs, m_feature2shuffle);
            return inputs;
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor1d_t inputs(const indices_cmap_t& samples, tensor_size_t feature) const override
        {
            auto inputs = m_source.inputs(samples, feature);
            if (m_feature2shuffle == feature)
            {
                shuffle(inputs, 0);
            }
            return inputs;
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor2d_t inputs(const indices_cmap_t& samples, const indices_t& features) const override
        {
            auto inputs = m_source.inputs(samples, features);
            const auto* const it = std::find(features.begin(), features.end(), m_feature2shuffle);
            if (it != features.end())
            {
                shuffle(inputs, std::distance(features.begin(), it));
            }
            return inputs;
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor4d_t targets(const indices_cmap_t& samples) const override
        {
            return m_source.targets(samples);
        }

        ///

        ///
        /// \brief @see dataset_t
        ///
        tensor3d_dims_t idims() const override
        {
            return m_source.idims();
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor3d_dims_t tdims() const override
        {
            return m_source.tdims();
        }

    private:

        template <typename ttensor>
        void shuffle(ttensor& inputs, tensor_size_t col) const
        {
            auto matrix = inputs.reshape(inputs.template size<0>(), -1);
            assert(col >= 0 && col < matrix.cols());

            auto&& rng = ::nano::make_rng();
            using diff_t = tensor_size_t;
            using dist_t = std::uniform_int_distribution<diff_t>;
            using param_t = dist_t::param_type;

            dist_t D;
            for (diff_t rows = matrix.rows(), row = rows - 1; row > 0; -- row)
            {
                using std::swap;
                swap(matrix(row, col), matrix(D(rng, param_t(0, row)), col));
            }
        }

        // attributes
        const dataset_t&    m_source;               ///< original dataset
        tensor_size_t       m_feature2shuffle{0};   ///< feature index to shuffle (across samples)
    };
}
