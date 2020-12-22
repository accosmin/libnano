#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief wrapper over a dataset to remove a given feature.
    ///
    /// NB: this is useful for estimating the importance of a feature by measuring
    ///     the difference in accuracy when that particular feature is removed from training.
    ///
    class dropcol_dataset_t : public dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        dropcol_dataset_t(const dataset_t& source, tensor_size_t feature2coldrop) :
            m_source(source),
            m_feature2coldrop(feature2coldrop)
        {
            assert(feature2coldrop >= 0 && feature2coldrop < m_source.features());
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
            assert(index >= 0 && index < ::nano::size(idims()));
            return m_source.feature(reindex(index));
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
            return dropcol(inputs, m_feature2coldrop);
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor1d_t inputs(const indices_cmap_t& samples, tensor_size_t feature) const override
        {
            return m_source.inputs(samples, reindex(feature));
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor2d_t inputs(const indices_cmap_t& samples, const indices_t& features) const override
        {
            indices_t dfeatures(features.size());
            std::transform(features.begin(), features.end(), dfeatures.begin(), [&] (tensor_size_t feature)
            {
                return reindex(feature);
            });
            return m_source.inputs(samples, dfeatures);
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor4d_t targets(const indices_cmap_t& samples) const override
        {
            return m_source.targets(samples);
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor3d_dims_t idims() const override
        {
            return make_dims(nano::size(m_source.idims()) - 1, 1, 1);
        }

        ///
        /// \brief @see dataset_t
        ///
        tensor3d_dims_t tdims() const override
        {
            return m_source.tdims();
        }

    private:

        tensor_size_t reindex(tensor_size_t feature) const
        {
            return (feature < m_feature2coldrop) ? (feature) : (feature + 1);
        }

        template <typename tinputs, typename tdinputs>
        void skipcol(const tinputs& inputs, tensor_size_t col, tdinputs& dinputs) const
        {
            const auto matrix = inputs.reshape(inputs.template size<0>(), -1).matrix();
            auto dmatrix = dinputs.reshape(dinputs.template size<0>(), -1).matrix();

            assert(col >= 0 && col < matrix.cols());
            assert(matrix.rows() == dmatrix.rows());
            assert(matrix.cols() == dmatrix.cols() + 1);

            for (tensor_size_t row = 0, rows = matrix.rows(), cols = matrix.cols(); row < rows; ++ row)
            {
                dmatrix.row(row).segment(0, col) = matrix.row(row).segment(0, col);
                dmatrix.row(row).segment(col, cols - col - 1) = matrix.row(row).segment(col + 1, cols - col - 1);
            }
        }

        tensor2d_t dropcol(const tensor2d_t& inputs, tensor_size_t col) const
        {
            tensor2d_t dinputs(inputs.size<0>(), inputs.size() / inputs.size<0>() - 1);
            skipcol(inputs, col, dinputs);
            return dinputs;
        }

        tensor4d_t dropcol(const tensor4d_t& inputs, tensor_size_t col) const
        {
            tensor4d_t dinputs(inputs.size<0>(), inputs.size() / inputs.size<0>() - 1, 1, 1);
            skipcol(inputs, col, dinputs);
            return dinputs;
        }

        // attributes
        const dataset_t&    m_source;               ///< original dataset
        tensor_size_t       m_feature2coldrop{0};   ///< feature index to remove
    };
}
