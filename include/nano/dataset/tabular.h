#pragma once

#include <nano/dataset.h>
#include <nano/dataset/csv.h>

namespace nano
{
    ///
    /// \brief machine learning dataset consisting of samples loaded from CSV files (aka tabular data).
    ///
    /// the tabular dataset is versatile:
    ///     - the target is optional, so it can address both supervised and unsupervised machine learning tasks
    ///     - the inputs can be both categorical and continuous
    ///     - missing feature values are supported
    ///
    class NANO_PUBLIC tabular_dataset_t : public dataset_t
    {
    public:

        ///
        /// \brief constructor, set the CSV files to load and the input features.
        ///
        tabular_dataset_t(csvs_t, features_t);

        ///
        /// \brief constructor, set the CSV files to load and the input and the target features.
        ///
        tabular_dataset_t(csvs_t, features_t, size_t target);

        ///
        /// \brief @see dataset_t
        ///
        void load() override;

    private:

        bool parse(const csv_t&, const string_t&, tensor_size_t, tensor_size_t);

        // attributes
        csvs_t      m_csvs;                     ///< describes the CSV files
        features_t  m_features;                 ///< describes the columns in the CSV files (aka the features)
        size_t      m_target{string_t::npos};   ///< index of the target column (if negative, then not provided)
    };
}
