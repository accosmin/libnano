#pragma once

#include <nano/dataset/csv.h>
#include <nano/dataset/memfixed.h>

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
    class NANO_PUBLIC tabular_dataset_t : public memfixed_dataset_t<scalar_t>
    {
    public:

        using memfixed_dataset_t::target;
        using memfixed_dataset_t::features;

        ///
        /// \brief default constructor
        ///
        tabular_dataset_t() = default;

        ///
        /// \brief constructor, set the CSV files to load and describe their input and target features.
        ///
        tabular_dataset_t(csvs_t, features_t, size_t target = string_t::npos);

        ///
        /// \brief @see dataset_t
        ///
        void load() override;

        ///
        /// \brief @see dataset_t
        ///
        feature_t feature(tensor_size_t index) const override;

        ///
        /// \brief @see dataset_t
        ///
        feature_t target() const override;

    protected:

        void store(tensor_size_t row, size_t col, scalar_t value);
        void store(tensor_size_t row, size_t col, tensor_size_t category);
        bool parse(const csv_t&, const string_t&, tensor_size_t, tensor_size_t);

    private:

        // attributes
        csvs_t      m_csvs;                     ///< describes the CSV files
        features_t  m_features;                 ///< describes the columns in the CSV files (aka the features)
        size_t      m_target{string_t::npos};   ///< index of the target column (if negative, then not provided)
    };
}
