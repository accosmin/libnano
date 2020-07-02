#pragma once

#include <nano/arch.h>
#include <nano/factory.h>
#include <nano/dataset/csv.h>
#include <nano/dataset/memfixed.h>

namespace nano
{
    class tabular_dataset_t;
    using tabular_dataset_factory_t = factory_t<tabular_dataset_t>;
    using rtabular_dataset_t = tabular_dataset_factory_t::trobject;

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

        ///
        /// \brief returns the available implementations
        ///
        static tabular_dataset_factory_t& all();

        ///
        /// \brief default constructor
        ///
        tabular_dataset_t() = default;

        ///
        /// \brief populate the dataset with samples
        ///
        bool load() override;

        ///
        /// \brief returns the total number of input features
        ///
        [[nodiscard]] size_t ifeatures() const;

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] feature_t ifeature(tensor_size_t index) const override;

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] feature_t tfeature() const override;

        ///
        /// \brief set the CSV files to load
        ///
        void csvs(std::vector<csv_t>);

        ///
        /// \brief set the input and the target features
        ///
        void features(std::vector<feature_t>, size_t target = string_t::npos);

        ///
        /// \brief generate a split into training, validation and test.
        ///
        [[nodiscard]] virtual split_t make_split() const = 0;

    protected:

        void store(tensor_size_t row, size_t col, scalar_t value);
        void store(tensor_size_t row, size_t col, tensor_size_t category);
        bool parse(const string_t&, const string_t&, const string_t&, tensor_size_t, tensor_size_t);

    private:

        // attributes
        csvs_t      m_csvs;                     ///< describes the CSV files
        features_t  m_features;                 ///< describes the columns in the CSV files (aka the features)
        size_t      m_target{string_t::npos};   ///< index of the target column (if negative, then not provided)
    };
}