#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief common class for MNIST-like datasets.
    ///
    class NANO_PUBLIC base_mnist_dataset_t : public dataset_t
    {
    public:

        ///
        /// \brief constructor
        ///
        base_mnist_dataset_t(string_t dir, string_t name, feature_t target);

        ///
        /// \brief @see dataset_t
        ///
        void load() override;

    private:

        bool iread(const string_t&, tensor_size_t offset, tensor_size_t expected);
        bool tread(const string_t&, tensor_size_t offset, tensor_size_t expected);

        // attributes
        string_t        m_dir;              ///< directory where to load the data from
        string_t        m_name;             ///< dataset name
        feature_t       m_target;           ///< target feature
    };

    ///
    /// MNIST dataset:
    ///      - classify digits
    ///      - 28x28 grayscale images as inputs
    ///      - 10 outputs (10 labels)
    ///
    /// http://yann.lecun.com/exdb/mnist/
    ///
    class NANO_PUBLIC mnist_dataset_t final : public base_mnist_dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        mnist_dataset_t();
    };

    ///
    /// fashion-MNIST dataset:
    ///      - classify fashion articles
    ///      - 28x28 grayscale images as inputs
    ///      - 10 outputs (10 labels)
    ///
    /// https://github.com/zalandoresearch/fashion-mnist
    ///
    class NANO_PUBLIC fashion_mnist_dataset_t final : public base_mnist_dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        fashion_mnist_dataset_t();
    };
}
