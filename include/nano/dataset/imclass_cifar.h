#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief base class for the CIFAR-10 and CIFAR-100 datasets.
    ///
    class NANO_PUBLIC cifar_dataset_t : public dataset_t
    {
    public:

        ///
        /// \brief constructor
        ///
        cifar_dataset_t(string_t dir, string_t name, feature_t target);

        ///
        /// \brief @see imclass_dataset_t
        ///
        void load() override;

    protected:

        void file(string_t filename, tensor_size_t, tensor_size_t, tensor_size_t, tensor_size_t);

    private:

        ///
        /// \brief describes how to load a file in the CIFAR-10/100 archives.
        ///
        struct file_t
        {
            file_t() = default;
            file_t(string_t&& filename,
                tensor_size_t offset, tensor_size_t expected,
                tensor_size_t label_size, tensor_size_t label_index) :
                m_filename(filename),
                m_offset(offset),
                m_expected(expected),
                m_label_size(label_size),
                m_label_index(label_index)
            {
            }

            // attributes
            string_t        m_filename;         ///<
            tensor_size_t   m_offset{0};        ///<
            tensor_size_t   m_expected{0};      ///<
            tensor_size_t   m_label_size{1};    ///<
            tensor_size_t   m_label_index{0};   ///<
        };
        using files_t = std::vector<file_t>;

        bool iread(const file_t& file);

        // attributes
        string_t        m_dir;              ///< directory where to load the data from
        string_t        m_path;             ///< path to the archive where to load the data from
        string_t        m_name;             ///< dataset name
        feature_t       m_target;           ///< target feature
        files_t         m_files;            ///<
    };

    ///
    /// CIFAR-10 task:
    ///      - image classification
    ///      - 32x32 color images as inputs
    ///      - 10 outputs (10 labels)
    ///
    /// http://www.cs.toronto.edu/~kriz/cifar.html
    ///
    class NANO_PUBLIC cifar10_dataset_t final : public cifar_dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        cifar10_dataset_t();
    };

    ///
    /// CIFAR-100 task:
    ///      - image classification
    ///      - 32x32 color images as inputs
    ///      - 20 outputs (20 coarse labels)
    ///
    /// http://www.cs.toronto.edu/~kriz/cifar.html
    ///
    class NANO_PUBLIC cifar100c_dataset_t final : public cifar_dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        cifar100c_dataset_t();
    };

    ///
    /// CIFAR-100 task:
    ///      - image classification
    ///      - 32x32 color images as inputs
    ///      - 100 outputs (100 fine labels)
    ///
    /// http://www.cs.toronto.edu/~kriz/cifar.html
    ///
    class NANO_PUBLIC cifar100f_dataset_t final : public cifar_dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        cifar100f_dataset_t();
    };
}
