#pragma once

#include <nano/tensor/base.h>
#include <nano/tensor/vector.h>

namespace nano
{
    template <typename, size_t>
    class tensor_vector_storage_t;

    template <typename, size_t>
    class tensor_carray_storage_t;

    template <typename, size_t>
    class tensor_marray_storage_t;

    ///
    /// \brief tensor storage using an Eigen vector.
    /// NB: the tensor owns the allocated memory and as such it is resizable.
    ///
    template <typename tscalar, size_t trank>
    class tensor_vector_storage_t : public tensor_base_t<tscalar, trank>
    {
    public:

        using tbase = tensor_base_t<tscalar, trank>;

        using tbase::size;
        using tdims = typename tbase::tdims;

        tensor_vector_storage_t() = default;
        ~tensor_vector_storage_t() = default;
        tensor_vector_storage_t(const tensor_vector_storage_t&) = default;
        tensor_vector_storage_t(tensor_vector_storage_t&&) noexcept = default;
        tensor_vector_storage_t& operator=(const tensor_vector_storage_t&) = default;
        tensor_vector_storage_t& operator=(tensor_vector_storage_t&&) noexcept = default;

        template <typename... tsizes>
        explicit tensor_vector_storage_t(tsizes... dims) :
            tbase(make_dims(dims...)),
            m_data(size())
        {
        }

        explicit tensor_vector_storage_t(tdims dims) :
            tbase(std::move(dims)),
            m_data(size())
        {
        }

        explicit tensor_vector_storage_t(const tensor_carray_storage_t<tscalar, trank>& other) :
            tbase(other.dims()),
            m_data(map_vector(other.data(), other.size()))
        {
        }

        explicit tensor_vector_storage_t(const tensor_marray_storage_t<tscalar, trank>& other) :
            tbase(other.dims()),
            m_data(map_vector(other.data(), other.size()))
        {
        }

        tensor_vector_storage_t& operator=(const tensor_carray_storage_t<tscalar, trank>& other)
        {
            tensor_vector_t<tscalar> data = map_vector(other.data(), other.size());
            resize(other.dims());
            std::swap(data, m_data);
            return *this;
        }

        tensor_vector_storage_t& operator=(const tensor_marray_storage_t<tscalar, trank>& other)
        {
            tensor_vector_t<tscalar> data = map_vector(other.data(), other.size());
            resize(other.dims());
            std::swap(data, m_data);
            return *this;
        }

        template <typename... tsizes>
        void resize(tsizes... dims)
        {
            tbase::resize(make_dims(dims...));
            m_data.resize(size());
        }

        void resize(const tdims& dims)
        {
            tbase::resize(dims);
            m_data.resize(size());
        }

        auto data()
        {
            return m_data.data();
        }
        auto data() const
        {
            return m_data.data();
        }

    private:

        // attributes
        tensor_vector_t<tscalar>    m_data;         ///< store tensor as a 1D vector.
    };

    ///
    /// \brief tensor storage using a constant C-array.
    /// NB: the tensor doesn't own the allocated memory and as such it is not resizable.
    ///
    template <typename tscalar, size_t trank>
    class tensor_carray_storage_t : public tensor_base_t<tscalar, trank>
    {
    public:

        using tbase = tensor_base_t<tscalar, trank>;

        using tbase::size;
        using tdims = typename tbase::tdims;

        tensor_carray_storage_t() = default;
        ~tensor_carray_storage_t() = default;
        tensor_carray_storage_t(const tensor_carray_storage_t&) = default;
        tensor_carray_storage_t(tensor_carray_storage_t&&) noexcept = default;
        tensor_carray_storage_t& operator=(tensor_carray_storage_t&& other) noexcept = default;

        template <typename... tsizes>
        explicit tensor_carray_storage_t(const tscalar* data, tsizes... dims) :
            tbase(make_dims(dims...)),
            m_data(data)
        {
            assert(data != nullptr || !size());
        }

        explicit tensor_carray_storage_t(const tscalar* data, tdims dims) :
            tbase(std::move(dims)),
            m_data(data)
        {
            assert(data != nullptr || !size());
        }

        explicit tensor_carray_storage_t(const tensor_vector_storage_t<tscalar, trank>& other) :
            tbase(other.dims()),
            m_data(other.data())
        {
        }

        explicit tensor_carray_storage_t(const tensor_marray_storage_t<tscalar, trank>& other) :
            tbase(other.dims()),
            m_data(other.data())
        {
        }

        tensor_carray_storage_t& operator=(const tensor_vector_storage_t<tscalar, trank>& other) = delete;
        tensor_carray_storage_t& operator=(const tensor_carray_storage_t<tscalar, trank>& other) = delete;
        tensor_carray_storage_t& operator=(const tensor_marray_storage_t<tscalar, trank>& other) = delete;

        template <typename... tsizes>
        void resize(tsizes...) = delete;
        void resize(const tdims&) = delete;

        auto data() const
        {
            return m_data;
        }

    private:

        // attributes
        const tscalar*      m_data{nullptr};    ///< wrap tensor over a contiguous array.
    };

    ///
    /// \brief tensor storage using a mutable C-array.
    /// NB: the tensor doesn't own the allocated memory and as such it is not resizable.
    ///
    template <typename tscalar, size_t trank>
    class tensor_marray_storage_t : public tensor_base_t<tscalar, trank>
    {
    public:

        using tbase = tensor_base_t<tscalar, trank>;

        using tbase::size;
        using tdims = typename tbase::tdims;

        tensor_marray_storage_t() = default;
        ~tensor_marray_storage_t() = default;
        tensor_marray_storage_t(const tensor_marray_storage_t&) = default;
        tensor_marray_storage_t(tensor_marray_storage_t&&) noexcept = default;
        tensor_marray_storage_t& operator=(tensor_marray_storage_t&& other) noexcept = default;

        template <typename... tsizes>
        explicit tensor_marray_storage_t(tscalar* data, tsizes... dims) :
            tbase(make_dims(dims...)),
            m_data(data)
        {
            assert(data != nullptr || !size());
        }

        explicit tensor_marray_storage_t(tscalar* data, tdims dims) :
            tbase(std::move(dims)),
            m_data(data)
        {
            assert(data != nullptr || !size());
        }

        explicit tensor_marray_storage_t(tensor_vector_storage_t<tscalar, trank>& other) :
            tbase(other.dims()),
            m_data(other.data())
        {
        }

        tensor_marray_storage_t& operator=(const tensor_vector_storage_t<tscalar, trank>& other)
        {
            copy(other);
            return *this;
        }

        tensor_marray_storage_t& operator=(const tensor_carray_storage_t<tscalar, trank>& other)
        {
            copy(other);
            return *this;
        }

        tensor_marray_storage_t& operator=(const tensor_marray_storage_t<tscalar, trank>& other) // NOLINT(cert-oop54-cpp,bugprone-unhandled-self-assignment)
        {
            copy(other);
            return *this;
        }

        template <typename... tsizes>
        void resize(tsizes...) = delete;
        void resize(const tdims&) = delete;

        auto data() const
        {
            return m_data;
        }

    private:

        template <typename tstorage>
        void copy(const tstorage& other)
        {
            assert(size() == other.size());
            map_vector(data(), size()) = map_vector(other.data(), other.size());
        }

        // attributes
        tscalar*            m_data{nullptr};    ///< wrap tensor over a contiguous array.
    };
}
