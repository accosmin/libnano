#pragma once

#include <nano/random.h>
#include <nano/numeric.h>
#include <nano/tensor/vector.h>
#include <nano/tensor/matrix.h>
#include <nano/tensor/storage.h>

#include <iostream>

namespace nano
{
    template <template <typename, size_t> class, typename, size_t>
    class tensor_t;

    ///
    /// \brief tensor that owns the allocated memory.
    ///
    template <typename tscalar, size_t trank>
    using tensor_mem_t = tensor_t<tensor_vector_storage_t, tscalar, trank>;

    ///
    /// \brief tensor mapping a non-constant array.
    ///
    template <typename tscalar, size_t trank>
    using tensor_map_t = tensor_t<tensor_marray_storage_t, tscalar, trank>;

    ///
    /// \brief tensor mapping a constant array.
    ///
    template <typename tscalar, size_t trank>
    using tensor_cmap_t = tensor_t<tensor_carray_storage_t, tscalar, trank>;

    ///
    /// \brief tensor indices.
    ///
    using indices_t = tensor_mem_t<tensor_size_t, 1>;
    using indices_map_t = tensor_map_t<tensor_size_t, 1>;
    using indices_cmap_t = tensor_cmap_t<tensor_size_t, 1>;

    ///
    /// \brief map non-constant data to tensors.
    ///
    template <typename tscalar, size_t trank>
    auto map_tensor(tscalar* data, const tensor_dims_t<trank>& dims)
    {
        return tensor_map_t<tscalar, trank>(data, dims);
    }

    ///
    /// \brief map constant data to tensors.
    ///
    template <typename tscalar, size_t trank>
    auto map_tensor(const tscalar* data, const tensor_dims_t<trank>& dims)
    {
        return tensor_cmap_t<tscalar, trank>(data, dims);
    }

    ///
    /// \brief map non-constant data to tensors.
    ///
    template <typename tscalar, typename... tsizes>
    auto map_tensor(tscalar* data, tsizes... dims)
    {
        return map_tensor(data, make_dims(dims...));
    }

    ///
    /// \brief map constant data to tensors.
    ///
    template <typename tscalar, typename... tsizes>
    auto map_tensor(const tscalar* data, tsizes... dims)
    {
        return map_tensor(data, make_dims(dims...));
    }

    ///
    /// \brief tensor w/o owning the allocated continuous memory.
    ///
    /// NB: all access operations (e.g. Eigen arrays, vectors or matrices or sub-tensors) are performed
    ///     using only continuous memory.
    ///
    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    class tensor_t : public tstorage<tscalar, trank>
    {
    public:

        using tbase = tstorage<tscalar, trank>;

        using tbase::data;
        using tbase::dims;
        using tbase::size;
        using tbase::rank;
        using tbase::rows;
        using tbase::cols;
        using tbase::offset;
        using tbase::offset0;
        using tdims = typename tbase::tdims;

        ///
        /// \brief default constructor
        ///
        tensor_t() = default;

        ///
        /// \brief construct to match the given size.
        ///
        template <typename... tsizes>
        explicit tensor_t(tsizes... dims) :
            tbase(make_dims(dims...))
        {
        }

        explicit tensor_t(tdims dims) :
            tbase(std::move(dims))
        {
        }

        ///
        /// \brief map non-ownning mutable C-arrays.
        ///
        explicit tensor_t(tscalar* ptr, tdims dims) :
            tbase(ptr, std::move(dims))
        {
        }

        ///
        /// \brief map non-ownning constant C-arrays.
        ///
        explicit tensor_t(const tscalar* ptr, tdims dims) :
            tbase(ptr, std::move(dims))
        {
        }

        ///
        /// \brief enable copying (delegate to the storage object).
        ///
        tensor_t(const tensor_t&) = default;
        tensor_t& operator=(const tensor_t&) = default;

        ///
        /// \brief enable moving (delegate to the storage object).
        ///
        tensor_t(tensor_t&&) noexcept = default;
        tensor_t& operator=(tensor_t&&) noexcept = default;

        ///
        /// \brief copy constructor from different types (e.g. const array from mutable array)
        ///
        template <template <typename, size_t> class tstorage2>
        // cppcheck-suppress noExplicitConstructor
        tensor_t(const tensor_t<tstorage2, tscalar, trank>& other) : // NOLINT(hicpp-explicit-conversions)
            tbase(static_cast<const tstorage2<tscalar, trank>&>(other))
        {
        }

        ///
        /// \brief copy constructor from different types (e.g. mutable array from mutable vector)
        ///
        template <template <typename, size_t> class tstorage2>
        // cppcheck-suppress noExplicitConstructor
        tensor_t(tensor_t<tstorage2, tscalar, trank>& other) : // NOLINT(hicpp-explicit-conversions)
            tbase(static_cast<tstorage2<tscalar, trank>&>(other))
        {
        }

        ///
        /// \brief assignment operator from different types (e.g. const from non-const scalars)
        ///
        template <template <typename, size_t> class tstorage2>
        tensor_t& operator=(const tensor_t<tstorage2, tscalar, trank>& other)
        {
            tbase::operator=(other);
            return *this;
        }

        ///
        /// \brief default destructor
        ///
        ~tensor_t() = default;

        ///
        /// \brief resize to new dimensions.
        ///
        template <typename... tsizes>
        void resize(tsizes... dims)
        {
            tbase::resize(make_dims(dims...));
        }

        void resize(const tdims& dims)
        {
            tbase::resize(dims);
        }

        ///
        /// \brief set all elements to zero.
        ///
        void zero()
        {
            array() = tscalar(0);
        }

        ///
        /// \brief set all elements to the given constant value.
        ///
        void full(tscalar value)
        {
            array() = value;
        }

        ///
        /// \brief set all elements to random values in the [min, max] range.
        ///
        void random(const tscalar min = -1, const tscalar max = +1)
        {
            random(array(), min, max);
        }

        ///
        /// \brief set all elements in an arithmetic progression from min to max.
        ///
        void lin_spaced(const tscalar min, const tscalar max)
        {
            array() = tensor_vector_t<tscalar>::LinSpaced(size(), min, max);
        }

        ///
        /// \brief returns the minimum value.
        ///
        auto min() const
        {
            return vector().minCoeff();
        }

        ///
        /// \brief returns the maximum value.
        ///
        auto max() const
        {
            return vector().maxCoeff();
        }

        ///
        /// \brief returns the average value.
        ///
        auto mean() const
        {
            return vector().mean();
        }

        ///
        /// \brief returns the sum of all its values.
        ///
        auto sum() const
        {
            return vector().sum();
        }

        ///
        /// \brief returns the variance of all its values.
        ///
        auto variance() const
        {
            double variance = 0.0;
            if (size() > 1)
            {
                const auto array = this->array().template cast<double>();
                const auto count = static_cast<double>(size());
                const auto average = array.mean();
                variance = array.square().sum() / count - average * average;
            }
            return variance;
        }

        ///
        /// \brief returns the sample standard deviation.
        ///
        auto stdev() const
        {
            double stdev = 0.0;
            if (size() > 1)
            {
                const auto count = static_cast<double>(size());
                stdev = std::sqrt(variance() / (count - 1));
            }
            return stdev;
        }

        ///
        /// \brief access the whole tensor as an Eigen vector
        ///
        auto vector() { return map_vector(data(), size()); }
        auto vector() const { return map_vector(data(), size()); }

        ///
        /// \brief access a continuous part of the tensor as an Eigen vector
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto vector(const tindices... indices) { return tvector(data(), indices...); }

        template <typename... tindices>
        auto vector(const tindices... indices) const { return tvector(data(), indices...); }

        ///
        /// \brief access the whole tensor as an Eigen array
        ///
        auto array() { return vector().array(); }
        auto array() const { return vector().array(); }

        ///
        /// \brief access a part of the tensor as an Eigen array
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto array(const tindices... indices) { return vector(indices...).array(); }

        template <typename... tindices>
        auto array(const tindices... indices) const { return vector(indices...).array(); }

        ///
        /// \brief access a part of the tensor as an Eigen matrix
        ///     (assuming that the last two dimensions are ignored)
        ///
        template <typename... tindices>
        auto matrix(const tindices... indices) { return tmatrix(data(), indices...); }

        template <typename... tindices>
        auto matrix(const tindices... indices) const { return tmatrix(data(), indices...); }

        ///
        /// \brief access a part of the tensor as a (sub-)tensor
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto tensor(const tindices... indices) { return ttensor(data(), indices...); }

        template <typename... tindices>
        auto tensor(const tindices... indices) const { return ttensor(data(), indices...); }

        ///
        /// \brief access a part of the tensor as a (sub-)tensor
        ///     (by taking the [begin, end) range of the first dimension)
        ///
        auto slice(tensor_size_t begin, tensor_size_t end)
        {
            return tslice(data(), begin, end);
        }

        auto slice(tensor_size_t begin, tensor_size_t end) const
        {
            return tslice(data(), begin, end);
        }

        auto slice(const tensor_range_t& range)
        {
            return tslice(data(), range.begin(), range.end());
        }

        auto slice(const tensor_range_t& range) const
        {
            return tslice(data(), range.begin(), range.end());
        }

        ///
        /// \brief copy some of (sub-)tensors using the given indices.
        /// NB: the indices are relative to the first dimension.
        ///
        template <typename tscalar_return = tscalar>
        void indexed(indices_cmap_t indices, tensor_mem_t<tscalar_return, trank>& subtensor) const
        {
            assert(indices.min() >= 0 && indices.max() < this->template size<0>());

            auto dims = this->dims();
            dims[0] = indices.size();

            subtensor.resize(dims);
            if constexpr (trank > 1)
            {
                for (tensor_size_t i = 0, indices_size = indices.size(); i < indices_size; ++ i)
                {
                    subtensor.vector(i) = vector(indices(i)).template cast<tscalar_return>();
                }
            }
            else
            {
                for (tensor_size_t i = 0, indices_size = indices.size(); i < indices_size; ++ i)
                {
                    subtensor(i) = static_cast<tscalar_return>(this->operator()(indices(i)));
                }
            }
        }

        ///
        /// \brief returns a copy of some (sub-)tensors using the given indices.
        /// NB: the indices are relative to the first dimension.
        ///
        template <typename tscalar_return = tscalar>
        auto indexed(indices_cmap_t indices) const
        {
            auto subtensor = tensor_mem_t<tscalar_return, trank>{};
            indexed(indices, subtensor);
            return subtensor;
        }

        ///
        /// \brief access an element of the tensor
        ///
        typename tbase::tmutableref operator()(const tensor_size_t index)
        {
            assert(const_cast<const tensor_t*>(this)->data() != nullptr);
            assert(index >= 0 && index < size());
            return data()[index];
        }

        typename tbase::tconstref operator()(const tensor_size_t index) const
        {
            assert(data() != nullptr);
            assert(index >= 0 && index < size());
            return data()[index];
        }

        template <typename... tindices>
        typename tbase::tmutableref operator()(const tensor_size_t index, const tindices... indices)
        {
            return operator()(offset(index, indices...));
        }

        template <typename... tindices>
        typename tbase::tconstref operator()(const tensor_size_t index, const tindices... indices) const
        {
            return operator()(offset(index, indices...));
        }

        ///
        /// \brief reshape to a new tensor (with the same number of elements).
        /// NB: a single -1 dimension can be inferred from the total size and the remaining positive dimensions!
        ///
        template <typename... tsizes>
        auto reshape(tsizes... sizes)
        {
            return treshape(data(), sizes...);
        }

        template <typename... tsizes>
        auto reshape(tsizes... sizes) const
        {
            return treshape(data(), sizes...);
        }

        ///
        /// \brief iterators for Eigen matrices for STL compatibility.
        ///
        auto begin() { return data(); }
        auto begin() const { return data(); }

        auto end() { return data() + size(); }
        auto end() const { return data() + size(); }

        ///
        /// \brief pretty-print the tensor.
        ///
        std::ostream& print(std::ostream& stream,
            tensor_size_t prefix_space = 0, tensor_size_t prefix_delim = 0, tensor_size_t suffix = 0) const
        {
            [[maybe_unused]] const auto sprint = [&] (const char c, const tensor_size_t count) -> std::ostream&
            {
                for (tensor_size_t i = 0; i < count; ++ i)
                {
                    stream << c;
                }
                return stream;
            };

            if (prefix_space == 0 && prefix_delim == 0 && suffix == 0)
            {
                stream << "shape: " << dims() << "\n";
            }

            if constexpr (trank == 1)
            {
                sprint('[', prefix_delim + 1);
                if constexpr (std::is_same_v<tscalar, int8_t>)
                {
                    stream << vector().transpose().template cast<int16_t>();
                }
                else if constexpr (std::is_same_v<tscalar, uint8_t>)
                {
                    stream << vector().transpose().template cast<uint16_t>();
                }
                else
                {
                    stream << vector().transpose();
                }
                sprint(']', suffix + 1);
            }
            else if constexpr (trank == 2)
            {
                const auto matrix = this->matrix();
                for (tensor_size_t row = 0, rows = matrix.rows(); row < rows; ++ row)
                {
                    if (row == 0)
                    {
                        sprint(' ', prefix_space);
                        sprint('[', prefix_delim + 2);
                    }
                    else
                    {
                        sprint(' ', prefix_space + prefix_delim + 1);
                        sprint('[', 1);
                    }

                    if constexpr (std::is_same_v<tscalar, int8_t>)
                    {
                        stream << matrix.row(row).template cast<int16_t>();
                    }
                    else if constexpr (std::is_same_v<tscalar, uint8_t>)
                    {
                        stream << matrix.row(row).template cast<uint16_t>();
                    }
                    else
                    {
                        stream << matrix.row(row);
                    }

                    if (row + 1 < rows)
                    {
                        stream << "]\n";
                    }
                    else
                    {
                        sprint(']', suffix + 2);
                    }
                }
            }
            else
            {
                for (tensor_size_t row = 0, rows = this->template size<0>(); row < rows; ++ row)
                {
                    tensor(row).print(stream,
                        (row == 0) ? prefix_space : (prefix_space + prefix_delim + 1),
                        (row == 0) ? (prefix_delim + 1) : 0,
                        suffix);

                    if (row + 1 < rows)
                    {
                        stream << "\n";
                    }
                    else
                    {
                        stream << "]";
                    }
                }
            }
            return stream;
        }

    private:

        template <typename tarray>
        static void random(tarray&& array, tscalar min, tscalar max)
        {
            assert(min < max);
            urand(min, max, array.data(), array.data() + array.size(), make_rng());
        }

        template <typename tdata, typename... tindices>
        auto tvector(tdata ptr, tindices... indices) const
        {
            static_assert(sizeof...(indices) < trank, "invalid number of tensor dimensions");
            return map_vector(ptr + offset0(indices...), nano::size(nano::dims0(dims(), indices...)));
        }

        template <typename tdata, typename... tindices>
        auto tmatrix(tdata ptr, tindices... indices) const
        {
            static_assert(sizeof...(indices) == trank - 2, "invalid number of tensor dimensions");
            return map_matrix(ptr + offset0(indices...), rows(), cols());
        }

        template <typename tdata, typename... tindices>
        auto ttensor(tdata ptr, tindices... indices) const
        {
            static_assert(sizeof...(indices) < trank, "invalid number of tensor dimensions");
            return map_tensor(ptr + offset0(indices...), nano::dims0(dims(), indices...));
        }

        template <typename tdata, typename... tsizes>
        auto treshape(tdata ptr, tsizes... sizes) const
        {
            auto dims = nano::make_dims(sizes...);
            for (auto& dim : dims)
            {
                assert(dim == -1 || dim >= 0);
                if (dim == -1)
                {
                    dim = -size() / nano::size(dims);
                }
            }
            assert(nano::size(dims) == size());
            return map_tensor(ptr, dims);
        }

        template <typename tdata>
        auto tslice(tdata ptr, tensor_size_t begin, tensor_size_t end) const
        {
            assert(begin >= 0 && begin < end && end <= this->template size<0>());
            auto dims = this->dims();
            dims[0] = end - begin;
            return map_tensor(ptr + offset0(begin), dims);
        }
    };

    ///
    /// \brief iterators for STL compatibility.
    ///
    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    auto begin(tensor_t<tstorage, tscalar, trank>& m)
    {
        return m.data();
    }

    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    auto begin(const tensor_t<tstorage, tscalar, trank>& m)
    {
        return m.data();
    }

    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    auto end(tensor_t<tstorage, tscalar, trank>& m)
    {
        return m.data() + m.size();
    }

    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    auto end(const tensor_t<tstorage, tscalar, trank>& m)
    {
        return m.data() + m.size();
    }

    ///
    /// \brief construct consecutive tensor indices in the range [min, max).
    ///
    inline auto arange(tensor_size_t min, tensor_size_t max)
    {
        assert(min <= max);

        indices_t indices(max - min);
        indices.lin_spaced(min, max - 1);
        return indices;
    }

    ///
    /// \brief compare two tensors element-wise.
    ///
    template
    <
        template <typename, size_t> class tstorage1,
        template <typename, size_t> class tstorage2,
        typename tscalar, size_t trank
    >
    bool operator==(
        const tensor_t<tstorage1, tscalar, trank>& lhs,
        const tensor_t<tstorage2, tscalar, trank>& rhs)
    {
        return  lhs.dims() == rhs.dims() &&
                lhs.vector() == rhs.vector();
    }

    ///
    /// \brief compare two tensors element-wise.
    ///
    template
    <
        template <typename, size_t> class tstorage1,
        template <typename, size_t> class tstorage2,
        typename tscalar, size_t trank
    >
    bool operator!=(
        const tensor_t<tstorage1, tscalar, trank>& lhs,
        const tensor_t<tstorage2, tscalar, trank>& rhs)
    {
        return  lhs.dims() != rhs.dims() ||
                lhs.vector() != rhs.vector();
    }

    ///
    /// \brief pretty-print the given tensor.
    ///
    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    std::ostream& operator<<(std::ostream& stream, const tensor_t<tstorage, tscalar, trank>& tensor)
    {
        return tensor.print(stream);
    }

    ///
    /// \brief traits to check if a given type is a tensor.
    ///
    template <class T>
    struct is_tensor : std::false_type
    {
    };

    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    struct is_tensor<tensor_t<tstorage, tscalar, trank>> : std::true_type
    {
    };

    template <class T>
    inline constexpr bool is_tensor_v = is_tensor<T>::value;

    ///
    /// \brief create a tensor from an initializer list.
    ///
    template <typename tscalar, size_t trank, typename... tvalues>
    auto make_tensor(const tensor_dims_t<trank>& dims, tvalues... values)
    {
        const auto list = {static_cast<tscalar>(values)...};
        assert(::nano::size(dims) == static_cast<tensor_size_t>(list.size()));

        tensor_mem_t<tscalar, trank> tensor(dims);
        std::transform(list.begin(), list.end(), tensor.begin(), [] (auto value) { return static_cast<tscalar>(value); });
        return tensor;
    }

    ///
    /// \brief returns true if the two tensors are close, ignoring not-finite values if present.
    ///
    template
    <
        template <typename, size_t> class tstorage1,
        template <typename, size_t> class tstorage2,
        typename tscalar, size_t trank
    >
    bool close(
        const tensor_t<tstorage1, tscalar, trank>& lhs,
        const tensor_t<tstorage2, tscalar, trank>& rhs,
        double epsilon)
    {
        if (lhs.dims() != rhs.dims())
        {
            return false;
        }
        for (nano::tensor_size_t i = 0, size = lhs.size(); i < size; ++ i)
        {
            const auto lhs_finite = std::isfinite(lhs(i));
            const auto rhs_finite = std::isfinite(rhs(i));
            if (lhs_finite != rhs_finite)
            {
                return false;
            }
            else if (lhs_finite && !close(static_cast<double>(lhs(i)), static_cast<double>(rhs(i)), epsilon))
            {
                return false;
            }
        }
        return true;
    }
}
