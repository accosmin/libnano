#pragma once

#include <nano/tensor/index.h>

namespace nano
{
    ///
    /// \brief base tensor class.
    ///     - stores dimensions
    ///     - handles the indexing
    ///
    template
    <
        typename tscalar, size_t trank,
        std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true
    >
    class tensor_base_t
    {
    public:

        static_assert(
            trank >= 1,
            "cannot create tensors with fewer than one dimension");

        using tscalar_remove_cvref = std::remove_cv_t<std::remove_reference_t<tscalar>>;
        static_assert(
            std::is_same_v<tscalar, tscalar_remove_cvref>,
            "cannot create tensors with cvref scalars");

        using tdims = tensor_dims_t<trank>;

        ///
        /// \brief default constructor
        ///
        tensor_base_t()
        {
            m_dims.fill(0);
        }

        ///
        /// \brief constructor.
        ///
        explicit tensor_base_t(tdims dims) :
            m_dims(std::move(dims))
        {
        }

        ///
        /// \brief enable copying.
        ///
        tensor_base_t(const tensor_base_t&) = default;
        tensor_base_t& operator=(const tensor_base_t&) = default;

        ///
        /// \brief enable moving.
        ///
        tensor_base_t(tensor_base_t&&) noexcept = default;
        tensor_base_t& operator=(tensor_base_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        ~tensor_base_t() = default;

        ///
        /// \brief number of dimensions (aka the rank of the tensor).
        ///
        static constexpr auto rank()
        {
            return trank;
        }

        ///
        /// \brief list of dimensions.
        ///
        const auto& dims() const
        {
            return m_dims;
        }

        ///
        /// \brief gather the missing dimensions in a multi-dimensional tensor
        ///     (assuming the last dimensions that are ignored are zero).
        ///
        template <typename... tindices>
        auto dims0(const tindices... indices) const
        {
            static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
            return nano::dims0(dims(), indices...);
        }

        ///
        /// \brief total number of elements.
        ///
        auto size() const
        {
            return nano::size(m_dims);
        }

        ///
        /// \brief number of elements for the given dimension.
        ///
        template <int idim>
        auto size() const
        {
            return std::get<idim>(m_dims);
        }

        ///
        /// \brief interpret the last two dimensions as rows/columns.
        /// NB: e.g. images represented as 3D tensors (color plane, rows, columns)
        /// NB: e.g. ML minibatches represented as 4D tensors (sample, feature plane, rows, columns)
        ///
        auto rows() const
        {
            static_assert(trank >= 2);
            return size<trank - 2>();
        }
        auto cols() const
        {
            static_assert(trank >= 2);
            return size<trank - 1>();
        }

        ///
        /// \brief compute the linearized index from the list of offsets.
        ///
        template <typename... tindices>
        auto offset(const tindices... indices) const
        {
            static_assert(sizeof...(indices) == trank, "invalid number of tensor dimensions");
            return nano::index(dims(), indices...);
        }

        ///
        /// \brief compute the linearized index from the list of offsets
        ///     (assuming the last dimensions that are ignored are zero).
        ///
        template <typename... tindices>
        auto offset0(const tindices... indices) const
        {
            static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
            return nano::index0(dims(), indices...);
        }

        ///
        /// \brief change size.
        ///
        void resize(const tdims& dims)
        {
            m_dims = dims;
        }

    private:

        // attributes
        tensor_dims_t<trank>    m_dims{};     ///<
    };
}
