#pragma once

#include <Eigen/Core>
#include <type_traits>

namespace nano
{
    ///
    /// \brief matrix types.
    ///
    template
    <
        typename tscalar_,
        int trows = Eigen::Dynamic,
        int tcols = Eigen::Dynamic,
        typename tscalar = std::remove_const_t<tscalar_>
    >
    using tensor_matrix_t = Eigen::Matrix<tscalar, trows, tcols, Eigen::RowMajor>;

    ///
    /// \brief map non-constant data to matrices.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = std::remove_const_t<tscalar_>,
        typename tresult = Eigen::Map<tensor_matrix_t<tscalar>, alignment>
    >
    tresult map_matrix(tscalar_* data, tsize rows, tsize cols)
    {
        return tresult(data, rows, cols);
    }

    ///
    /// \brief map constant data to Eigen matrices.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = std::remove_const_t<tscalar_>,
        typename tresult = Eigen::Map<const tensor_matrix_t<tscalar>, alignment>
    >
    tresult map_matrix(const tscalar_* data, tsize rows, tsize cols)
    {
        return tresult(data, rows, cols);
    }
}

namespace Eigen
{
    ///
    /// \brief iterators for Eigen matrices for STL compatibility.
    ///
    template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
    auto begin(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m)
    {
        return m.data();
    }

    template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
    auto begin(const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m)
    {
        return m.data();
    }

    template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
    auto end(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m)
    {
        return m.data() + m.size();
    }

    template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
    auto end(const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m)
    {
        return m.data() + m.size();
    }
}
