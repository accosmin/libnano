#pragma once

#include <Eigen/Core>
#include <type_traits>

namespace nano
{
    ///
    /// \brief vector types.
    ///
    template
    <
        typename tscalar_,
        int trows = Eigen::Dynamic,
        typename tscalar = std::remove_const_t<tscalar_>
    >
    using tensor_vector_t = Eigen::Matrix<tscalar, trows, 1, Eigen::ColMajor>;

    ///
    /// \brief map non-constant arrays to vectors.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = std::remove_const_t<tscalar_>,
        typename tresult = Eigen::Map<tensor_vector_t<tscalar>, alignment>
    >
    tresult map_vector(tscalar_* data, tsize rows)
    {
        return tresult(data, rows);
    }

    ///
    /// \brief map constant arrays to vectors.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = std::remove_const_t<tscalar_>,
        typename tresult = Eigen::Map<const tensor_vector_t<tscalar>, alignment>
    >
    tresult map_vector(const tscalar_* data, tsize rows)
    {
        return tresult(data, rows);
    }
}
