#pragma once

#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief target value of the positive class.
    ///
    inline scalar_t pos_target()
    {
        return +1;
    }

    ///
    /// \brief target value of the negative class.
    ///
    inline scalar_t neg_target()
    {
        return -1;
    }

    ///
    /// \brief check if a target value maps to a positive class.
    ///
    inline bool is_pos_target(const scalar_t target)
    {
        return target > 0;
    }

    ///
    /// \brief target tensor for single and multi-label classification problems with [n_labels] classes.
    ///
    namespace detail
    {
        inline void class_target(tensor3d_t&)
        {
        }

        template <typename... tindices>
        inline void class_target(tensor3d_t& target, const tensor_size_t index, const tindices... indices)
        {
            if (index >= 0 && index < target.size())
            {
                target(index) = pos_target();
            }
            class_target(target, indices...);
        }
    }

    template <typename... tindices>
    inline tensor3d_t class_target(const tensor_size_t n_labels, const tindices... indices)
    {
        tensor3d_t target(n_labels, 1, 1);
        target.full(neg_target());
        detail::class_target(target, indices...);
        return target;
    }

    ///
    /// \brief target vector for multi-label classification problems based on the sign of the predictions.
    ///
    inline tensor3d_t class_target(const tensor3d_t& outputs)
    {
        tensor3d_t target(outputs.dims());
        for (tensor_size_t i = 0; i < outputs.size(); ++ i)
        {
            target(i) = is_pos_target(outputs(i)) ? pos_target() : neg_target();
        }
        return target;
    }
}
