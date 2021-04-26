#pragma once

#include <nano/generator.h>

namespace nano
{
    ///
    /// \brief
    ///
    NANO_PUBLIC std::vector<tensor_size_t> select_scalar_components(
        const memory_dataset_t&, struct2scalar, const indices_t& feature_indices);
}
