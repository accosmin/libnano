#pragma once

#include <nano/generator.h>

namespace nano
{
    using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;

    ///
    /// \brief
    ///
    NANO_PUBLIC feature_mapping_t select_scalar_components(
        const memory_dataset_t&, struct2scalar, const indices_t& feature_indices);
}
