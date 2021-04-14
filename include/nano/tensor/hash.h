#pragma once

#include <functional>
#include <nano/tensor/index.h>

namespace nano
{
    namespace detail
    {
        inline constexpr uint32_t tensor_version()
        {
            return 0;
        }

        inline uint64_t hash_combine(const uint64_t h1, const uint64_t h2)
        {
            return h1 ^ (h2 << 1U);
        }

        template <typename tscalar>
        uint64_t hash(const tscalar* data, const tensor_size_t size)
        {
            const auto hasher = std::hash<tscalar>{};

            uint64_t hash = 0;
            for (tensor_size_t i = 0 ; i < size; ++ i)
            {
                hash = hash_combine(hash, hasher(data[i]));
            }
            return hash;
        }
    }
}
