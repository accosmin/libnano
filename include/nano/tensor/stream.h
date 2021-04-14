#pragma once

#include <nano/stream.h>
#include <nano/tensor/hash.h>
#include <nano/tensor/tensor.h>

namespace nano
{
    ///
    /// \brief write unformatted the given tensor.
    ///
    template <template <typename tscalar, size_t> class tstorage, typename tscalar, size_t trank>
    std::ostream& write(std::ostream& stream, const tensor_t<tstorage, tscalar, trank>& tensor)
    {
        if (!::nano::write(stream, detail::tensor_version()) ||                     // version
            !::nano::write(stream, static_cast<uint32_t>(trank)) ||                 // rank
            !::nano::write_cast<int32_t>(stream, tensor.dims().data(), trank) ||    // dimensions
            !::nano::write(stream, static_cast<uint32_t>(sizeof(tscalar))) ||       // sizeof(scalar)
            !::nano::write(stream, detail::hash(tensor.data(), tensor.size())) ||   // hash(content)
            !::nano::write(stream, tensor.data(), tensor.size()))                   // content
        {
            stream.setstate(std::ios_base::failbit);
        }
        return stream;
    }

    ///
    /// \brief read unformatted the given tensor.
    ///
    template <template <typename tscalar, size_t> class tstorage, typename tscalar, size_t trank>
    std::istream& read(std::istream& stream, tensor_t<tstorage, tscalar, trank>& tensor)
    {
        typename tensor_t<tstorage, tscalar, trank>::tdims dims;
        uint32_t iversion = 0xFFFFFFFF, irank = 0, iscalar = 0;
        uint64_t ihash = 0;

        if (!::nano::read(stream, iversion) ||                                      // version
            !::nano::read(stream, irank) ||                                         // rank
            !::nano::read_cast<int32_t>(stream, dims.data(), trank) ||              // dimensions
            !::nano::read(stream, iscalar) ||                                       // sizeof(scalar)
            !::nano::read(stream, ihash) ||                                         // hash(content)
            iversion != detail::tensor_version() ||
            static_cast<size_t>(irank) != trank ||
            static_cast<size_t>(iscalar) != sizeof(tscalar))
        {
            stream.setstate(std::ios_base::failbit);
            return stream;
        }

        tensor.resize(dims);
        if (!::nano::read(stream, tensor.data(), tensor.size()) ||                  // content
            ihash != detail::hash(tensor.data(), tensor.size()))
        {
            stream.setstate(std::ios_base::failbit);
        }
        return stream;
    }
}
