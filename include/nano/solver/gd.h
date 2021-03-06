#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief gradient descent with line-search.
    ///
    class NANO_PUBLIC solver_gd_t final : public solver_t
    {
    public:

        using solver_t::minimize;

        ///
        /// \brief default constructor
        ///
        solver_gd_t();

        ///
        /// \brief @see lsearch_solver_t
        ///
        solver_state_t iterate(const solver_function_t&, const lsearch_t&, const vector_t& x0) const final;
    };
}
