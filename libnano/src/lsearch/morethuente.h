#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief More & Thunte line-search.
    ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease",
    ///     by Jorge J. More and David J. Thuente
    ///
    /// NB: this implementation ports the 'dcsrch' and the 'dcstep' Fortran routines from MINPACK-2.
    ///     see http://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/
    ///
    class lsearch_morethuente_t final : public lsearch_strategy_t
    {
    public:

        lsearch_morethuente_t() = default;
        void to_json(json_t&) const final;
        void from_json(const json_t&) final;
        bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;
    };
}