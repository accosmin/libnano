#pragma once

#include <nano/lsearch/strategy.h>

namespace nano
{
    ///
    /// \brief backtracking line-search that stops when the Armijo condition is satisfied,
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
    ///
    class lsearch_backtrack_t final : public lsearch_strategy_t
    {
    public:

        lsearch_backtrack_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        bool get(const solver_state_t& state0, solver_state_t& state) final;

    private:

        // attributes
        scalar_t    m_ro{static_cast<scalar_t>(0.5)};   ///<
    };
}
