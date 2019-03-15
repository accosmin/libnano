#include "lemarechal.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_lemarechal_t::config() const
{
    json_t json;
    json["ro"] = strcat(m_ro, "(1,inf)");
    return json;
}

void lsearch_lemarechal_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "ro", m_ro, 1 + eps, inf);
}

bool lsearch_lemarechal_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
    lsearch_step_t L = state0;
    lsearch_step_t R = L;

    bool R_updated = false;
    for (int i = 1; i < max_iterations() && t < stpmax(); ++ i)
    {
        if (!state.update(state0, t))
        {
            log(state);
            return false;
        }
        else if (state.has_armijo(state0, c1()))
        {
            log(state);
            if (state.has_wolfe(state0, c2()))
            {
                return true;
            }
            else
            {
                L = state;

                if (!R_updated)
                {
                    t *= m_ro;
                }
                else
                {
                    t = lsearch_step_t::cubic(L, R);
                }
            }
        }
        else
        {
            log(state);
            R = state;
            R_updated = true;
            t = lsearch_step_t::cubic(L, R);
        }
    }

    return false;
}
