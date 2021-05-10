#include <nano/core/numeric.h>
#include <nano/lsearchk/lemarechal.h>

using namespace nano;

rlsearchk_t lsearchk_lemarechal_t::clone() const
{
    return std::make_unique<lsearchk_lemarechal_t>(*this);
}

bool lsearchk_lemarechal_t::get(const solver_state_t& state0, solver_state_t& state)
{
    lsearch_step_t L = state0;
    lsearch_step_t R = L;

    bool R_updated = false;
    for (int64_t i = 1; i < max_iterations(); ++ i)
    {
        scalar_t tmin = 0, tmax = 0;
        if (state.has_armijo(state0, c1()))
        {
            if (state.has_wolfe(state0, c2()))
            {
                return true;
            }
            else
            {
                L = state;
                if (!R_updated)
                {
                    tmin = std::max(L.t, R.t) + 2 * std::fabs(L.t - R.t);
                    tmax = std::max(L.t, R.t) + tau1() * std::fabs(L.t - R.t);
                }
                else
                {
                    tmin = std::min(L.t, R.t);
                    tmax = std::max(L.t, R.t);
                }
            }
        }
        else
        {
            R = state;
            R_updated = true;
            tmin = std::min(L.t, R.t);
            tmax = std::max(L.t, R.t);
        }

        // next trial
        const auto next = lsearch_step_t::interpolate(L, R, m_interpolation);
        const auto ok = state.update(state0, std::clamp(next, tmin, tmax));
        log(state0, state);

        if (!ok)
        {
            return false;
        }
    }

    return false;
}
