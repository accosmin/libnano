#include <nano/core/numeric.h>
#include <nano/lsearchk/morethuente.h>

using namespace nano;

rlsearchk_t lsearchk_morethuente_t::clone() const
{
    return std::make_unique<lsearchk_morethuente_t>(*this);
}

void lsearchk_morethuente_t::dcstep(
    scalar_t& stx, scalar_t& fx, scalar_t& dx,
    scalar_t& sty, scalar_t& fy, scalar_t& dy,
    scalar_t& stp, const scalar_t& fp, const scalar_t& dp,
    bool& brackt,
    const scalar_t stpmin, const scalar_t stpmax) const
{
    scalar_t stpc = 0, stpq = 0, stpf = 0;

    const auto sgnd = dp * (dx / std::fabs(dx));

    if (fp > fx)
    {
        stpc = lsearch_step_t::cubic({stx, fx, dx}, {stp, fp, dp});
        stpq = lsearch_step_t::quadratic({stx, fx, dx}, {stp, fp, dp});

        if (std::fabs(stpc - stx) < std::fabs(stpq - stx))
        {
            stpf = stpc;
        }
        else
        {
            stpf = stpc + (stpq - stpc) / 2;
        }
        brackt = true;
    }

    else if (sgnd < 0)
    {
        stpc = lsearch_step_t::cubic({stx, fx, dx}, {stp, fp, dp});
        stpq = lsearch_step_t::secant({stx, fx, dx}, {stp, fp, dp});

        if (std::fabs(stpc - stp) > std::fabs(stpq - stp))
        {
            stpf = stpc;
        }
        else
        {
            stpf = stpq;
        }
        brackt = true;
    }

    else if (std::fabs(dp) < std::fabs(dx))
    {
        stpc = lsearch_step_t::cubic({stx, fx, dx}, {stp, fp, dp});
        stpq = lsearch_step_t::secant({stx, fx, dx}, {stp, fp, dp});

        if (std::isfinite(stpc) && (stp - stx) * (stpc - stp) > 0)
        {
        }
        else if (stp > stx)
        {
            stpc = stpmax;
        }
        else
        {
            stpc = stpmin;
        }

        if (brackt)
        {
            if (std::fabs(stpc - stp) < std::fabs(stpq - stp))
            {
                stpf = stpc;
            }
            else
            {
                stpf = stpq;
            }
            if (stp > stx)
            {
                stpf = std::min(stpf, stp + (sty - stp) * delta());
            }
            else
            {
                stpf = std::max(stpf, stp + (sty - stp) * delta());
            }
        }
        else
        {
            if (std::fabs(stpc - stp) > std::fabs(stpq - stp))
            {
                stpf = stpc;
            }
            else
            {
                stpf = stpq;
            }
            stpf = std::min(stpmax, stpf);
            stpf = std::max(stpmin, stpf);
        }
    }

    else
    {
        if (brackt)
        {
            stpc = lsearch_step_t::cubic({stp, fp, dp}, {sty, fy, dy});

            stpf = stpc;
        }
        else if (stp > stx)
        {
            stpf = stpmax;
        }
        else
        {
            stpf = stpmin;
        }
    }


    if (fp > fx)
    {
        sty = stp;
        fy = fp;
        dy = dp;
    }
    else
    {
        if (sgnd < 0.)
        {
            sty = stx;
            fy = fx;
            dy = dx;
        }
        stx = stp;
        fx = fp;
        dx = dp;
    }

    stp = stpf;
}

bool lsearchk_morethuente_t::get(const solver_state_t& state0, solver_state_t& state)
{
    const auto ftol = c1();
    const auto gtol = c2();
    const auto xtol = epsilon0<scalar_t>();

    int stage = 1;
    bool brackt = false;

    scalar_t stp = state.t, f = state.f, g = state.dg();
    scalar_t stmin = 0, stmax = stp + stp * 4;

    scalar_t width = stpmax() - stpmin();
    scalar_t width1 = 2 * width;

    scalar_t finit = state0.f, ginit = state0.dg(), gtest = ftol * ginit;
    scalar_t stx = 0, fx = finit, gx = ginit;
    scalar_t sty = 0, fy = finit, gy = ginit;

    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        const auto ftest = finit + stp * gtest;
        if (stage == 1 && f <= ftest && g >= scalar_t(0))
        {
            stage = 2;
        }

        // Check if further progress can be made
        if (brackt && (stp <= stmin || stp >= stmax))       { return true; }
        if (brackt && (stmax - stmin) <= xtol * stmax)      { return true; }
        if (stp >= stpmax() && f <= ftest && g <= gtest)    { return true; }
        if (stp <= stpmin() && (f > ftest || g >= gtest))   { return true; }

        // Check convergence
        if (f <= ftest && std::fabs(g) <= gtol * (-ginit))
        {
            return true;
        }

        // Interpolate the next point to evaluate
        if (stage == 1 && f <= fx && f > ftest)
        {
            auto fm = f - stp * gtest;
            auto fxm = fx - stx * gtest;
            auto fym = fy - sty * gtest;
            auto gm = g - gtest;
            auto gxm = gx - gtest;
            auto gym = gy - gtest;

            dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);

            fx = fxm + stx * gtest;
            fy = fym + sty * gtest;
            gx = gxm + gtest;
            gy = gym + gtest;
        }
        else
        {
            dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax);
        }

        // Decide if a bisection step is needed
        if (brackt)
        {
            if (std::fabs(sty - stx) >= width1 * scalar_t(.66))
            {
                stp = stx + (sty - stx) * scalar_t(0.5);
            }
            width1 = width;
            width = std::fabs(sty - stx);

            // Set the minimum and maximum steps allowed for stp
            stmin = std::min(stx, sty);
            stmax = std::max(stx, sty);
        }
        else
        {
            // Set the minimum and maximum steps allowed for stp
            stmin = stp + (stp - stx) * scalar_t(1.1);
            stmax = stp + (stp - stx) * scalar_t(4.0);
        }

        // Force the step to be within the bounds stpmax and stpmin
        stp = std::clamp(stp, stpmin(), stpmax());

        // If further progress is not possible, let stp be the best point obtained during the search
        if ((brackt && (stp <= stmin || stp >= stmax)) ||
            (brackt && stmax - stmin <= xtol * stmax))
        {
            stp = stx;
        }

        // Obtain another function and derivative
        state.update(state0, stp);
        log(state0, state);
        f = state.f;
        g = state.dg();
    }

    return false;
}
