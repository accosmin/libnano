#include <nano/core/numeric.h>
#include <nano/lsearchk/cgdescent.h>

using namespace nano;

rlsearchk_t lsearchk_cgdescent_t::clone() const
{
    auto lsearchk = std::make_unique<lsearchk_cgdescent_t>(*this);
    lsearchk->m_sumQ = 0;
    lsearchk->m_sumC = 0;
    lsearchk->m_approx = false;
    return rlsearchk_t{lsearchk.release()};
}

lsearchk_cgdescent_t::status lsearchk_cgdescent_t::updateU(const solver_state_t& state0,
    lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    for (int64_t i = 0; i < max_iterations() && (b.t - a.t) > stpmin(); ++ i)
    {
        if (evaluate(state0, (1 - theta()) * a.t + theta() * b.t, c))
        {
            return status::exit;
        }
        else if (!c.has_descent())
        {
            b = c;
            return status::done;
        }
        else if (c.has_approx_armijo(state0, epsilon() * m_sumC))
        {
            a = c;
        }
        else
        {
            b = c;
        }
    }

    return status::fail;
}

lsearchk_cgdescent_t::status lsearchk_cgdescent_t::update(const solver_state_t& state0,
    lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    if (!c || c.t <= a.t || c.t >= b.t)
    {
        return status::done;
    }
    else if (!c.has_descent())
    {
        b = c;
        return status::done;
    }
    else if (c.has_approx_armijo(state0, epsilon() * m_sumC))
    {
        a = c;
        return status::done;
    }
    else
    {
        b = c;
        return updateU(state0, a, b, c);
    }
}

lsearchk_cgdescent_t::status lsearchk_cgdescent_t::secant2(const solver_state_t& state0,
    lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    const auto a0 = a, b0 = b;
    const auto tc = lsearch_step_t::secant(a0, b0);

    if (evaluate(state0, tc, c))
    {
        return status::exit;
    }

    const auto update_status = update(state0, a, b, c);
    if (update_status != status::done)
    {
        return update_status;
    }
    else if (std::fabs(tc - a.t) < epsilon0<scalar_t>())
    {
        return  evaluate(state0, lsearch_step_t::secant(a0, a), c) ?
                status::exit : update(state0, a, b, c);
    }
    else if (std::fabs(tc - b.t) < epsilon0<scalar_t>())
    {
        return  evaluate(state0, lsearch_step_t::secant(b0, b), c) ?
                status::exit : update(state0, a, b, c);
    }
    else
    {
        return status::done;
    }
}

lsearchk_cgdescent_t::status lsearchk_cgdescent_t::bracket(const solver_state_t& state0,
    lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    auto last_a = a;
    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        if (!c.has_descent())
        {
            a = last_a;
            b = c;
            return status::done;
        }
        else if (!c.has_approx_armijo(state0, epsilon() * m_sumC))
        {
            a = state0;
            b = c;
            return updateU(state0, a, b, c);
        }
        else
        {
            last_a = c;
            if (evaluate(state0, ro() * c.t, c))
            {
                return status::exit;
            }
        }
    }

    return status::fail;
}

bool lsearchk_cgdescent_t::evaluate(const solver_state_t& state0, const scalar_t t, solver_state_t& c)
{
    const bool ok = c.update(state0, t);
    log(state0, c);

    return ok && evaluate(state0, c);
}

bool lsearchk_cgdescent_t::evaluate(const solver_state_t& state0, const solver_state_t& state)
{
    switch (m_criterion)
    {
    case criterion::wolfe:
        return  state.has_armijo(state0, c1()) &&
                state.has_wolfe(state0, c2());

    case criterion::approx_wolfe:
        return  state.has_approx_armijo(state0, epsilon() * m_sumC) &&
                state.has_approx_wolfe(state0, c1(), c2());

    case criterion::wolfe_approx_wolfe:
    default:
        if (!m_approx)
        {
            if (state.has_armijo(state0, c1()) &&
                state.has_wolfe(state0, c2()))
            {
                // decide if to switch permanently to the approximate Wolfe conditions
                m_approx = std::fabs(state.f - state0.f) <= omega() * m_sumC;
                return true;
            }
            return false;
        }
        else
        {
            return  state.has_approx_armijo(state0, epsilon() * m_sumC) &&
                    state.has_approx_wolfe(state0, c1(), c2());
        }
    }
}

bool lsearchk_cgdescent_t::get(const solver_state_t& state0, solver_state_t& state)
{
    // estimate an upper bound of the function value
    // (to be used for the approximate Wolfe condition)
    m_sumQ = 1 + m_sumQ * delta();
    m_sumC = m_sumC + (std::fabs(state0.f) - m_sumC) / m_sumQ;

    // evaluate the initial step length
    auto& c = state;
    if (evaluate(state0, state))
    {
        return true;
    }

    // bracket the initial step size
    lsearch_step_t a = state0, b = c;
    switch (bracket(state0, a, b, c))
    {
    case status::exit:  return true;
    case status::fail:  return false;
    default:            break;
    }

    // iteratively update the search interval [a, b]
    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        // secant interpolation
        const auto prev_width = b.t - a.t;
        switch (secant2(state0, a, b, c))
        {
        case status::exit:  return true;
        case status::fail:  return false;
        default:            break;
        }

        // update search interval
        if (b.t - a.t > gamma() * prev_width)
        {
            if (evaluate(state0, (a.t + b.t) / 2, c))
            {
                return true;
            }

            switch (update(state0, a, b, c))
            {
            case status::exit:  return true;
            case status::fail:  return false;
            default:            break;
            }
        }
    }

    return false;
}
