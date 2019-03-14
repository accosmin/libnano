#pragma once

#include <nano/lsearch_algo.h>
#include <nano/lsearch_init.h>
#include <nano/solver_function.h>

namespace nano
{
    class solver_t;
    using solver_factory_t = factory_t<solver_t>;
    using rsolver_t = solver_factory_t::trobject;

    ///
    /// \brief returns the registered solvers.
    ///
    NANO_PUBLIC solver_factory_t& get_solvers();

    ///
    /// \brief generic (batch) optimization algorithm typically using an adaptive line-search method.
    ///
    class NANO_PUBLIC solver_t : public json_configurable_t
    {
    public:

        ///
        /// logging operator: op(state), returns false if the optimization should stop
        ///
        using logger_t = std::function<bool(const solver_state_t&)>;

        ///
        /// \brief constructor
        ///
        solver_t(const scalar_t c1 = 1e-1, const scalar_t c2 = 9e-1);

        ///
        /// \brief minimize the given function starting from the initial point x0 until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of iterations is reached or
        ///     - the user canceled the optimization (using the logging function) or
        ///     - the solver failed (e.g. line-search failed)
        ///
        solver_state_t minimize(const function_t& f, const vector_t& x0) const
        {
            assert(f.size() == x0.size());
            return minimize(solver_function_t(f), x0);
        }

        ///
        /// \brief
        ///
        void to_json(json_t&) const override;
        void from_json(const json_t&) override;

        ///
        /// \brief change parameters
        ///
        void lsearch(rlsearch_init_t&&);
        void lsearch(rlsearch_algo_t&&);
        void logger(const logger_t& logger) { m_logger = logger; }
        void epsilon(const scalar_t epsilon) { m_epsilon = epsilon; }
        void max_iterations(const int max_iterations) { m_max_iterations = max_iterations; }

        ///
        /// \brief access functions
        ///
        auto epsilon() const { return m_epsilon; }
        auto max_iterations() const { return m_max_iterations; }

    protected:

        ///
        /// \brief minimize the given function starting from the initial point x0
        ///
        virtual solver_state_t minimize(const solver_function_t&, const vector_t& x0) const = 0;

        ///
        /// \brief log the current optimization state (if the logger is provided)
        ///
        auto log(const solver_state_t& state) const
        {
            return !m_logger ? true : m_logger(state);
        }

        ///
        /// \brief update the current state using line-search
        ///
        bool lsearch(solver_state_t& state) const;

        ///
        /// \brief check if the optimization is done (convergence or error) after an iteration
        ///
        bool done(const solver_function_t& function, solver_state_t& state, const bool iter_ok) const;

    private:

        // attributes
        scalar_t            m_epsilon{1e-6};            ///< required precision (~magnitude of the gradient)
        int                 m_max_iterations{1000};     ///< maximum number of iterations
        logger_t            m_logger;                   ///<
        rlsearch_init_t     m_lsearch_init;             ///<
        rlsearch_algo_t     m_lsearch_algo;             ///<
    };
}
