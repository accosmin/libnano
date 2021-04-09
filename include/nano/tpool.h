#pragma once

#include <deque>
#include <mutex>
#include <future>
#include <thread>
#include <vector>
#include <cassert>
#include <condition_variable>

namespace nano
{
    using future_t = std::future<void>;
    using tpool_task_t = std::packaged_task<void()>;

    ///
    /// \brief enqueue tasks to be run in a thread pool.
    ///
    class tpool_queue_t
    {
    public:

        ///
        /// \brief constructor
        ///
        tpool_queue_t() = default;

        ///
        /// \brief enqueue a new task to execute
        ///
        template <typename tfunction>
        auto enqueue(tfunction&& f)
        {
            auto task = tpool_task_t(f);
            auto future = task.get_future();

            const std::lock_guard<std::mutex> lock(m_mutex);
            m_tasks.emplace_back(std::move(task));
            m_condition.notify_one();

            return future;
        }

        // attributes
        std::deque<tpool_task_t>        m_tasks;                ///< tasks to execute
        mutable std::mutex              m_mutex;                ///< synchronization
        mutable std::condition_variable m_condition;            ///< signaling
        bool                            m_stop{false};          ///< stop requested
    };

    ///
    /// \brief worker to process tasks enqueued in a thread pool.
    ///
    class tpool_worker_t
    {
    public:

        ///
        /// \brief constructor
        ///
        explicit tpool_worker_t(tpool_queue_t& queue) : m_queue(queue) {}

        ///
        /// \brief execute tasks when available
        ///
        void operator()() const
        {
            while (true)
            {
                tpool_task_t task;

                // wait for a new task to be available in the queue
                {
                    std::unique_lock<std::mutex> lock(m_queue.m_mutex);

                    m_queue.m_condition.wait(lock, [&]
                    {
                        return m_queue.m_stop || !m_queue.m_tasks.empty();
                    });

                    if (m_queue.m_stop)
                    {
                        m_queue.m_tasks.clear();
                        m_queue.m_condition.notify_all();
                        break;
                    }

                    task = std::move(m_queue.m_tasks.front());
                    m_queue.m_tasks.pop_front();
                }

                // execute the task
                task();
            }
        }

    private:

        // attributes
        tpool_queue_t&          m_queue;        ///< task queue to process
    };

    ///
    /// \brief RAII object to wait for a given set of futures (aka barrier).
    ///
    template <typename tfuture>
    class tpool_section_t : public std::vector<tfuture>
    {
    public:

        using std::vector<tfuture>::vector;

        ///
        /// \brief default constructor
        ///
        tpool_section_t() = default;

        ///
        /// \brief disable copying
        ///
        tpool_section_t(const tpool_section_t&) = delete;
        tpool_section_t& operator=(const tpool_section_t&) = delete;

        ///
        /// \brief enable moving
        ///
        tpool_section_t(tpool_section_t&&) noexcept = default;
        tpool_section_t& operator=(tpool_section_t&&) noexcept = default;

        ///
        /// \brief destructor
        ///
        ~tpool_section_t()
        {
            // block until all futures are done
            for (auto it = this->begin(); it != this->end(); ++ it)
            {
                it->wait();
            }
        }
    };

    ///
    /// \brief thread pool.
    /// NB: this is heavily copied/inspired by http://progsch.net/wordpress/?p=81
    ///
    class tpool_t
    {
    public:

        ///
        /// \brief single instance
        ///
        static tpool_t& instance()
        {
            static tpool_t the_pool;
            return the_pool;
        }

        ///
        /// \brief disable copying
        ///
        tpool_t(const tpool_t&) = delete;
        tpool_t& operator=(const tpool_t&) = delete;

        ///
        /// \brief disable moving
        ///
        tpool_t(tpool_t&&) noexcept = delete;
        tpool_t& operator=(tpool_t&&) noexcept = delete;

        ///
        /// \brief destructor
        ///
        ~tpool_t()
        {
            stop();
        }

        ///
        /// \brief enqueue a new task to execute
        ///
        template <typename tfunction>
        auto enqueue(tfunction f)
        {
            return m_queue.enqueue(std::move(f));
        }

        ///
        /// \brief returns the number of available worker threads.
        ///
        static size_t size()
        {
            return std::max(size_t(1), static_cast<size_t>(std::thread::hardware_concurrency()));
        }

        ///
        /// \brief returns the underlying threads.
        ///
        const auto& threads() const
        {
            return m_threads;
        }

    private:

        tpool_t()
        {
            const auto n_workers = size();

            m_workers.reserve(n_workers);
            for (size_t i = 0; i < n_workers; ++ i)
            {
                m_workers.emplace_back(m_queue);
            }

            for (size_t i = 0; i < n_workers; ++ i)
            {
                m_threads.emplace_back(std::ref(m_workers[i]));
            }
        }

        void stop()
        {
            // stop & join
            {
                const std::lock_guard<std::mutex> lock(m_queue.m_mutex);
                m_queue.m_stop = true;
                m_queue.m_condition.notify_all();
            }

            for (auto& thread : m_threads)
            {
                thread.join();
            }
        }

        // attributes
        std::vector<std::thread>        m_threads;      ///<
        std::vector<tpool_worker_t>     m_workers;      ///<
        tpool_queue_t                   m_queue;        ///< tasks to execute + synchronization
    };

    ///
    /// \brief split a loop computation of the given size in fixed-sized chunks using a thread pool.
    /// NB: the operator receives the range [begin, end) to process and the assigned thread index: op(begin, end, tnum)
    ///
    template <typename tsize, typename tchunk_, typename toperator>
    void loopr(const tsize size, const tchunk_ chunk_, const toperator& op)
    {
        const auto chunk = static_cast<tsize>(chunk_);

        assert(size >= tsize(0));
        assert(chunk >= tsize(1));

        auto& pool = tpool_t::instance();
        const auto workers = static_cast<tsize>(tpool_t::size());
        const auto tchunk = std::max((size + workers - 1) / workers, chunk);

        tpool_section_t<future_t> section;
        for (tsize tnum = 0, tbegin = 0; tnum < workers && tbegin < size; ++ tnum, tbegin += tchunk)
        {
            section.push_back(pool.enqueue([&op, size=size, chunk=chunk, tchunk=tchunk, tnum=tnum, tbegin=tbegin] ()
            {
                for (auto begin = tbegin, tend = std::min(tbegin + tchunk, size); begin < tend; begin += chunk)
                {
                    op(begin, std::min(begin + chunk, tend), static_cast<size_t>(tnum));
                }
            }));
        }

        // NB: the section is destroyed here waiting for all tasks to finish!
    }

    ///
    /// \brief split a loop computation of the given size using a thread pool.
    /// NB: the operator receives the index to process and the assigned thread index: op(index, tnum)
    ///
    template <typename tsize, typename toperator>
    void loopi(const tsize size, const toperator& op)
    {
        assert(size >= tsize(0));

        auto& pool = tpool_t::instance();
        const auto workers = static_cast<tsize>(tpool_t::size());
        const auto tchunk = (size + workers - 1) / workers;

        tpool_section_t<future_t> section;
        for (tsize tnum = 0, tbegin = 0; tnum < workers && tbegin < size; ++ tnum, tbegin += tchunk)
        {
            section.push_back(pool.enqueue([&op, size=size, tchunk=tchunk, tnum=tnum, tbegin=tbegin] ()
            {
                for (auto begin = tbegin, tend = std::min(tbegin + tchunk, size); begin < tend; ++ begin)
                {
                    op(begin, static_cast<size_t>(tnum));
                }
            }));
        }

        // NB: the section is destroyed here waiting for all tasks to finish!
    }
}
