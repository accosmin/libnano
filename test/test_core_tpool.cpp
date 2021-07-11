#include <numeric>
#include <algorithm>
#include <utest/utest.h>
#include <nano/core/tpool.h>
#include <nano/core/random.h>
#include <nano/core/numeric.h>

using namespace nano;

namespace
{
    // single-threaded
    template <typename toperator>
    auto test_single(const size_t size, const toperator op)
    {
        std::vector<double> results(size);
        for (size_t i = 0; i < results.size(); ++ i)
        {
            results[i] = op(i);
        }

        return std::accumulate(results.begin(), results.end(), 0.0);
    }

    // multi-threaded (by index)
    template <typename toperator>
    auto test_loopi(const size_t size, const toperator op)
    {
        std::vector<double> results(size, -1);
        nano::loopi(size, [&] (const size_t i, const size_t tnum)
        {
            UTEST_CHECK_LESS(i, size);
            UTEST_CHECK_LESS(tnum, tpool_t::size());

            results[i] = op(i);
        });

        return std::accumulate(results.begin(), results.end(), 0.0);
    }

    // multi-threaded (by range)
    template <typename toperator>
    auto test_loopr(const size_t size, const size_t chunk, const toperator op)
    {
        std::vector<double> results(size, -1);
        nano::loopr(size, chunk, [&] (const size_t begin, const size_t end, const size_t tnum)
        {
            UTEST_CHECK_LESS(begin, end);
            UTEST_CHECK_LESS_EQUAL(end, size);
            UTEST_CHECK_LESS(tnum, tpool_t::size());
            UTEST_CHECK_LESS_EQUAL(end - begin, chunk);

            for (auto i = begin; i < end; ++ i)
            {
                results[i] = op(i);
            }
        });

        return std::accumulate(results.begin(), results.end(), 0.0);
    }
}

UTEST_BEGIN_MODULE(test_core_tpool)

UTEST_CASE(empty)
{
    auto& pool = tpool_t::instance();

    UTEST_CHECK_EQUAL(pool.size(), std::thread::hardware_concurrency());
    UTEST_CHECK_EQUAL(pool.threads().size(), std::thread::hardware_concurrency());

    UTEST_CHECK_EQUAL(tpool_t::size(), std::thread::hardware_concurrency());
}

UTEST_CASE(future)
{
    std::packaged_task<int(int,int)> task([] (int a, int b) { return std::pow(a, b); });

    auto future = task.get_future();
    task(2, 9);

    UTEST_CHECK_EQUAL(future.get(), 512);
}

UTEST_CASE(future_join)
{
    std::packaged_task<int(int,int)> task([] (int a, int b) { return std::pow(a, b); });

    auto future = task.get_future();
    auto thread = std::thread{std::move(task), 2, 10};
    thread.join();

    UTEST_CHECK_EQUAL(future.get(), 1024);
}

UTEST_CASE(future_detach)
{
    std::packaged_task<int(int,int)> task([] (int a, int b) { return std::pow(a, b); });

    auto future = task.get_future();
    auto thread = std::thread{std::move(task), 2, 11};
    thread.detach();

    UTEST_CHECK_EQUAL(future.get(), 2048);
}

UTEST_CASE(enqueue)
{
    auto& pool = tpool_t::instance();

    const size_t max_tasks = 1024;
    const auto tasks = urand<size_t>(1U, max_tasks, make_rng());

    std::mutex mutex;
    std::vector<size_t> tasks_done;
    {
        tpool_section_t<future_t> futures;
        for (size_t j = 0; j < tasks; ++ j)
        {
            futures.push_back(pool.enqueue([=, &mutex, &tasks_done]()
            {
                const auto sleep1 = urand<size_t>(1, 5, make_rng());
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                {
                    const std::lock_guard<std::mutex> lock(mutex);

                    tasks_done.push_back(j + 1);
                }
            }));
        }
    }

    UTEST_CHECK_EQUAL(tasks_done.size(), tasks);
    for (size_t j = 0; j < tasks; ++ j)
    {
        UTEST_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
    }
}

UTEST_CASE(loopi)
{
    const auto op = [] (const size_t i) { return std::sin(i); };

    for (size_t size = 1; size <= size_t(123); size *= 3)
    {
        const auto eps = epsilon1<double>();
        const auto ref = test_single(size, op);

        UTEST_CHECK_CLOSE(ref, test_loopi(size, op), eps);
    }
}

UTEST_CASE(loopr)
{
    const auto op = [] (const size_t i) { return std::cos(i); };

    for (size_t size = 1; size <= size_t(128); size *= 2)
    {
        const auto eps = epsilon1<double>();
        const auto ref = test_single(size, op);

        UTEST_CHECK_CLOSE(ref, test_loopr(size, 1, op), eps);
        UTEST_CHECK_CLOSE(ref, test_loopr(size, 2, op), eps);
        UTEST_CHECK_CLOSE(ref, test_loopr(size, 3, op), eps);
        UTEST_CHECK_CLOSE(ref, test_loopr(size, 4, op), eps);
        UTEST_CHECK_CLOSE(ref, test_loopr(size, size, op), eps);
        UTEST_CHECK_CLOSE(ref, test_loopr(size, size + 1, op), eps);
    }
}

UTEST_END_MODULE()
