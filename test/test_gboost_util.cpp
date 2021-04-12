#include <utest/utest.h>
#include <nano/gboost/util.h>

using namespace nano;

struct cache_t
{
    cache_t() = default;

    cache_t(scalar_t score, tensor_size_t index) :
        m_score(score),
        m_index(index)
    {
    }

    cache_t& operator+=(const cache_t& other)
    {
        m_score += other.m_score;
        return *this;
    }

    cache_t& operator/=(tensor_size_t samples)
    {
        m_score /= static_cast<scalar_t>(samples);
        return *this;
    }

    scalar_t        m_score{0};
    tensor_size_t   m_index{0};
};

UTEST_BEGIN_MODULE(test_gboost_util)

UTEST_CASE(reduce)
{
    std::vector<cache_t> caches;
    caches.emplace_back(1.0, tensor_size_t{0});
    caches.emplace_back(0.0, tensor_size_t{1});
    caches.emplace_back(2.0, tensor_size_t{2});
    caches.emplace_back(5.0, tensor_size_t{3});

    const auto& min = gboost::min_reduce(caches);
    UTEST_CHECK_EQUAL(min.m_index, 1);
    UTEST_CHECK_CLOSE(min.m_score, 0.0, 1e-12);

    const auto& sum = gboost::sum_reduce(caches, 10);
    UTEST_CHECK_EQUAL(sum.m_index, 0);
    UTEST_CHECK_CLOSE(sum.m_score, 0.8, 1e-12);
}

UTEST_CASE(accumulator)
{
    const auto tdims = make_dims(3, 1, 1);

    auto acc = gboost::accumulator_t(tdims);
    acc.clear(2);

    UTEST_CHECK_EQUAL(acc.fvalues(), 2);
    UTEST_CHECK_EQUAL(acc.tdims(), tdims);

    tensor4d_t vgrads(cat_dims(5, tdims));
    vgrads.tensor(0).full(+0.0);
    vgrads.tensor(1).full(+1.0);
    vgrads.tensor(2).full(+2.0);
    vgrads.tensor(3).full(+3.0);
    vgrads.tensor(4).full(+4.0);

    acc.update(-2.0, vgrads.array(0), 0);
    acc.update(+2.0, vgrads.array(0), 1);

    acc.update(-1.0, vgrads.array(1), 0);
    acc.update(+1.0, vgrads.array(1), 1);

    acc.update(-3.0, vgrads.array(2), 0);
    acc.update(+3.0, vgrads.array(2), 1);

    acc.update(-4.0, vgrads.array(3), 0);
    acc.update(+4.0, vgrads.array(3), 1);

    acc.update(-1.0, vgrads.array(4), 1);
    acc.update(+1.0, vgrads.array(4), 1);

    UTEST_CHECK_CLOSE(acc.x0(0), +4.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.x0(1), +6.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.x1(0), -10.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.x1(1), +10.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.x2(0), +30.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.x2(1), +32.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.r1(0).minCoeff(), -6.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.r1(0).maxCoeff(), -6.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.r1(1).minCoeff(), -14.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.r1(1).maxCoeff(), -14.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.rx(0).minCoeff(), +19.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rx(0).maxCoeff(), +19.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.rx(1).minCoeff(), -19.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rx(1).maxCoeff(), -19.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.r2(0).minCoeff(), +14.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.r2(0).maxCoeff(), +14.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.r2(1).minCoeff(), +46.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.r2(1).maxCoeff(), +46.0, 1e-12);
}

UTEST_END_MODULE()
