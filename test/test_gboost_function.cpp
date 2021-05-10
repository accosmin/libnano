#include "fixture/utils.h"
#include <nano/core/numeric.h>
#include <nano/gboost/function.h>
#include <nano/dataset/memfixed.h>

using namespace nano;

class gboost_dataset_t final : public memfixed_dataset_t<scalar_t>
{
public:

    using memfixed_dataset_t::idims;
    using memfixed_dataset_t::tdims;
    using memfixed_dataset_t::target;
    using memfixed_dataset_t::samples;

    gboost_dataset_t() = default;

    void load() override
    {
        resize(cat_dims(m_samples, m_idims), cat_dims(m_samples, m_tdims));

        m_scale = vector_t::Random(m_groups);
        m_scale.array() += 1.1;

        m_outputs.resize(all_targets().dims());
        m_woutputs.resize(all_targets().dims());

        m_outputs.random();
        m_woutputs.random();

        for (tensor_size_t s = 0; s < m_samples; ++ s)
        {
            const auto group = s % m_groups;
            input(s).random();
            target(s).vector() = m_outputs.vector(s) + m_scale(group) * m_woutputs.vector(s);
        }
    }

    feature_t target() const override
    {
        return feature_t{"const"};
    }

    cluster_t cluster(const indices_t& samples) const
    {
        cluster_t cluster(this->samples(), m_groups);
        for (const auto sample : samples)
        {
            if (sample % 7 > 0)
            {
                cluster.assign(sample, sample % m_groups);
            }
        }
        return cluster;
    }

    void idims(const tensor3d_dims_t idims) { m_idims = idims; }
    void tdims(const tensor3d_dims_t tdims) { m_tdims = tdims; }
    void groups(const tensor_size_t groups) { m_groups = groups; }
    void samples(const tensor_size_t samples) { m_samples = samples; }

    vector_t bias(const indices_t& samples) const
    {
        const auto targets = this->targets(samples);
        const auto tmatrix = targets.reshape(targets.size<0>(), -1).matrix();
        return tmatrix.colwise().mean();
    }

    auto groups() const { return m_groups; }
    const auto& scale() const { return m_scale; }
    auto outputs(const indices_t& samples) const { return m_outputs.indexed<scalar_t>(samples); }
    auto woutputs(const indices_t& samples) const { return m_woutputs.indexed<scalar_t>(samples); }

private:

    // attributes
    vector_t            m_scale;            ///<
    tensor4d_t          m_outputs;          ///<
    tensor4d_t          m_woutputs;         ///<
    tensor_size_t       m_groups{1};        ///<
    tensor_size_t       m_samples{100};     ///< total number of samples to generate (train + validation + test)
    tensor3d_dims_t     m_idims{{10, 1, 1}};///< dimension of an input sample
    tensor3d_dims_t     m_tdims{{3, 1, 1}}; ///< dimension of a target/output sample
};

static auto make_samples()
{
    return ::nano::arange(0, 60);
}

static auto make_dataset(const tensor_size_t isize = 3, const tensor_size_t tsize = 2, const tensor_size_t groups = 3)
{
    auto dataset = gboost_dataset_t{};
    dataset.idims(make_dims(isize, 1, 1));
    dataset.tdims(make_dims(tsize, 1, 1));
    dataset.samples(100);
    dataset.groups(groups);
    UTEST_CHECK_NOTHROW(dataset.load());
    return dataset;
}

static void check_function(gboost_function_t& function, tensor_size_t expected_size)
{
    UTEST_CHECK_EQUAL(function.size(), expected_size);
    UTEST_CHECK_THROW(function.vAreg(-1e+0), std::runtime_error);
    UTEST_CHECK_THROW(function.vAreg(+1e+9), std::runtime_error);
    UTEST_CHECK_NOTHROW(function.vAreg(1e-1));

    UTEST_CHECK_NOTHROW(function.vAreg(0e-1));
    const auto f0 = function.vgrad(vector_t::Zero(function.size()));
    UTEST_CHECK_LESS(0, f0);
    UTEST_CHECK_NOTHROW(function.vAreg(1e-1));
    const auto fV = function.vgrad(vector_t::Zero(function.size()));
    UTEST_CHECK_LESS(f0, fV);
}

static void check_gradient(gboost_function_t& function, int trials)
{
    for (auto trial = 0; trial < trials; ++ trial)
    {
        const vector_t x = vector_t::Random(function.size());

        UTEST_CHECK_NOTHROW(function.vAreg(0e-1));
        UTEST_CHECK_LESS(function.grad_accuracy(x), 10 * epsilon2<scalar_t>());

        UTEST_CHECK_NOTHROW(function.vAreg(5e-1));
        UTEST_CHECK_LESS(function.grad_accuracy(x), 10 * epsilon2<scalar_t>());
    }
}

static void check_optimum(gboost_function_t& function, const vector_t& expected_optimum)
{
    UTEST_CHECK_NOTHROW(function.vAreg(1e-6));

    const auto solver = make_solver();
    const auto state = solver->minimize(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state);
    UTEST_CHECK(state.converged(solver->epsilon()));
    UTEST_CHECK_EIGEN_CLOSE(state.x, expected_optimum, 1e+2 * solver->epsilon());
}

template <typename ttmatrix, typename tomatrix>
static void check_value(gboost_function_t& function, const ttmatrix& tmatrix, const tomatrix& omatrix)
{
    const auto values1 = 0.5 * (tmatrix - omatrix).array().square().rowwise().sum();
    const auto values2 = values1.square();

    const auto f0 = values1.mean();
    const auto fV = f0 + 1e-1 * (values2.mean() - values1.mean() * values1.mean());

    UTEST_CHECK_NOTHROW(function.vAreg(0e-1));
    UTEST_CHECK_CLOSE(function.vgrad(vector_t::Zero(function.size())), f0, 1e-6);
    UTEST_CHECK_NOTHROW(function.vAreg(1e-1));
    UTEST_CHECK_CLOSE(function.vgrad(vector_t::Zero(function.size())), fV, 1e-6);
}

UTEST_BEGIN_MODULE(test_gboost_function)

UTEST_CASE(bias)
{
    const auto loss = make_loss();
    const auto dataset = make_dataset();
    const auto samples = make_samples();

    auto function = gboost_bias_function_t{*loss, dataset, samples};
    check_function(function, 2);
    check_gradient(function, 4);
    check_optimum(function, dataset.bias(samples));

    const auto targets = dataset.targets(samples);
    const auto tmatrix = targets.reshape(targets.size<0>(), -1).matrix();
    const auto omatrix = matrix_t::Zero(tmatrix.rows(), tmatrix.cols());
    check_value(function, tmatrix, omatrix);
}

UTEST_CASE(scale)
{
    const auto loss = make_loss();
    const auto solver = make_solver();
    const auto dataset = make_dataset();
    const auto samples = make_samples();
    const auto cluster = dataset.cluster(samples);
    const auto outputs = dataset.outputs(samples);
    const auto woutputs = dataset.woutputs(samples);

    auto function = gboost_scale_function_t{*loss, dataset, samples, cluster, outputs, woutputs};
    check_function(function, dataset.groups());
    check_gradient(function, 4);
    check_optimum(function, dataset.scale());

    const auto targets = dataset.targets(samples);
    const auto tmatrix = targets.reshape(targets.size<0>(), -1).matrix();
    const auto omatrix = outputs.reshape(tmatrix.rows(), tmatrix.cols()).matrix();
    check_value(function, tmatrix, omatrix);
}

UTEST_CASE(grads)
{
    const auto loss = make_loss();
    const auto solver = make_solver();
    const auto dataset = make_dataset();
    const auto samples = make_samples();
    const auto targets = dataset.targets(samples);

    auto function = gboost_grads_function_t{*loss, dataset, samples};
    check_function(function, samples.size() * 2);
    check_gradient(function, 4);
    check_optimum(function, targets.vector());

    const auto tmatrix = targets.reshape(targets.size<0>(), -1).matrix();
    const auto omatrix = matrix_t::Zero(tmatrix.rows(), tmatrix.cols());
    check_value(function, tmatrix, omatrix);
}

UTEST_END_MODULE()
