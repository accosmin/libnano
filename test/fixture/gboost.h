#include "utils.h"
#include <nano/dataset/memfixed.h>
#include <nano/gboost/wlearner_dstep.h>
#include <nano/gboost/wlearner_dtree.h>
#include <nano/gboost/wlearner_hinge.h>
#include <nano/gboost/wlearner_table.h>
#include <nano/gboost/wlearner_stump.h>
#include <nano/gboost/wlearner_affine.h>

using namespace nano;

class fixture_dataset_t : public memfixed_dataset_t<scalar_t>
{
public:

    using memfixed_dataset_t::idims;
    using memfixed_dataset_t::tdims;
    using memfixed_dataset_t::target;
    using memfixed_dataset_t::samples;

    fixture_dataset_t() = default;

    virtual tensor_size_t groups() const = 0;

    virtual void make_target(tensor_size_t) = 0;

    template <typename toperator>
    scalar_t make_target(tensor_size_t sample, tensor_size_t feature, tensor_size_t modulo, const toperator& op)
    {
        auto input = this->input(sample);
        if (!feature_t::missing(input(feature)))
        {
            return op(input(feature) = static_cast<scalar_t>(sample % modulo));
        }
        else
        {
            return 0.0;
        }
    }

    scalar_t make_stump_target(
        tensor_size_t sample, tensor_size_t feature, tensor_size_t modulo,
        scalar_t threshold, scalar_t pred0, scalar_t pred1, tensor_size_t cluster = 0)
    {
        return make_target(sample, feature, modulo, [&] (const scalar_t x)
        {
            assign(sample, cluster + (x < threshold ? 0 : 1));
            return (x < threshold) ? pred0 : pred1;
        });
    }

    scalar_t make_hinge_target(
        tensor_size_t sample, tensor_size_t feature, tensor_size_t modulo,
        scalar_t threshold, scalar_t beta, ::nano::hinge type, tensor_size_t cluster = 0)
    {
        return make_target(sample, feature, modulo, [&] (const scalar_t x)
        {
            assign(sample, cluster);
            return (type == ::nano::hinge::left) ?
                ((x < threshold) ? (beta * (x - threshold)) : 0.0) :
                ((x < threshold) ? 0.0 : (beta * (x - threshold)));
        });
    }

    scalar_t make_table_target(
        tensor_size_t sample, tensor_size_t feature, tensor_size_t modulo,
        scalar_t scale, tensor_size_t cluster = 0)
    {
        return make_target(sample, feature, modulo, [&] (const scalar_t x)
        {
            assign(sample, cluster + (sample % modulo));
            return scale * (x - 1.0);
        });
    }

    scalar_t make_dstep_target(
        tensor_size_t sample, tensor_size_t feature, tensor_size_t modulo,
        scalar_t beta, tensor_size_t fvalue, tensor_size_t cluster = 0)
    {
        return make_target(sample, feature, modulo, [&] (const scalar_t x)
        {
            assign(sample, cluster);
            return (static_cast<tensor_size_t>(x) == fvalue) ? beta : 0.0;
        });
    }

    template <typename tfun1>
    scalar_t make_affine_target(
        tensor_size_t sample, tensor_size_t feature, tensor_size_t modulo,
        scalar_t weight, scalar_t bias, tensor_size_t cluster = 0)
    {
        return make_target(sample, feature, modulo, [&] (const scalar_t x)
        {
            assign(sample, cluster);
            return weight * tfun1::get(x) + bias;
        });
    }

    void load() override
    {
        resize(make_dims(m_samples, m_isize, 1, 1),
               make_dims(m_samples, m_tsize, 1, 1));

        auto rng = make_rng();
        auto udistd = make_udist<tensor_size_t>(0, 2);
        auto udistc = make_udist<scalar_t>(-1.0, +1.0);

        m_cluster = cluster_t{m_samples, this->groups()};

        for (tensor_size_t s = 0; s < m_samples; ++ s)
        {
            auto input = this->input(s);
            for (tensor_size_t f = 0; f < features(); ++ f)
            {
                if (is_discrete(f))
                {
                    input(f) = is_optional(s, f) ? feature_t::placeholder_value() : static_cast<scalar_t>(udistd(rng));
                }
                else
                {
                    input(f) = is_optional(s, f) ? feature_t::placeholder_value() : static_cast<scalar_t>(udistc(rng));
                }
            }

            auto target = this->target(s);
            target.random(-100.0, +100.0);

            make_target(s);
        }
    }

    feature_t target() const override
    {
        return feature_t{"wlearner"};
    }

    feature_t feature(const tensor_size_t index) const override
    {
        UTEST_REQUIRE_LESS_EQUAL(0, index);
        UTEST_REQUIRE_LESS(index, features());

        feature_t feature(scat("feature", index));
        if (is_discrete(index))
        {
            // discrete, optional
            feature.optional(true);
            feature.sclass(strings_t{"cat1", "cat2", "cat3"});
            UTEST_REQUIRE(feature.discrete());
            UTEST_REQUIRE(feature.optional());
        }
        else
        {
            // continuous, optional
            feature.optional(true);
            UTEST_REQUIRE(!feature.discrete());
            UTEST_REQUIRE(feature.optional());
        }

        return feature;
    }

    void isize(const tensor_size_t isize)
    {
        m_isize = isize;
    }

    void tsize(const tensor_size_t tsize)
    {
        m_tsize = tsize;
    }

    void samples(const tensor_size_t samples)
    {
        m_samples = samples;
    }

    void assign(tensor_size_t sample, tensor_size_t group)
    {
        UTEST_REQUIRE_LESS_EQUAL(0, sample);
        UTEST_REQUIRE_LESS(sample, m_samples);

        m_cluster.assign(sample, group);
    }

    virtual bool is_discrete(tensor_size_t feature) const
    {
        return (feature % 2) == 0;
    }

    virtual bool is_optional(tensor_size_t sample, tensor_size_t feature) const
    {
        return (sample + feature) % 23 == 0;
    }

    tensor_size_t get_feature(bool discrete) const
    {
        return get_feature(isize(), discrete);
    }

    tensor_size_t get_feature(tensor_size_t feature, bool discrete) const
    {
        -- feature;
        for (; feature >= 0; -- feature)
        {
            if (is_discrete(feature) == discrete)
            {
                return feature;
            }
        }
        assert(isize() > 0);
        return 0;
    }

    tensor_size_t isize() const { return m_isize; }
    tensor_size_t tsize() const { return m_tsize; }
    const auto& cluster() const { return m_cluster; }

private:

    // attributes
    tensor_size_t       m_isize{10};        ///<
    tensor_size_t       m_tsize{1};         ///<
    tensor_size_t       m_samples{100};     ///< total number of samples to generate
    cluster_t           m_cluster;          ///< split of the training samples using the ground truth feature
};

template <typename tdataset>
class no_discrete_features_dataset_t final : public tdataset
{
public:

    no_discrete_features_dataset_t() = default;

    bool is_discrete(const tensor_size_t) const override
    {
        return false;
    }
};

template <typename tdataset>
class no_continuous_features_dataset_t final : public tdataset
{
public:

    no_continuous_features_dataset_t() = default;

    bool is_discrete(const tensor_size_t) const override
    {
        return true;
    }
};

template <typename tdataset>
class different_discrete_feature_dataset_t final : public tdataset
{
public:

    different_discrete_feature_dataset_t() = default;

    feature_t feature(const tensor_size_t index) const override
    {
        auto feature = tdataset::feature(index);
        if (index == tdataset::the_discrete_feature())
        {
            feature.sclass(strings_t{"cat1", "more", "more", "too many"});
        }
        return feature;
    }
};

template <typename tdataset>
auto make_dataset(tensor_size_t isize = 10, tensor_size_t tsize = 1, tensor_size_t samples = 100)
{
    auto dataset = tdataset{};
    dataset.isize(isize);
    dataset.tsize(tsize);
    dataset.samples(samples);
    UTEST_CHECK_NOTHROW(dataset.load());
    return dataset;
}

template <typename twlearner>
auto make_wlearner(int batch = 32)
{
    auto wlearner = twlearner{};
    wlearner.batch(batch);
    return wlearner;
}

inline auto make_samples(const dataset_t& dataset)
{
    return arange(1 * dataset.samples() / 10, 9 * dataset.samples() / 10);
}

inline auto make_all_samples(const dataset_t& dataset)
{
    return arange(0, dataset.samples());
}

inline auto make_invalid_samples(const dataset_t& dataset)
{
    auto indices = make_samples(dataset);
    UTEST_REQUIRE_GREATER(indices.size(), 1);
    indices(0) = indices(1) + 1; // NB: valid indices should be sorted!
    return indices;
}

inline auto make_residuals(const dataset_t& dataset, const loss_t& loss)
{
    tensor4d_t outputs(cat_dims(dataset.samples(), dataset.tdims()));
    outputs.full(+0.0);

    tensor4d_t residuals;
    loss.vgrad(dataset.targets(arange(0, dataset.samples())), outputs, residuals);
    return residuals;
}

inline auto check_fit(wlearner_t& wlearner, const fixture_dataset_t& dataset)
{
    const auto loss = make_loss();
    const auto samples = make_samples(dataset);
    const auto residuals = make_residuals(dataset, *loss);

    auto fit_score = feature_t::placeholder_value();
    UTEST_REQUIRE(!std::isfinite(fit_score));
    UTEST_REQUIRE_NOTHROW(fit_score = wlearner.fit(dataset, samples, residuals));
    UTEST_REQUIRE(std::isfinite(fit_score));
    return fit_score;
}

inline void check_no_fit(wlearner_t& wlearner, const fixture_dataset_t& dataset)
{
    const auto loss = make_loss();
    const auto samples = make_samples(dataset);
    const auto residuals = make_residuals(dataset, *loss);

    auto fit_score = feature_t::placeholder_value();
    UTEST_CHECK_EQUAL(std::isfinite(fit_score), false);
    UTEST_CHECK_NOTHROW(fit_score = wlearner.fit(dataset, samples, residuals));
    UTEST_CHECK_EQUAL(std::isfinite(fit_score), true);
    UTEST_CHECK_EQUAL(fit_score, wlearner_t::no_fit_score());
}

inline void check_split(const wlearner_t& wlearner, const fixture_dataset_t& dataset)
{
    const auto samples = make_all_samples(dataset);
    const auto& gcluster = dataset.cluster();

    cluster_t wcluster;
    UTEST_CHECK_NOTHROW(wcluster = wlearner.split(dataset, samples));

    UTEST_REQUIRE_EQUAL(wcluster.samples(), dataset.samples());
    UTEST_REQUIRE_EQUAL(wcluster.samples(), gcluster.samples());

    UTEST_REQUIRE_EQUAL(wcluster.groups(), gcluster.groups());
    for (tensor_size_t g = 0; g < gcluster.groups(); ++ g)
    {
        UTEST_REQUIRE_EQUAL(wcluster.count(g), gcluster.count(g));
        UTEST_CHECK_EQUAL(wcluster.indices(g), gcluster.indices(g));
    }
}

inline void check_split_throws(const wlearner_t& wlearner, const indices_t& samples, const dataset_t& dataset)
{
    cluster_t wcluster;
    UTEST_CHECK_THROW(wcluster = wlearner.split(dataset, samples), std::runtime_error);
}

template <typename... tdatasets>
inline void check_split_throws(const wlearner_t& wlearner, const indices_t& samples, const dataset_t& dataset, const tdatasets&... datasets)
{
    check_split_throws(wlearner, samples, dataset);
    check_split_throws(wlearner, samples, datasets...);
}

inline void check_predict(const wlearner_t& wlearner, const fixture_dataset_t& dataset)
{
    const auto samples = make_samples(dataset);
    const auto inputs = dataset.inputs(samples);
    const auto targets = dataset.targets(samples);
    const auto imatrix = inputs.reshape(samples.size(), -1);

    const auto& cluster = dataset.cluster();
    const auto tsize = ::nano::size(dataset.tdims());

    tensor4d_t outputs;
    UTEST_REQUIRE_NOTHROW(outputs = wlearner.predict(dataset, samples));

    for (tensor_size_t s = 0; s < imatrix.rows(); ++ s)
    {
        if (cluster.group(samples(s)) < 0)
        {
            UTEST_CHECK_EIGEN_CLOSE(outputs.vector(s), vector_t::Zero(tsize), 1e-8);
        }
        else
        {
            UTEST_CHECK_EIGEN_CLOSE(outputs.array(s), targets.array(s), 1e-8);
        }
    }
}

inline void check_predict_throws(const wlearner_t& wlearner, const dataset_t& dataset)
{
    const auto samples = make_samples(dataset);
    UTEST_CHECK_THROW(wlearner.predict(dataset, samples), std::runtime_error);
}

template <typename... tdatasets>
inline void check_predict_throws(const wlearner_t& wlearner, const dataset_t& dataset, const tdatasets&... datasets)
{
    check_predict_throws(wlearner, dataset);
    check_predict_throws(wlearner, datasets...);
}

inline void check_scale(wlearner_t& wlearner, const fixture_dataset_t& dataset)
{
    const auto samples = make_samples(dataset);
    tensor4d_t outputs, outputs_scaled;
    UTEST_CHECK_NOTHROW(outputs = wlearner.predict(dataset, samples));

    const auto& cluster = dataset.cluster();
    {
        vector_t scale = vector_t::Constant(1, 2.0);

        UTEST_CHECK_NOTHROW(wlearner.scale(scale));
        UTEST_CHECK_NOTHROW(outputs_scaled = wlearner.predict(dataset, samples));
        UTEST_CHECK_EIGEN_CLOSE(outputs.array() * scale(0), outputs_scaled.array(), 1e-8);

        scale = vector_t::Constant(1, 0.5);
        UTEST_CHECK_NOTHROW(wlearner.scale(scale));
    }
    if (cluster.groups() != 1)
    {
        vector_t scale = vector_t::Random(cluster.groups());
        scale.array() += 2.0;

        UTEST_CHECK_NOTHROW(wlearner.scale(scale));
        UTEST_CHECK_NOTHROW(outputs_scaled = wlearner.predict(dataset, samples));
        for (tensor_size_t s = 0; s < samples.size(); ++ s)
        {
            const auto group = cluster.group(samples(s));
            const auto factor = (group < 0) ? 1.0 : scale(group);
            UTEST_CHECK_EIGEN_CLOSE(outputs.array(s) * factor, outputs_scaled.array(s), 1e-8);
        }
    }
    {
        const vector_t scale = vector_t::Constant(cluster.groups(), -1.0);
        UTEST_CHECK_THROW(wlearner.scale(scale), std::runtime_error);
    }
    {
        vector_t scale = vector_t::Constant(cluster.groups() + 10, +1.0);
        UTEST_CHECK_THROW(wlearner.scale(scale), std::runtime_error);
    }
}

template <typename twlearner>
auto stream_wlearner(const twlearner& wlearner)
{
    string_t blob;
    {
        std::ostringstream ostream;
        UTEST_REQUIRE_NOTHROW(wlearner.write(ostream));
        blob = ostream.str();
    }
    {
        twlearner default_wlearner;
        std::ostringstream ostream;
        UTEST_REQUIRE_NOTHROW(default_wlearner.write(ostream));
    }
    {
        std::ostringstream ostream;
        UTEST_REQUIRE_NOTHROW(wlearner.clone()->write(ostream));
        UTEST_CHECK_EQUAL(ostream.str(), blob);
    }
    {
        std::istringstream istream(blob);
        auto iwlearner = twlearner{};
        UTEST_REQUIRE_NOTHROW(iwlearner.read(istream));
        UTEST_CHECK_EQUAL(iwlearner.batch(), wlearner.batch());
        return iwlearner;
    }
}

template <typename twlearner, typename tdataset, typename... tinvalid_datasets>
void check_wlearner(twlearner& wlearner, const tdataset& dataset, const tinvalid_datasets&... idatasets)
{
    // the weak learner should not be usable before fitting
    check_predict_throws(wlearner, dataset);
    check_predict_throws(wlearner, idatasets...);

    check_split_throws(wlearner, make_samples(dataset), dataset);
    check_split_throws(wlearner, make_samples(dataset), idatasets...);

    // check fitting
    const auto score = check_fit(wlearner, dataset);
    UTEST_CHECK_CLOSE(score, 0.0, 1e-8);
    dataset.check_wlearner(wlearner);

    // check prediction
    check_predict(wlearner, dataset);
    check_predict_throws(wlearner, idatasets...);

    // check splitting
    check_split(wlearner, dataset);
    check_split_throws(wlearner, make_samples(dataset), idatasets...);
    check_split_throws(wlearner, make_invalid_samples(dataset), dataset);
    check_split_throws(wlearner, make_invalid_samples(dataset), idatasets...);

    // check model loading and saving from and to binary streams
    const auto iwlearner = stream_wlearner(wlearner);
    dataset.check_wlearner(iwlearner);

    // check scaling
    check_scale(wlearner, dataset);
}
