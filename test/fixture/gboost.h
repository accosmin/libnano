#include <nano/loss.h>
#include <utest/utest.h>
#include <nano/dataset/memfixed.h>
#include <nano/gboost/wlearner_dtree.h>
#include <nano/gboost/wlearner_table.h>
#include <nano/gboost/wlearner_stump.h>
#include <nano/gboost/wlearner_affine.h>

using namespace nano;

class fixture_dataset_t : public memfixed_dataset_t<scalar_t>
{
public:

    using memfixed_dataset_t::idim;
    using memfixed_dataset_t::tdim;
    using memfixed_dataset_t::samples;

    fixture_dataset_t() = default;

    [[nodiscard]] virtual tensor_size_t groups() const = 0;

    virtual void make_scale(tensor_size_t) {};
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
        scalar_t threshold, scalar_t pred0, scalar_t pred1, tensor_size_t cluster)
    {
        return make_target(sample, feature, modulo, [&] (const scalar_t x)
        {
            assign(sample, cluster + (x < threshold ? 0 : 1));
            return (x < threshold) ? pred0 : pred1;
        });
    }

    scalar_t make_table_target(
        tensor_size_t sample, tensor_size_t feature, tensor_size_t modulo,
        scalar_t scale, tensor_size_t cluster)
    {
        return make_target(sample, feature, modulo, [&] (const scalar_t x)
        {
            assign(sample, cluster + (sample % modulo));
            return scale * (x - 1.0);
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

    bool load() override
    {
        resize(make_dims(m_samples, m_isize, 1, 1),
               make_dims(m_samples, m_tsize, 1, 1));

        m_scales.resize(make_dims(m_samples, m_tsize, 1, 1));
        m_scales.constant(1.0);

        const auto tr_samples = m_samples * train_percentage() / 100;
        const auto vd_samples = (m_samples - tr_samples) / 2;
        const auto te_samples = m_samples - tr_samples - vd_samples;

        auto rng = make_rng();
        auto udistd = make_udist<tensor_size_t>(0, 2);
        auto udistc = make_udist<scalar_t>(-1.0, +1.0);

        m_tr_cluster = cluster_t{tr_samples, this->groups()};
        m_vd_cluster = cluster_t{vd_samples, this->groups()};
        m_te_cluster = cluster_t{te_samples, this->groups()};

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

        for (size_t f = 0; f < folds(); ++ f)
        {
            this->split(f) = split_t{std::make_tuple(
                arange(0, tr_samples),
                arange(tr_samples, tr_samples + vd_samples),
                arange(tr_samples + vd_samples, m_samples))
            };
        }
        return true;
    }

    [[nodiscard]] feature_t tfeature() const override
    {
        return feature_t{"wlearner+noise"};
    }

    [[nodiscard]] feature_t ifeature(const tensor_size_t index) const override
    {
        UTEST_REQUIRE_LESS_EQUAL(0, index);
        UTEST_REQUIRE_LESS(index, features());

        feature_t feature(scat("feature", index));
        if (is_discrete(index))
        {
            // discrete, optional
            feature.placeholder("N/A");
            feature.labels({"cat1", "cat2", "cat3"});
            UTEST_REQUIRE(feature.discrete());
            UTEST_REQUIRE(feature.optional());
        }
        else
        {
            // continuous, optional
            feature.placeholder("N/A");
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

    void assign(const tensor_size_t sample, const tensor_size_t group)
    {
        const auto tr_samples = m_samples * train_percentage() / 100;
        const auto vd_samples = (m_samples - tr_samples) / 2;

        UTEST_REQUIRE_LESS_EQUAL(0, sample);
        UTEST_REQUIRE_LESS(sample, m_samples);

        if (sample < tr_samples)
        {
            m_tr_cluster.assign(sample, group);
        }
        else if (sample < tr_samples + vd_samples)
        {
            m_vd_cluster.assign(sample - tr_samples, group);
        }
        else
        {
            m_te_cluster.assign(sample - tr_samples - vd_samples, group);
        }
    }

    [[nodiscard]] virtual bool is_discrete(const tensor_size_t feature) const
    {
        return (feature % 2) == 0;
    }

    [[nodiscard]] virtual bool is_optional(const tensor_size_t sample, const tensor_size_t feature) const
    {
        return (sample + feature) % 23 == 0;
    }

    [[nodiscard]] tensor_size_t get_feature(const bool discrete) const
    {
        return get_feature(isize(), discrete);
    }

    [[nodiscard]] tensor_size_t get_feature(tensor_size_t feature, const bool discrete) const
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

    [[nodiscard]] tensor_size_t isize() const { return m_isize; }
    [[nodiscard]] tensor_size_t tsize() const { return m_tsize; }
    [[nodiscard]] const auto& tr_cluster() const { return m_tr_cluster; }
    [[nodiscard]] const auto& vd_cluster() const { return m_vd_cluster; }
    [[nodiscard]] const auto& te_cluster() const { return m_te_cluster; }

    [[nodiscard]] const auto& cluster(const fold_t fold) const
    {
        switch (fold.m_protocol)
        {
        case protocol::train:   return tr_cluster();
        case protocol::valid:   return vd_cluster();
        default:                return te_cluster();
        }
    }

    [[nodiscard]] auto scale(tensor_size_t sample)
    {
        return m_scales.tensor(sample);
    }
    [[nodiscard]] auto scale(tensor_size_t sample) const
    {
        return m_scales.tensor(sample);
    }
    [[nodiscard]] auto scales(const fold_t fold) const
    {
        return m_scales.indexed<scalar_t>(indices(fold));
    }

private:

    // attributes
    tensor_size_t       m_isize{10};        ///<
    tensor_size_t       m_tsize{1};         ///<
    tensor_size_t       m_samples{100};     ///< total number of samples to generate (train + validation + test)
    cluster_t           m_tr_cluster;       ///< split of the training samples using the ground truth feature
    cluster_t           m_vd_cluster;       ///< split of the validation samples using the ground truth feature
    cluster_t           m_te_cluster;       ///< split of the testing samples using the ground truth feature
    tensor4d_t          m_scales;           ///<
};

template <typename tdataset>
class no_discrete_features_dataset_t final : public tdataset
{
public:

    no_discrete_features_dataset_t() = default;

    [[nodiscard]] bool is_discrete(const tensor_size_t) const override
    {
        return false;
    }
};

template <typename tdataset>
class no_continuous_features_dataset_t final : public tdataset
{
public:

    no_continuous_features_dataset_t() = default;

    [[nodiscard]] bool is_discrete(const tensor_size_t) const override
    {
        return true;
    }
};

template <typename tdataset>
class different_discrete_feature_dataset_t final : public tdataset
{
public:

    different_discrete_feature_dataset_t() = default;

    [[nodiscard]] feature_t ifeature(const tensor_size_t index) const override
    {
        auto feature = tdataset::ifeature(index);
        if (index == tdataset::the_discrete_feature())
        {
            feature.labels({"cat1", "more", "more", "too many"});
        }
        return feature;
    }
};

inline auto make_fold()
{
    return fold_t{0, protocol::train};
}

inline auto make_loss()
{
    auto loss = loss_t::all().get("squared");
    UTEST_REQUIRE(loss);
    return loss;
}

template <typename tdataset>
auto make_dataset(const tensor_size_t isize = 10, const tensor_size_t tsize = 1, const tensor_size_t samples = 100)
{
    auto dataset = tdataset{};
    dataset.folds(1);
    dataset.isize(isize);
    dataset.tsize(tsize);
    dataset.samples(samples);
    dataset.train_percentage(80);
    UTEST_CHECK_NOTHROW(dataset.load());
    return dataset;
}

template <typename twlearner>
auto make_wlearner(const ::nano::wlearner type)
{
    auto wlearner = twlearner{};
    wlearner.type(type);
    return wlearner;
}

inline auto make_indices(const dataset_t& dataset, fold_t fold)
{
    return arange(0, dataset.samples(fold));
}

inline auto make_invalid_indices(const dataset_t& dataset, fold_t fold)
{
    const auto samples = dataset.samples(fold);
    auto indices = arange(0, samples);
    UTEST_REQUIRE_GREATER(samples, 1);
    indices(0) = indices(1) + 1; // NB: valid indices should be sorted!
    return indices;
}

inline auto make_residuals(const dataset_t& dataset, fold_t fold, const loss_t& loss)
{
    tensor4d_t outputs(cat_dims(dataset.samples(fold), dataset.tdim()));
    outputs.constant(+0.0);

    tensor4d_t residuals;
    loss.vgrad(dataset.targets(fold), outputs, residuals);
    return residuals;
}

inline void check_fit(const fixture_dataset_t& dataset, fold_t fold, wlearner_t& wlearner)
{
    const auto loss = make_loss();
    const auto indices = make_indices(dataset, fold);
    const auto residuals = make_residuals(dataset, fold, *loss);

    auto fit_score = feature_t::placeholder_value();
    UTEST_REQUIRE(!std::isfinite(fit_score));
    UTEST_REQUIRE_NOTHROW(fit_score = wlearner.fit(dataset, fold, residuals, indices, dataset.scales(fold)));
    UTEST_REQUIRE(std::isfinite(fit_score));
    UTEST_CHECK_EQUAL(wlearner.odim(), dataset.tdim());
}

inline void check_no_fit(const fixture_dataset_t& dataset, fold_t fold, wlearner_t& wlearner)
{
    const auto loss = make_loss();
    const auto indices = make_indices(dataset, fold);
    const auto residuals = make_residuals(dataset, fold, *loss);

    auto fit_score = feature_t::placeholder_value();
    UTEST_REQUIRE(!std::isfinite(fit_score));
    UTEST_REQUIRE_NOTHROW(fit_score = wlearner.fit(dataset, fold, residuals, indices, dataset.scales(fold)));
    UTEST_REQUIRE(std::isfinite(fit_score));
    UTEST_CHECK_EQUAL(fit_score, std::numeric_limits<scalar_t>::max());
}

inline void check_fit_throws(const fixture_dataset_t& dataset, fold_t fold, wlearner_t& wlearner)
{
    const auto loss = make_loss();
    const auto indices = make_indices(dataset, fold);
    const auto residuals = make_residuals(dataset, fold, *loss);

    auto fit_score = feature_t::placeholder_value();
    UTEST_REQUIRE(!std::isfinite(fit_score));
    UTEST_REQUIRE_THROW(fit_score = wlearner.fit(dataset, fold, residuals, indices, dataset.scales(fold)), std::runtime_error);
}

inline void check_split(const dataset_t& dataset, fold_t fold, const cluster_t& gcluster, const wlearner_t& wlearner)
{
    const auto indices = make_indices(dataset, fold);

    cluster_t wcluster;
    UTEST_CHECK_NOTHROW(wcluster = wlearner.split(dataset, fold, indices));

    UTEST_REQUIRE_EQUAL(wcluster.samples(), indices.size());
    UTEST_REQUIRE_EQUAL(wcluster.samples(), gcluster.samples());

    UTEST_REQUIRE_EQUAL(wcluster.groups(), gcluster.groups());
    for (tensor_size_t g = 0; g < gcluster.groups(); ++ g)
    {
        UTEST_REQUIRE_EQUAL(wcluster.count(g), gcluster.count(g));
        UTEST_CHECK_EQUAL(wcluster.indices(g), gcluster.indices(g));
    }
}

inline void check_split(const fixture_dataset_t& dataset, const wlearner_t& wlearner)
{
    check_split(dataset, fold_t{0, protocol::train}, dataset.tr_cluster(), wlearner);
    check_split(dataset, fold_t{0, protocol::valid}, dataset.vd_cluster(), wlearner);
    check_split(dataset, fold_t{0, protocol::test}, dataset.te_cluster(), wlearner);
}

inline void check_split_throws(const dataset_t& dataset, fold_t fold, const indices_t& indices, const wlearner_t& wlearner)
{
    cluster_t wcluster;
    UTEST_CHECK_THROW(wcluster = wlearner.split(dataset, fold, indices), std::runtime_error);
}

inline void predict(const dataset_t& dataset, fold_t fold, const wlearner_t& wlearner, tensor4d_t& outputs)
{
    outputs.resize(cat_dims(dataset.samples(fold), dataset.tdim()));
    dataset.loop(execution::seq, fold, wlearner.batch(), [&] (tensor_range_t range, size_t)
    {
        wlearner.predict(dataset, fold, range, outputs.slice(range));
    });
}

inline void check_predict(const fixture_dataset_t& dataset, fold_t fold, const wlearner_t& wlearner)
{
    const auto inputs = dataset.inputs(fold);
    const auto scales = dataset.scales(fold);
    const auto targets = dataset.targets(fold);
    const auto imatrix = inputs.reshape(dataset.samples(fold), -1);

    const auto& cluster = dataset.cluster(fold);
    const auto tsize = ::nano::size(dataset.tdim());

    tensor4d_t outputs;
    UTEST_REQUIRE_NOTHROW(predict(dataset, fold, wlearner, outputs));

    UTEST_REQUIRE_EQUAL(imatrix.rows(), cluster.samples());
    for (tensor_size_t s = 0; s < imatrix.rows(); ++ s)
    {
        if (cluster.group(s) < 0)
        {
            UTEST_CHECK_EIGEN_CLOSE(outputs.vector(s), vector_t::Zero(tsize), 1e-8);
        }
        else if (wlearner.type() == ::nano::wlearner::real)
        {
            UTEST_CHECK_EIGEN_CLOSE(outputs.array(s) * scales.array(s), targets.array(s), 1e-8);
        }
        else
        {
            UTEST_CHECK_EIGEN_CLOSE(outputs.array(s), targets.array(s).sign(), 1e-8);
        }
    }
}

inline void check_predict_throws(const dataset_t& dataset, fold_t fold, const wlearner_t& wlearner)
{
    tensor4d_t outputs;
    UTEST_CHECK_THROW(predict(dataset, fold, wlearner, outputs), std::runtime_error);
}


inline void check_scale(const fixture_dataset_t& dataset, fold_t fold, wlearner_t& wlearner)
{
    tensor4d_t outputs, outputs_scaled;
    UTEST_CHECK_NOTHROW(predict(dataset, fold, wlearner, outputs));

    const auto& cluster = dataset.cluster(fold);
    {
        vector_t scale = vector_t::Constant(1, 2.0);

        UTEST_CHECK_NOTHROW(wlearner.scale(scale));
        UTEST_CHECK_NOTHROW(predict(dataset, fold, wlearner, outputs_scaled));
        UTEST_CHECK_EIGEN_CLOSE(outputs.array() * scale(0), outputs_scaled.array(), 1e-8);

        scale = vector_t::Constant(1, 0.5);
        UTEST_CHECK_NOTHROW(wlearner.scale(scale));
    }
    if (cluster.groups() != 1)
    {
        vector_t scale = vector_t::Random(cluster.groups());
        scale.array() += 2.0;

        UTEST_CHECK_NOTHROW(wlearner.scale(scale));
        UTEST_CHECK_NOTHROW(predict(dataset, fold, wlearner, outputs_scaled));
        for (tensor_size_t s = 0; s < cluster.samples(); ++ s)
        {
            const auto group = cluster.group(s);
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
        UTEST_CHECK_EQUAL(iwlearner.type(), wlearner.type());
        UTEST_CHECK_EQUAL(iwlearner.batch(), wlearner.batch());
        return iwlearner;
    }
}
