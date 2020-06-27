#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"
#include <nano/gboost/wlearner_dtree.h>

using namespace nano;

class wdtree_dataset_t : public fixture_dataset_t
{
public:

    wdtree_dataset_t() = default;

    [[nodiscard]] virtual int min_split() const = 0;
    [[nodiscard]] virtual int max_depth() const = 0;
    [[nodiscard]] virtual bool can_discrete() const = 0;
    [[nodiscard]] virtual indices_t features() const = 0;
    [[nodiscard]] virtual tensor4d_t rtables() const = 0;
    [[nodiscard]] virtual tensor4d_t dtables() const = 0;
    [[nodiscard]] virtual dtree_nodes_t nodes() const = 0;
};

class wdtree_stump1_dataset_t : public wdtree_dataset_t
{
public:

    wdtree_stump1_dataset_t() = default;

    [[nodiscard]] int min_split() const override { return 1; }
    [[nodiscard]] int max_depth() const override { return 1; }
    [[nodiscard]] bool can_discrete() const override { return true; }
    [[nodiscard]] tensor_size_t groups() const override { return 2; }
    [[nodiscard]] tensor_size_t feature(bool discrete = false) const { return get_feature(discrete); }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(make_stump_target(sample, feature(), 5, 1.5, -4.0, +3.7, 0));
    }

    [[nodiscard]] indices_t features() const override
    {
        return std::array<tensor_size_t, 1>{{feature()}};
    }

    [[nodiscard]] tensor4d_t rtables() const override
    {
        return {make_dims(2, 1, 1, 1), std::array<scalar_t, 2>{{-4.0, +3.7}}};
    }

    [[nodiscard]] tensor4d_t dtables() const override
    {
        return {make_dims(2, 1, 1, 1), std::array<scalar_t, 2>{{-1.0, +1.0}}};
    }

    [[nodiscard]] dtree_nodes_t nodes() const override
    {
        return
        {
            dtree_node_t{+0, -1, 1.5, 0U, +0},
            dtree_node_t{+0, -1, 1.5, 0U, +1}
        };
    }
};

class wdtree_table1_dataset_t : public wdtree_dataset_t
{
public:

    wdtree_table1_dataset_t() = default;

    [[nodiscard]] int min_split() const override { return 1; }
    [[nodiscard]] int max_depth() const override { return 1; }
    [[nodiscard]] bool can_discrete() const override { return true; }
    [[nodiscard]] tensor_size_t groups() const override { return 3; }
    [[nodiscard]] tensor_size_t the_discrete_feature() const { return feature(); }
    [[nodiscard]] tensor_size_t feature(bool discrete = true) const { return get_feature(discrete); }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(make_table_target(sample, feature(), 3, 5.0, 0));
    }

    [[nodiscard]] indices_t features() const override
    {
        return std::array<tensor_size_t, 1>{{feature()}};
    }

    [[nodiscard]] tensor4d_t rtables() const override
    {
        return {make_dims(3, 1, 1, 1), std::array<scalar_t, 3>{{-5.0, +0.0, +5.0}}};
    }

    [[nodiscard]] tensor4d_t dtables() const override
    {
        return {make_dims(3, 1, 1, 1), std::array<scalar_t, 3>{{-1.0, +0.0, +1.0}}};
    }

    [[nodiscard]] dtree_nodes_t nodes() const override
    {
        return
        {
            dtree_node_t{+0, +3, 0, 0U, +0},
            dtree_node_t{+0, +3, 0, 0U, +1},
            dtree_node_t{+0, +3, 0, 0U, +2}
        };
    }
};

class wdtree_depth2_dataset_t : public wdtree_dataset_t
{
public:

    wdtree_depth2_dataset_t() = default;

    [[nodiscard]] int min_split() const override { return 1; }
    [[nodiscard]] int max_depth() const override { return 2; }
    [[nodiscard]] bool can_discrete() const override { return true; }
    [[nodiscard]] tensor_size_t groups() const override { return 6; }
    [[nodiscard]] tensor_size_t the_discrete_feature() const { return feature0(); }
    [[nodiscard]] tensor_size_t feature0(bool discrete = true) const { return get_feature(discrete); }
    [[nodiscard]] tensor_size_t feature10(bool discrete = false) const { return get_feature(discrete); }
    [[nodiscard]] tensor_size_t feature11(bool discrete = false) const { return get_feature(feature10(), discrete); }
    [[nodiscard]] tensor_size_t feature12(bool discrete = false) const { return get_feature(feature11(), discrete); }

    void make_target(const tensor_size_t sample) override
    {
        auto input = this->input(sample);

        const auto tf0 = feature0();
        if (!feature_t::missing(input(tf0)))
        {
            input(tf0) = static_cast<scalar_t>(sample % 3);
            switch (sample % 3)
            {
            case 0:
                target(sample).constant(make_stump_target(sample, feature10(), 5, 3.5, -1.2, +3.4, 0));
                break;

            case 1:
                target(sample).constant(make_stump_target(sample, feature11(), 7, 4.5, -1.3, +3.5, 2));
                break;

            default:
                target(sample).constant(make_stump_target(sample, feature12(), 11, 5.5, -1.4, +3.6, 4));
                break;
            }
        }
    }

    [[nodiscard]] indices_t features() const override
    {
        return std::array<tensor_size_t, 4>{{feature12(), feature11(), feature0(), feature10()}};
    }

    [[nodiscard]] tensor4d_t rtables() const override
    {
        return {make_dims(6, 1, 1, 1), std::array<scalar_t, 6>{{-1.2, +3.4, -1.3, +3.5, -1.4, +3.6}}};
    }

    [[nodiscard]] tensor4d_t dtables() const override
    {
        return {make_dims(6, 1, 1, 1), std::array<scalar_t, 6>{{-1.0, +1.0, -1.0, +1.0, -1.0, +1.0}}};
    }

    [[nodiscard]] dtree_nodes_t nodes() const override
    {
        // NB: features = {5, 7, 8, 9} aka {stump12, stump11, table0, stump10}
        return
        {
            dtree_node_t{+2, +3, 0.0, 3U, -1},
            dtree_node_t{+2, +3, 0.0, 5U, -1},
            dtree_node_t{+2, +3, 0.0, 7U, -1},
            dtree_node_t{+3, -1, 3.5, 0U, +0},
            dtree_node_t{+3, -1, 3.5, 0U, +1},
            dtree_node_t{+1, -1, 4.5, 0U, +2},
            dtree_node_t{+1, -1, 4.5, 0U, +3},
            dtree_node_t{+0, -1, 5.5, 0U, +4},
            dtree_node_t{+0, -1, 5.5, 0U, +5}
        };
    }
};

class wdtree_depth3_dataset_t : public wdtree_dataset_t
{
public:

    wdtree_depth3_dataset_t() = default;

    [[nodiscard]] int min_split() const override { return 1; }
    [[nodiscard]] int max_depth() const override { return 3; }
    [[nodiscard]] bool can_discrete() const override { return false; }
    [[nodiscard]] tensor_size_t groups() const override { return 11; }
    [[nodiscard]] tensor_size_t the_discrete_feature() const { return feature22(); }
    [[nodiscard]] tensor_size_t feature0(bool discrete = false) const { return get_feature(discrete); }
    [[nodiscard]] tensor_size_t feature10(bool discrete = false) const { return get_feature(feature0(), discrete); }
    [[nodiscard]] tensor_size_t feature11(bool discrete = false) const { return get_feature(feature10(), discrete); }
    [[nodiscard]] tensor_size_t feature20(bool discrete = true) const { return get_feature(discrete); }
    [[nodiscard]] tensor_size_t feature21(bool discrete = false) const { return get_feature(feature11(), discrete); }
    [[nodiscard]] tensor_size_t feature22(bool discrete = true) const { return get_feature(feature20(), discrete); }
    [[nodiscard]] tensor_size_t feature23(bool discrete = true) const { return get_feature(feature22(), discrete); }

    void make_target(const tensor_size_t sample) override
    {
        auto input = this->input(sample);
        auto target = this->target(sample);

        const auto tf0 = feature0();
        const auto tf10 = feature10();
        const auto tf11 = feature11();

        if (!feature_t::missing(input(tf0)))
        {
            input(tf0) = static_cast<scalar_t>(sample % 7);
            if ((sample % 7) < 3)
            {
                if (!feature_t::missing(input(tf10)))
                {
                    input(tf10) = static_cast<scalar_t>(sample % 9);
                    if ((sample % 9) < 5)
                    {
                        target.constant(make_table_target(sample, feature20(), 3, 2.0, 0));
                    }
                    else
                    {
                        target.constant(make_stump_target(sample, feature21(), 5, 3.5, +1.9, -0.7, 3));
                    }
                    target.array() += 10.0;
                }
            }
            else
            {
                if (!feature_t::missing(input(tf11)))
                {
                    input(tf11) = static_cast<scalar_t>(sample % 11);
                    if ((sample % 11) < 7)
                    {
                        target.constant(make_table_target(sample, feature22(), 3, 3.0, 5));
                    }
                    else
                    {
                        target.constant(make_table_target(sample, feature23(), 3, 3.0, 8));
                    }
                    target.array() -= 20.0;
                }
            }
        }
    }

    [[nodiscard]] indices_t features() const override
    {
        // NB: features = {3, 4, 5, 6, 7, 8, 9} aka {stump21, table23, stump11, table22, stump10, table20, stump0}
        return std::array<tensor_size_t, 7>{{
            feature21(), feature23(), feature11(), feature22(), feature10(), feature20(), feature0()}};
    }

    [[nodiscard]] tensor4d_t rtables() const override
    {
        return {make_dims(11, 1, 1, 1), std::array<scalar_t, 11>{{
            8.0, +10.0, +12.0, +11.9, +9.3, -23.0, -20.0, -17.0, -23.0, -20.0, -17.0}}};
    }

    [[nodiscard]] tensor4d_t dtables() const override
    {
        return {make_dims(11, 1, 1, 1), std::array<scalar_t, 11>{{
            +1.0, +0.0, +1.0, +1.0, +1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0}}};
    }

    [[nodiscard]] dtree_nodes_t nodes() const override
    {
        // NB: features = {3, 4, 5, 6, 7, 8, 9} aka {stump21, table23, stump11, table22, stump10, table20, stump0}
        return
        {
            // stump0
            dtree_node_t{+6, -1, 2.5, 2U, -1},
            dtree_node_t{+6, -1, 2.5, 4U, -1},

            // stump10
            dtree_node_t{+4, -1, 4.5, 6U, -1},
            dtree_node_t{+4, -1, 4.5, 9U, -1},

            // stump11
            dtree_node_t{+2, -1, 6.5, 11U, -1},
            dtree_node_t{+2, -1, 6.5, 14U, -1},

            // table20
            dtree_node_t{+5, +3, 0.0, 0U, +0},
            dtree_node_t{+5, +3, 0.0, 0U, +1},
            dtree_node_t{+5, +3, 0.0, 0U, +2},

            // stump21
            dtree_node_t{+0, -1, 3.5, 0U, +3},
            dtree_node_t{+0, -1, 3.5, 0U, +4},

            // table22
            dtree_node_t{+3, +3, 0.0, 0U, +5},
            dtree_node_t{+3, +3, 0.0, 0U, +6},
            dtree_node_t{+3, +3, 0.0, 0U, +7},

            // table23
            dtree_node_t{+1, +3, 0.0, 0U, +8},
            dtree_node_t{+1, +3, 0.0, 0U, +9},
            dtree_node_t{+1, +3, 0.0, 0U, +10}
        };
    }
};

static auto make_wdtree(const wdtree_dataset_t& dataset, const ::nano::wlearner type)
{
    auto wlearner = make_wlearner<wlearner_dtree_t>(type);
    wlearner.min_split(dataset.min_split());
    wlearner.max_depth(dataset.max_depth());
    return wlearner;
}

template <typename tdataset, typename... targs>
static std::unique_ptr<wdtree_dataset_t> make_datasetw(targs... args)
{
    return std::make_unique<tdataset>(make_dataset<tdataset>(args...));
}

static auto make_datasets()
{
    using udataset = std::unique_ptr<wdtree_dataset_t>;

    std::vector<std::pair<udataset, std::vector<udataset>>> datasets;
    {
        using tdataset = wdtree_stump1_dataset_t;
        auto dataset = make_datasetw<tdataset>();
        auto xdatasets = std::vector<udataset>{};
        xdatasets.emplace_back(make_datasetw<tdataset>(dataset->isize(), dataset->tsize() + 1));
        xdatasets.emplace_back(make_datasetw<tdataset>(dataset->features().max(), dataset->tsize()));
        xdatasets.emplace_back(make_datasetw<no_continuous_features_dataset_t<tdataset>>());
        datasets.emplace_back(std::move(dataset), std::move(xdatasets));
    }
    {
        using tdataset = wdtree_table1_dataset_t;
        auto dataset = make_datasetw<tdataset>();
        auto xdatasets = std::vector<udataset>{};
        xdatasets.emplace_back(make_datasetw<tdataset>(dataset->isize(), dataset->tsize() + 1));
        xdatasets.emplace_back(make_datasetw<tdataset>(dataset->features().max(), dataset->tsize()));
        xdatasets.emplace_back(make_datasetw<no_discrete_features_dataset_t<tdataset>>());
        xdatasets.emplace_back(make_datasetw<different_discrete_feature_dataset_t<tdataset>>());
        datasets.emplace_back(std::move(dataset), std::move(xdatasets));
    }
    {
        using tdataset = wdtree_depth2_dataset_t;
        auto dataset = make_datasetw<tdataset>(10, 1, 400);
        auto xdatasets = std::vector<udataset>{};
        xdatasets.emplace_back(make_datasetw<tdataset>(dataset->isize(), dataset->tsize() + 1));
        xdatasets.emplace_back(make_datasetw<tdataset>(dataset->features().max(), dataset->tsize()));
        xdatasets.emplace_back(make_datasetw<no_discrete_features_dataset_t<tdataset>>());
        xdatasets.emplace_back(make_datasetw<no_continuous_features_dataset_t<tdataset>>());
        xdatasets.emplace_back(make_datasetw<different_discrete_feature_dataset_t<tdataset>>());
        datasets.emplace_back(std::move(dataset), std::move(xdatasets));
    }
    {
        using tdataset = wdtree_depth3_dataset_t;
        auto dataset = make_datasetw<tdataset>(10, 1, 1600);
        auto xdatasets = std::vector<udataset>{};
        xdatasets.emplace_back(make_datasetw<tdataset>(dataset->isize(), dataset->tsize() + 1));
        xdatasets.emplace_back(make_datasetw<tdataset>(dataset->features().max(), dataset->tsize()));
        xdatasets.emplace_back(make_datasetw<no_discrete_features_dataset_t<tdataset>>());
        xdatasets.emplace_back(make_datasetw<no_continuous_features_dataset_t<tdataset>>());
        xdatasets.emplace_back(make_datasetw<different_discrete_feature_dataset_t<tdataset>>());
        datasets.emplace_back(std::move(dataset), std::move(xdatasets));
    }
    return datasets;
}

static const auto& the_datasets()
{
    static const auto datasets = make_datasets();
    return datasets;
}

UTEST_BEGIN_MODULE(test_gboost_wdtree)

// TODO: test min_split, max_depth, leaves begin cut if not enough samples

UTEST_CASE(print)
{
    const auto nodes = dtree_nodes_t
    {
        dtree_node_t{+5, +3, 0.0, 0U, +2},
        dtree_node_t{+0, -1, 3.5, 0U, -1},
    };

    {
        std::stringstream stream;
        stream << nodes[0];
        UTEST_CHECK_EQUAL(stream.str(),
            scat("node: feature=5,classes=3,threshold=", nodes[0].m_threshold, ",next=0,table=2"));
    }
    {
        std::stringstream stream;
        stream << nodes;
        UTEST_CHECK_EQUAL(stream.str(),
            scat("nodes:{\n",
                "\tnode: feature=5,classes=3,threshold=", nodes[0].m_threshold, ",next=0,table=2\n",
                "\tnode: feature=0,classes=-1,threshold=", nodes[1].m_threshold, ",next=0,table=-1\n"
                "}"));
    }
}

UTEST_CASE(fitting)
{
    for (const auto& pdataset : the_datasets())
    {
        const auto fold = make_fold();
        const auto& dataset = *pdataset.first;

        for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
        {
            // check fitting
            auto wlearner = make_wdtree(dataset, type);
            check_fit(dataset, fold, wlearner);

            if (type == ::nano::wlearner::discrete && !dataset.can_discrete())
            {
                continue;
            }

            const auto tables = (type == ::nano::wlearner::real) ? dataset.rtables() : dataset.dtables();

            UTEST_CHECK_EQUAL(wlearner.odim(), dataset.tdim());
            UTEST_CHECK_EQUAL(wlearner.features(), dataset.features());
            UTEST_CHECK_EQUAL(wlearner.nodes(), dataset.nodes());
            UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables.array(), 1e-8);

            // check scaling
            check_scale(dataset, fold, wlearner);

            // check model loading and saving from and to binary streams
            const auto iwlearner = stream_wlearner(wlearner);
            UTEST_CHECK_EQUAL(wlearner.nodes(), iwlearner.nodes());
            UTEST_CHECK_EQUAL(wlearner.features(), iwlearner.features());
            UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), iwlearner.tables().array(), 1e-8);
        }
    }
}

UTEST_CASE(no_fitting)
{
    for (const auto& pdataset : the_datasets())
    {
        const auto fold = make_fold();
        const auto& dataset = *pdataset.first;

        for (const auto type : {static_cast<::nano::wlearner>(-1)})
        {
            auto wlearner = make_wlearner<wlearner_dtree_t>(type);
            check_fit_throws(dataset, fold, wlearner);
        }
    }
}

UTEST_CASE(predict)
{
    for (const auto& pdataset : the_datasets())
    {
        const auto fold = make_fold();
        const auto& dataset = *pdataset.first;

        for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
        {
            auto wlearner = make_wdtree(dataset, type);
            check_predict_throws(dataset, fold, wlearner);

            if (type == ::nano::wlearner::discrete && !dataset.can_discrete())
            {
                continue;
            }

            check_fit(dataset, fold, wlearner);

            check_predict(dataset, fold, wlearner);
            for (const auto& pdatasetx : pdataset.second)
            {
                check_predict_throws(*pdatasetx, fold, wlearner);
            }
        }
    }
}

UTEST_CASE(split)
{
    for (const auto& pdataset : the_datasets())
    {
        const auto fold = make_fold();
        const auto& dataset = *pdataset.first;

        for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
        {
            auto wlearner = make_wdtree(dataset, type);
            check_split_throws(dataset, fold, make_indices(dataset, fold), wlearner);

            if (type == ::nano::wlearner::discrete && !dataset.can_discrete())
            {
                continue;
            }

            check_fit(dataset, fold, wlearner);

            check_split(dataset, wlearner);
            check_split_throws(dataset, fold, make_invalid_indices(dataset, fold), wlearner);
        }
    }
}

UTEST_END_MODULE()
