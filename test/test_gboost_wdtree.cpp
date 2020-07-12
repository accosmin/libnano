#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"

using namespace nano;

class wdtree_dataset_t : public fixture_dataset_t
{
public:

    wdtree_dataset_t() = default;

    [[nodiscard]] virtual int min_split() const = 0;
    [[nodiscard]] virtual int max_depth() const = 0;
    [[nodiscard]] virtual indices_t features() const = 0;
    [[nodiscard]] virtual tensor4d_t rtables() const = 0;
    [[nodiscard]] virtual tensor4d_t dtables() const = 0;
    [[nodiscard]] virtual dtree_nodes_t nodes() const = 0;

    void check_wlearner(const wlearner_dtree_t& wlearner) const
    {
        const auto tables = (wlearner.type() == ::nano::wlearner::real) ? rtables() : dtables();
        UTEST_CHECK_EQUAL(wlearner.features(), features());
        UTEST_CHECK_EQUAL(wlearner.nodes(), nodes());
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables.array(), 1e-8);
    }
};

class wdtree_stump1_dataset_t : public wdtree_dataset_t
{
public:

    wdtree_stump1_dataset_t() = default;

    [[nodiscard]] int min_split() const override { return 1; }
    [[nodiscard]] int max_depth() const override { return 1; }
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
            if ((input(tf0) = static_cast<scalar_t>(sample % 7)) < 3.0)
            {
                if (!feature_t::missing(input(tf10)))
                {
                    if ((input(tf10) = static_cast<scalar_t>(sample % 9)) < 5.0)
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
                    if ((input(tf11) = static_cast<scalar_t>(sample % 11)) < 7.0)
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

UTEST_CASE(fitting_stump1)
{
    const auto dataset = make_dataset<wdtree_stump1_dataset_t>();
    const auto datasetx1 = make_dataset<wdtree_stump1_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wdtree_stump1_dataset_t>(dataset.features().max(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<wdtree_stump1_dataset_t>>();

    for (const auto type : {static_cast<::nano::wlearner>(-1)})
    {
        auto wlearner = make_wdtree(dataset, type);
        check_fit_throws(wlearner, dataset);
    }

    for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
    {
        auto wlearner = make_wdtree(dataset, type);
        check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
    }
}

UTEST_CASE(fitting_table1)
{
    const auto dataset = make_dataset<wdtree_table1_dataset_t>();
    const auto datasetx1 = make_dataset<wdtree_table1_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wdtree_table1_dataset_t>(dataset.features().max(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_discrete_features_dataset_t<wdtree_table1_dataset_t>>();
    const auto datasetx4 = make_dataset<different_discrete_feature_dataset_t<wdtree_table1_dataset_t>>();

    for (const auto type : {static_cast<::nano::wlearner>(-1)})
    {
        auto wlearner = make_wdtree(dataset, type);
        check_fit_throws(wlearner, dataset);
    }

    for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
    {
        auto wlearner = make_wdtree(dataset, type);
        check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3, datasetx4);
    }
}

UTEST_CASE(fitting_depth2)
{
    const auto dataset = make_dataset<wdtree_depth2_dataset_t>(10, 1, 400);
    const auto datasetx1 = make_dataset<wdtree_depth2_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wdtree_depth2_dataset_t>(dataset.features().max(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_discrete_features_dataset_t<wdtree_depth2_dataset_t>>();
    const auto datasetx4 = make_dataset<no_continuous_features_dataset_t<wdtree_depth2_dataset_t>>();
    const auto datasetx5 = make_dataset<different_discrete_feature_dataset_t<wdtree_depth2_dataset_t>>();

    for (const auto type : {static_cast<::nano::wlearner>(-1)})
    {
        auto wlearner = make_wdtree(dataset, type);
        check_fit_throws(wlearner, dataset);
    }

    for (const auto type : {::nano::wlearner::real, ::nano::wlearner::discrete})
    {
        auto wlearner = make_wdtree(dataset, type);
        check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3, datasetx4, datasetx5);
    }
}

UTEST_CASE(fitting_depth3)
{
    const auto dataset = make_dataset<wdtree_depth3_dataset_t>(10, 1, 1600);
    const auto datasetx1 = make_dataset<wdtree_depth3_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wdtree_depth3_dataset_t>(dataset.features().max(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_discrete_features_dataset_t<wdtree_depth3_dataset_t>>();
    const auto datasetx4 = make_dataset<no_continuous_features_dataset_t<wdtree_depth3_dataset_t>>();
    const auto datasetx5 = make_dataset<different_discrete_feature_dataset_t<wdtree_depth3_dataset_t>>();

    for (const auto type : {static_cast<::nano::wlearner>(-1)})
    {
        auto wlearner = make_wdtree(dataset, type);
        check_fit_throws(wlearner, dataset);
    }

    for (const auto type : {::nano::wlearner::real})
    {
        auto wlearner = make_wdtree(dataset, type);
        check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3, datasetx4, datasetx5);
    }
}

UTEST_END_MODULE()
