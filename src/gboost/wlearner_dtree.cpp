#include <set>
#include <deque>
#include <iomanip>
#include <nano/logger.h>
#include <nano/gboost/util.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_dtree.h>
#include <nano/gboost/wlearner_stump.h>
#include <nano/gboost/wlearner_table.h>

using namespace nano;

namespace
{
    class cache_t
    {
    public:

        cache_t() = default;

        explicit cache_t(indices_t samples) :
            m_samples(std::move(samples))
        {
        }

        // attributes
        indices_t       m_samples;      ///<
        tensor_size_t   m_depth{0};     ///<
        tensor3d_t      m_table;        ///<
        size_t          m_parent{0};    ///<
    };

    std::istream& read(std::istream& stream, std::vector<dtree_node_t>& nodes)
    {
        uint32_t count = 0;
        if (!::nano::read(stream, count))
        {
            stream.setstate(std::ios_base::failbit);
            return stream;
        }

        nodes.resize(count);
        for (auto& node : nodes)
        {
            if (!::nano::read_cast<int32_t>(stream, node.m_feature) ||
                !::nano::read_cast<int32_t>(stream, node.m_classes) ||
                !::nano::read(stream, node.m_threshold) ||
                !::nano::read_cast<uint32_t>(stream, node.m_next) ||
                !::nano::read_cast<int32_t>(stream, node.m_table))
            {
                stream.setstate(std::ios_base::failbit);
                return stream;
            }
        }
        return stream;
    }

    std::ostream& write(std::ostream& stream, const std::vector<dtree_node_t>& nodes)
    {
        if (!::nano::write(stream, static_cast<uint32_t>(nodes.size())))
        {
            stream.setstate(std::ios_base::failbit);
            return stream;
        }

        for (const auto& node : nodes)
        {
            if (!::nano::write(stream, static_cast<int32_t>(node.m_feature)) ||
                !::nano::write(stream, static_cast<int32_t>(node.m_classes)) ||
                !::nano::write(stream, node.m_threshold) ||
                !::nano::write(stream, static_cast<uint32_t>(node.m_next)) ||
                !::nano::write(stream, static_cast<int32_t>(node.m_table)))
            {
                stream.setstate(std::ios_base::failbit);
                return stream;
            }
        }
        return stream;
    }

    void append(tensor4d_t& tables, const tensor3d_cmap_t& table)
    {
        // NB: This conservative resize is not very efficient!
        const auto copy = tables;
        const auto count = tables.size<0>();
        tables.resize(cat_dims(count + 1, table.dims()));
        if (count > 0)
        {
            tables.slice(0, count) = copy;
        }
        tables.tensor(count) = table;
    }

    void update(dtree_nodes_t& nodes, indices_t& features)
    {
        // gather the unique set of selected features
        std::set<tensor_size_t> ufeatures;
        for (const auto& node : nodes)
        {
            if (node.m_feature >= 0)
            {
                ufeatures.insert(node.m_feature);
            }
        }

        features.resize(static_cast<tensor_size_t>(ufeatures.size()));

        auto it = ufeatures.begin();
        for (tensor_size_t i = 0; i < features.size(); ++ i)
        {
            features(i) = *(it ++);
        }

        // map nodes to the unique set of selected features
        for (auto& node : nodes)
        {
            if (node.m_feature >= 0)
            {
                const auto pos = ufeatures.find(node.m_feature);
                node.m_feature = static_cast<tensor_size_t>(std::distance(ufeatures.begin(), pos));
            }
        }
    }

    template <typename toperator>
    void evaluate(const dtree_nodes_t& nodes, const tensor4d_t& tables, const tensor1d_cmap_t& fvalues, const toperator& op)
    {
        for (size_t inode = 0, dnode = 0; ; )
        {
            const auto& node = nodes[inode];

            // split based on a feature's value ...
            critical(
                node.m_feature < 0 || node.m_feature >= fvalues.size(),
                "dtree weak learner: out-of-range feature index!");

            const auto x = fvalues(node.m_feature);
            if (feature_t::missing(x))
            {
                op(-1);
                break;
            }

            else
            {
                // ... discrete feature
                if (node.m_classes > 0)
                {
                    const auto iclass = static_cast<tensor_size_t>(x);
                    critical(
                        iclass < 0 || iclass >= node.m_classes,
                        "dtree weak learner: out-of-range discrete feature!");

                    dnode = static_cast<size_t>(iclass);
                }

                // ... continuous feature
                else
                {
                    dnode = static_cast<size_t>(x < node.m_threshold ? 0U : 1U);
                }

                critical(
                    inode + dnode >= nodes.size(),
                    "dtree weak learner: out-of-range node index!");

                const auto next = nodes[inode + dnode].m_next;
                if (next <= inode)
                {
                    // no jump, so it must be a valid terminal node
                    inode += dnode;
                    critical(
                        inode >= nodes.size(),
                        "dtree weak learner: out-of-range node index!");
                    critical(
                        nodes[inode].m_table < 0 || nodes[inode].m_table >= tables.size<0>(),
                        "dtree weak learner: out-of-range table index!");

                    op(nodes[inode].m_table);
                    break;
                }
                else
                {
                    // jump to new node
                    inode = next;
                    critical(
                        inode >= nodes.size(),
                        "dtree weak learner: out-of-range node index!");
                }
            }
        }
    }
}

wlearner_dtree_t::wlearner_dtree_t() = default;

void wlearner_dtree_t::max_depth(const int max_depth)
{
    m_max_depth = max_depth;
}

void wlearner_dtree_t::min_split(const int min_split)
{
    m_min_split = min_split;
}

void wlearner_dtree_t::read(std::istream& stream)
{
    int32_t idepth = 0;
    int32_t isplit = 0;

    wlearner_t::read(stream);
    critical(
        !::nano::read(stream, idepth) ||
        !::nano::read(stream, isplit) ||
        !::read(stream, m_nodes) ||
        !::read(stream, m_features) ||
        !::nano::read(stream, m_tables),
        "dtree weak learner: failed to read from stream!");

    max_depth(idepth);
    min_split(isplit);
}

void wlearner_dtree_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);
    critical(
        !::nano::write(stream, static_cast<int32_t>(max_depth())) ||
        !::nano::write(stream, static_cast<int32_t>(min_split())) ||
        !::write(stream, m_nodes) ||
        !::write(stream, m_features) ||
        !::nano::write(stream, m_tables),
        "dtree weak learner: failed to write to stream!");
}

rwlearner_t wlearner_dtree_t::clone() const
{
    return std::make_unique<wlearner_dtree_t>(*this);
}

void wlearner_dtree_t::scale(const vector_t& scale)
{
    wlearner_t::scale(m_tables, scale);
}

scalar_t wlearner_dtree_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.tdims()));

    scalar_t score = 0;

    m_nodes.clear();
    m_tables.resize(cat_dims(0, dataset.tdims()));

    auto stump = wlearner_stump_t{};
    auto table = wlearner_table_t{};

    const auto min_samples_size = std::min<tensor_size_t>(10, dataset.samples() * min_split() / 100);

    std::deque<cache_t> caches;
    caches.emplace_back(samples);
    while (!caches.empty())
    {
        const auto cache = caches.front();

        // split the node using both discrete and continuous features...
        log_info() << std::fixed << std::setprecision(8)
            << " +++ depth=" << cache.m_depth << ",samples=" << cache.m_samples.size()
            << ",score=" << (score == wlearner_t::no_fit_score() ? scat("N/A") : scat(score)) << "...";
        const auto score_stump = stump.fit(dataset, cache.m_samples, gradients);
        const auto score_table = table.fit(dataset, cache.m_samples, gradients);

        cluster_t cluster;
        tensor4d_t tables;
        dtree_node_t node;

        cache_t ncache;
        ncache.m_depth = cache.m_depth + 1;

        if (score_stump < score_table)
        {
            tables = stump.tables();
            cluster = stump.split(dataset, cache.m_samples);

            node.m_feature = stump.feature();
            node.m_threshold = stump.threshold();
        }
        else
        {
            tables = table.tables();
            cluster = table.split(dataset, cache.m_samples);

            node.m_feature = table.feature();
            node.m_classes = tables.size<0>();
        }
        assert(cluster.groups() == tables.size<0>());

        // have the parent node point to the current terminal node (to be added)
        if (cache.m_parent < m_nodes.size())
        {
            m_nodes[cache.m_parent].m_next = m_nodes.size();
        }

        // terminal nodes...
        if (cache.m_samples.size() < min_samples_size || (cache.m_depth + 1) >= max_depth())
        {
            for (tensor_size_t i = 0, size = tables.size<0>(); i < size; ++ i)
            {
                ncache.m_parent = m_nodes.size();
                ncache.m_samples = cluster.indices(i);

                node.m_table = m_tables.size<0>();
                m_nodes.emplace_back(node);
                append(m_tables, tables.tensor(i));
            }

            // also, update the total score
            score += std::min(score_table, score_stump);
        }

        // can still split the samples
        else
        {
            for (tensor_size_t i = 0, size = tables.size<0>(); i < size; ++ i)
            {
                ncache.m_parent = m_nodes.size();
                ncache.m_samples = cluster.indices(i);

                node.m_table = -1;
                m_nodes.push_back(node);
                caches.push_back(ncache);
            }
        }

        caches.pop_front();
    }

    // OK, compact the selected features
    ::update(m_nodes, m_features);

    log_info() << std::fixed << std::setprecision(8) << " === tree(features="
        << m_features.size() << ",nodes=" << m_nodes.size() << "), score=" << score << ".";

    return score;
}

void wlearner_dtree_t::compatible(const dataset_t& dataset) const
{
    critical(
        m_tables.size<0>() == 0,
        "dtree weak learner: empty weak learner!");

    critical(
        make_dims(m_tables.size<1>(), m_tables.size<2>(), m_tables.size<3>()) != dataset.tdims() ||
        m_features.min() < 0 || m_features.max() >= dataset.features(),
        "dtree weak learner: mis-matching dataset!");

    for (const auto& node : m_nodes)
    {
        if (node.m_feature < 0)
        {
            continue;
        }

        const auto classes = node.m_classes;
        const auto feature = node.m_feature;

        critical(
            feature >= m_features.size() ||
            m_features(feature) >= dataset.features() ||
            dataset.feature(m_features(feature)).discrete() != (classes > 0) ||
            dataset.feature(m_features(feature)).labels().size() != static_cast<size_t>(std::max(tensor_size_t(0), classes)),
            "dtree weak learner: mis-matching dataset!");
    }
}

void wlearner_dtree_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    compatible(dataset);

    for (tensor_size_t begin = 0; begin < samples.size(); begin += batch())
    {
        const auto end = std::min(samples.size(), begin + static_cast<tensor_size_t>(batch()));
        const auto range = make_range(begin, end);
        const auto fvalues = dataset.inputs(samples.slice(range), m_features);
        for (tensor_size_t i = begin; i < end; ++ i)
        {
            ::evaluate(m_nodes, m_tables, fvalues.tensor(i - begin), [&] (const tensor_size_t table)
            {
                if (table >= 0)
                {
                    outputs.vector(i) += m_tables.vector(table);
                }
            });
        }
    }
}

cluster_t wlearner_dtree_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    compatible(dataset);
    wlearner_t::check(samples);

    cluster_t cluster(dataset.samples(), m_tables.size());
    loopr(samples.size(), batch(), [&] (tensor_size_t begin, tensor_size_t end, size_t)
    {
        const auto range = make_range(begin, end);
        const auto fvalues = dataset.inputs(samples.slice(range), m_features);
        for (tensor_size_t i = begin; i < end; ++ i)
        {
            ::evaluate(m_nodes, m_tables, fvalues.tensor(i - begin), [&] (const tensor_size_t table)
            {
                if (table >= 0)
                {
                    cluster.assign(i, table);
                }
            });
        }
    });

    return cluster;
}

indices_t wlearner_dtree_t::features() const
{
    return m_features;
}
