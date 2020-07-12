#pragma once

#include <nano/gboost/wlearner.h>

namespace nano
{
    class wlearner_dtree_t;

    template <>
    struct factory_traits_t<wlearner_dtree_t>
    {
        static string_t id() { return "dtree"; }
        static string_t description() { return "decision tree weak learner"; }
    };

    ///
    /// \brief node in the decision tree.
    ///
    class dtree_node_t
    {
    public:

        dtree_node_t() = default;

        dtree_node_t(
            tensor_size_t feature, tensor_size_t classes, scalar_t threshold, size_t next, tensor_size_t table) :
            m_feature(feature),
            m_classes(classes),
            m_threshold(threshold),
            m_next(next),
            m_table(table)
        {
        }

        // attributes
        tensor_size_t   m_feature{-1};      ///< feature to evaluate (if a decision node)
        tensor_size_t   m_classes{-1};      ///< number of classes (distinct values), if a discrete feature
        scalar_t        m_threshold{0};     ///< feature value threshold, if a continuous feature
        size_t          m_next{0};          ///< offset to the next node
        tensor_size_t   m_table{-1};        ///< index in the tables at the leaves
    };

    using dtree_nodes_t = std::vector<dtree_node_t>;

    inline bool operator==(const dtree_node_t& lhs, const dtree_node_t& rhs)
    {
        return  lhs.m_feature == rhs.m_feature &&
                lhs.m_classes == rhs.m_classes &&
                std::fabs(lhs.m_threshold - rhs.m_threshold) < 1e-8 &&
                lhs.m_next == rhs.m_next &&
                lhs.m_table == rhs.m_table;
    }

    inline std::ostream& operator<<(std::ostream& os, const dtree_node_t& node)
    {
        return os << "node: feature=" << node.m_feature << ",classes=" << node.m_classes
                  << ",threshold=" << node.m_threshold << ",next=" << node.m_next << ",table=" << node.m_table;
    }

    inline std::ostream& operator<<(std::ostream& os, const dtree_nodes_t& nodes)
    {
        os << "nodes:{\n";
        for (const auto& node : nodes)
        {
            os << "\t" << node << "\n";
        }
        return os << "}";
    }

    ///
    /// \brief a decision tree is a weak learner partitions the data using:
    ///     - look-up-tables for discrete features and
    ///     - decision stumps for continuous features.
    ///
    /// NB: the missing feature values are skipped during fiting.
    /// NB: the splitting feature per level can be either discrete or continuous,
    ///     depending on how well the associated weak learner matches the residuals
    ///     (tables for discrete feature and stumps for continuous features).
    ///
    class NANO_PUBLIC wlearner_dtree_t final : public wlearner_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        wlearner_dtree_t();

        ///
        /// \brief @see wlearner_t
        ///
        void read(std::istream&) override;

        ///
        /// \brief @see wlearner_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        void scale(const vector_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        void predict(const dataset_t&, fold_t, tensor_range_t, tensor4d_map_t&& outputs) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] scalar_t fit(const dataset_t&, fold_t,
            const tensor4d_t&, const indices_t&, const tensor4d_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] cluster_t split(const dataset_t&, fold_t, const indices_t&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        [[nodiscard]] indices_t features() const override;

        ///
        /// \brief change the maximum depth of the tree.
        ///
        /// NB: the effective depth may be smaller, e.g. if not enough samples to further split.
        ///
        void max_depth(int max_depth);

        ///
        /// \brief change the minimum percentage of samples to consider for splitting.
        ///
        /// NB: this is useful to eliminate branches rarely hit.
        ///
        void min_split(int min_split);

        ///
        /// \brief access functions
        ///
        [[nodiscard]] const auto& nodes() const { return m_nodes; }
        [[nodiscard]] const auto& tables() const { return m_tables; }
        [[nodiscard]] auto max_depth() const { return m_max_depth.get(); }
        [[nodiscard]] auto min_split() const { return m_min_split.get(); }

    private:

        void compatible(const dataset_t&) const;

        // attributes
        iparam1_t       m_max_depth{"dtree::max_depth", 1, LE, 3, LE, 10};  ///< maximum depth
        iparam1_t       m_min_split{"dtree::min_split", 1, LE, 5, LE, 10};  ///< minimum ratio of samples to split
        dtree_nodes_t   m_nodes;                ///< nodes in the decision tree
        tensor4d_t      m_tables;               ///< (#feature values, #outputs) - predictions at the leaves
        indices_t       m_features;             ///< unique set of the selected features
    };
}
