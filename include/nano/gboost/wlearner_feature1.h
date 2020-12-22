#pragma once

#include <nano/gboost/wlearner.h>

namespace nano
{
    ///
    /// \brief interface for weak learner that are parametrized by a single feature,
    ///     that is either continuous or discrete.
    ///
    /// NB: the invalid features and the missing feature values are skipped during fiting.
    ///
    class NANO_PUBLIC wlearner_feature1_t : public wlearner_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        wlearner_feature1_t();

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
        void scale(const vector_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        indices_t features() const override;

        ///
        /// \brief access functions
        ///
        auto feature() const { return m_feature; }
        const auto& tables() const { return m_tables; }
        auto vector(tensor_size_t i) { return m_tables.array(i); }
        auto vector(tensor_size_t i) const { return m_tables.vector(i); }

    protected:

        void compatible(const dataset_t&) const;

        template <typename toperator>
        static void loopc(const dataset_t& dataset, const indices_t& samples, const toperator& op)
        {
            loopi(dataset.features(), [&] (tensor_size_t feature, size_t tnum)
            {
                const auto& ifeature = dataset.feature(feature);
                if (!ifeature.discrete())
                {
                    const auto fvalues = dataset.inputs(samples, feature);
                    op(feature, fvalues, tnum);
                }
            });
        }

        template <typename toperator>
        static void loopd(const dataset_t& dataset, const indices_t& samples, const toperator& op)
        {
            loopi(dataset.features(), [&] (tensor_size_t feature, size_t tnum)
            {
                const auto& ifeature = dataset.feature(feature);
                if (ifeature.discrete())
                {
                    const auto n_fvalues = static_cast<tensor_size_t>(ifeature.labels().size());
                    const auto fvalues = dataset.inputs(samples, feature);
                    op(feature, fvalues, n_fvalues, tnum);
                }
            });
        }

        template <typename toperator>
        void predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs,
            const toperator& op) const
        {
            compatible(dataset);

            assert(outputs.dims() == cat_dims(samples.size(), dataset.tdims()));
            for (tensor_size_t begin = 0; begin < samples.size(); begin += batch())
            {
                const auto end = std::min(samples.size(), begin + static_cast<tensor_size_t>(batch()));
                const auto range = make_range(begin, end);
                const auto fvalues = dataset.inputs(samples.slice(range), m_feature);
                for (tensor_size_t i = begin; i < end; ++ i)
                {
                    const auto x = fvalues(i - begin);
                    if (!feature_t::missing(x))
                    {
                        op(x, outputs.tensor(i));
                    }
                }
            }
        }

        template <typename toperator>
        cluster_t split(
            const dataset_t& dataset, const indices_t& samples, tensor_size_t groups, const toperator& op) const
        {
            compatible(dataset);
            wlearner_t::check(samples);

            cluster_t cluster(dataset.samples(), groups);
            loopr(samples.size(), batch(), [&] (tensor_size_t begin, tensor_size_t end, size_t)
            {
                const auto range = make_range(begin, end);
                const auto fvalues = dataset.inputs(samples.slice(range), m_feature);
                for (tensor_size_t i = begin; i < end; ++ i)
                {
                    const auto x = fvalues(i - begin);
                    if (!feature_t::missing(x))
                    {
                        cluster.assign(samples(i), op(x));
                    }
                }
            });

            return cluster;
        }

        void set(tensor_size_t feature, const tensor4d_t& tables, size_t labels = 0);

    private:

        // attributes
        size_t          m_labels{0};        ///< expected number of labels if discrete
        tensor_size_t   m_feature{-1};      ///< index of the selected feature
        tensor4d_t      m_tables;           ///< coefficients (:, #outputs)
    };
}
