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
        [[nodiscard]] indices_t features() const override;

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto feature() const { return m_feature; }
        [[nodiscard]] const auto& tables() const { return m_tables; }
        [[nodiscard]] auto vector(tensor_size_t i) const { return m_tables.vector(i); }

    protected:

        void compatible(const dataset_t&) const;

        template <typename toperator>
        void predict(const dataset_t& dataset, fold_t fold, tensor_range_t range, tensor4d_map_t& outputs,
            const toperator& op) const
        {
            compatible(dataset);

            const auto fvalues = dataset.inputs(fold, range, m_feature);
            for (tensor_size_t i = 0; i < range.size(); ++ i)
            {
                const auto x = fvalues(i);
                if (feature_t::missing(x))
                {
                    outputs.vector(i).setZero();
                }
                else
                {
                    op(x, i);
                }
            }
        }

        template <typename toperator>
        void split(const dataset_t& dataset, fold_t fold, const indices_t& indices, const toperator& op) const
        {
            compatible(dataset);
            wlearner_t::check(indices);

            dataset.loop(execution::par, fold, batch(), [&] (tensor_range_t range, size_t)
            {
                const auto fvalues = dataset.inputs(fold, range, m_feature);
                wlearner_t::for_each(range, indices, [&] (const tensor_size_t i)
                {
                    const auto x = fvalues(i - range.begin());
                    if (!feature_t::missing(x))
                    {
                        op(x, i);
                    }
                });
            });
        }

        void set(tensor_size_t feature, const tensor4d_t& tables, size_t labels = 0);

    private:

        // attributes
        size_t          m_labels{0};        ///< expected number of labels if discrete
        tensor_size_t   m_feature{-1};      ///< index of the selected feature
        tensor4d_t      m_tables;           ///< (:, #outputs)
    };
}
