#pragma once

#include <nano/loss.h>
#include <nano/dataset.h>
#include <nano/function.h>
#include <nano/parameter.h>

namespace nano
{
    ///
    /// \brief the ERM criterion used for optimizing the parameters of a linear model,
    ///     using a given loss function.
    ///
    /// NB: the ERM loss can be optionally regularized by penalizing:
    ///     - (1) the L1-norm of the weights matrix - like in LASSO
    ///     - (2) the L2-norm of the weights matrix - like in RIDGE (regression)
    ///     - (3) both the L1 and the L2-norms of the weights matrix - like in elastic net regularization
    ///     - (4) the variance of the loss values - like in VadaBoost
    ///
    class NANO_PUBLIC linear_function_t final : public function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        linear_function_t(const loss_t&, const dataset_t&, const indices_t&);

        ///
        /// \brief enable coying
        ///
        linear_function_t(const linear_function_t&) = default;
        linear_function_t& operator=(const linear_function_t&) = delete;

        ///
        /// \brief enable moving
        ///
        linear_function_t(linear_function_t&&) noexcept = default;
        linear_function_t& operator=(linear_function_t&&) noexcept = delete;

        ///
        /// \brief default destructor
        ///
        ~linear_function_t() override = default;

        ///
        /// \brief extract the weight matrix from the given tensor
        ///
        template <typename ttensor>
        auto weights(ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data(), m_isize, m_tsize);
        }

        template <typename ttensor>
        auto weights(const ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data(), m_isize, m_tsize);
        }

        ///
        /// \brief extract the bias vector from the given tensor
        ///
        template <typename ttensor>
        auto bias(ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data() + m_isize * m_tsize, m_tsize);
        }

        template <typename ttensor>
        auto bias(const ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data() + m_isize * m_tsize, m_tsize);
        }

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

        ///
        /// \brief change parameters
        ///
        void l1reg(scalar_t l1reg) { m_l1reg.set(l1reg); }
        void l2reg(scalar_t l2reg) { m_l2reg.set(l2reg); }
        void vAreg(scalar_t vAreg) { m_vAreg.set(vAreg); }
        void batch(tensor_size_t batch) { m_batch.set(batch); }
        void scaling(feature_scaling scaling) { m_scaling = scaling; }

        ///
        /// \brief access functions
        ///
        auto isize() const { return m_isize; }
        auto tsize() const { return m_tsize; }
        auto l1reg() const { return m_l1reg.get(); }
        auto l2reg() const { return m_l2reg.get(); }
        auto vAreg() const { return m_vAreg.get(); }
        auto batch() const { return m_batch.get(); }
        const auto& loss() const { return m_loss; }
        const auto& istats() const { return m_istats; }
        const auto& dataset() const { return m_dataset; }
        const auto& samples() const { return m_samples; }
        auto scaling() const { return m_scaling; }

    private:

        // attributes
        const loss_t&       m_loss;         ///<
        const dataset_t&    m_dataset;      ///<
        const indices_t&    m_samples;      ///<
        tensor_size_t       m_isize{0};     ///< #inputs (e.g. size of the flatten input feature tensor)
        tensor_size_t       m_tsize{0};     ///< #targets (e.g. size of the flatten target tensor, number of classes)
        sparam1_t           m_l1reg{"linear::L1", 0, LE, 0, LE, 1e+8};  ///< regularization factor - see (1), (3)
        sparam1_t           m_l2reg{"linear::L2", 0, LE, 0, LE, 1e+8};  ///< regularization factor - see (2), (3)
        sparam1_t           m_vAreg{"linear::VA", 0, LE, 0, LE, 1e+8};  ///< regularization factor - see (4)
        iparam1_t           m_batch{"linear::batch", 1, LE, 32, LE, 4092};///< batch size in number of samples
        feature_scaling     m_scaling{feature_scaling::none};///<
        elemwise_stats_t    m_istats;       ///< element-wise statistics to be used for feature_scaling
    };
}
