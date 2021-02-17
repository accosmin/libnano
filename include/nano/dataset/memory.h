#pragma once

#include <nano/logger.h>
#include <nano/dataset/dataset.h>

namespace nano
{
    ///
    /// \brief allocate and initialize a tensor bitmask where the last dimension is the number of samples.
    ///
    template <size_t trank>
    inline auto make_mask(const tensor_dims_t<trank>& dims)
    {
        const auto samples = std::get<trank - 1>(dims);

        auto bit_dims = dims;
        bit_dims[trank - 1] = (samples + 7) / 8;

        tensor_mem_t<uint8_t, trank> mask(bit_dims);
        mask.zero();
        return mask;
    }

    ///
    /// \brief mark a feature value as set for a particular sample.
    ///
    inline void setbit(tensor_map_t<uint8_t, 1>& mask, tensor_size_t sample)
    {
        assert(sample >= 0 && sample < (8 * mask.size()));
        mask(sample / 8) |= static_cast<uint8_t>(0x01 << (7 - (sample % 8)));
    }

    ///
    /// \brief check if a feature value exists for a particular sample.
    ///
    inline bool getbit(const tensor_map_t<uint8_t, 1>& mask, tensor_size_t sample)
    {
        assert(sample >= 0 && sample < (8 * mask.size()));
        return (mask(sample / 8) & (0x01 << (7 - (sample % 8)))) != 0x00;
    }

    ///
    /// \brief returns true if the feature is optional (aka some samples haven't been set).
    ///
    inline bool optional(const tensor_map_t<uint8_t, 1>& mask, tensor_size_t samples)
    {
        const auto bytes = samples / 8;
        for (tensor_size_t byte = 0; byte < bytes; ++ byte)
        {
            if (mask(byte) != 0xFF)
            {
                return true;
            }
        }
        for (tensor_size_t sample = 8 * bytes; sample < samples; ++ sample)
        {
            if (!getbit(mask, sample))
            {
                return true;
            }
        }
        return false;
    }

    ///
    /// \brief in-memory dataset that efficiently stores a mixture of different features.
    ///
    /// NB: the features can be optional.
    /// NB: the categorical input features can be single-label or multi-label.
    /// NB: the continuous input features be multi-dimensional as well (::dims() != (1, 1, 1)).
    ///
    class NANO_PUBLIC memory_dataset_t : public dataset_t
    {
    public:

        memory_dataset_t();

        task_type type() const override;
        tensor_size_t samples() const override;

        const auto& istorage() const { return m_inputs; }
        const auto& tstorage() const { return m_target; }
        const feature_storage_t& istorage(tensor_size_t feature) const;

        rfeature_dataset_iterator_t feature_iterator(indices_t samples) const override;
        rflatten_dataset_iterator_t flatten_iterator(indices_t samples) const override;

    protected:

        void resize(tensor_size_t samples, const features_t& features, size_t target);

        template <typename tvalue>
        void set(tensor_size_t sample, const tvalue& value)
        {
            m_target.set(sample, value);
        }

        template <typename tvalue>
        void set(tensor_size_t sample, tensor_size_t feature, const tvalue& value)
        {
            critical(
                feature < 0 || feature >= static_cast<tensor_size_t>(m_inputs.size()),
                "failed to access input feature: index ", feature, " not in [0, ", m_inputs.size());

            m_inputs[static_cast<size_t>(feature)].set(sample, value);
        }

    private:

        using inputs_t = std::vector<feature_storage_t>;
        using target_t = feature_storage_t;

        using inputs_mask_t = tensor_mem_t<uint8_t, 2>;
        using target_mask_t = tensor_mem_t<uint8_t, 1>;

        // attributes
        target_t                m_target;       ///<
        inputs_t                m_inputs;       ///<
        inputs_mask_t           m_inputs_mask;  ///< given is the bit at (feature, sample) is 1
        target_mask_t           m_target_mask;  ///< given is the bit at (sample) is 1
    };

    ///
    /// \brief
    ///
    class NANO_PUBLIC memory_feature_dataset_iterator_t final : public feature_dataset_iterator_t
    {
    public:

        memory_feature_dataset_iterator_t(const memory_dataset_t&, indices_t samples);

        const indices_t& samples() const override;
        bool cache_inputs(int64_t bytes, execution) override;
        bool cache_targets(int64_t bytes, execution) override;
        bool cache_inputs(int64_t bytes, indices_cmap_t features, execution) override;

        feature_t target() const override;
        tensor3d_dims_t target_dims() const override;
        tensor4d_cmap_t targets(tensor_range_t samples, tensor4d_t& buffer) const override;

        tensor_size_t features() const override;
        indices_t scalar_features() const override;
        indices_t struct_features() const override;
        indices_t sclass_features() const override;
        indices_t mclass_features() const override;
        feature_t feature(tensor_size_t feature) const override;

        sindices_cmap_t input(tensor_size_t feature, sindices_t& buffer) const override;
        mindices_cmap_t input(tensor_size_t feature, mindices_t& buffer) const override;
        tensor1d_cmap_t input(tensor_size_t feature, tensor1d_t& buffer) const override;
        tensor4d_cmap_t input(tensor_size_t feature, tensor4d_t& buffer) const override;

        // TODO: need to flatten structured scalar features - one scalar feature per component

    private:

        // attributes
        const memory_dataset_t& m_dataset;  ///<
        indices_t               m_samples;  ///<
    };

    ///
    /// \brief
    ///
    /*class NANO_PUBLIC memory_flatten_dataset_iterator_t final : public flatten_dataset_iterator_t
    {
    public:

        memory_flatten_dataset_iterator_t(const memory_dataset_t&, indices_t samples);

        const indices_t& samples() const override;
        bool cache_inputs(int64_t bytes, execution) override;
        bool cache_targets(int64_t bytes, execution) override;

        feature_t target() const override;
        tensor3d_dims_t target_dims() const override;
        tensor4d_cmap_t targets(tensor_range_t samples, tensor4d_t& buffer) const override;

        tensor1d_dims_t inputs_dims() const override;
        tensor2d_cmap_t inputs(tensor_range_t samples, tensor2d_t& buffer) const override;

    private:

        // attributes
        const memory_dataset_t& m_dataset;  ///<
        indices_t               m_samples;  ///<
    };*/
}
