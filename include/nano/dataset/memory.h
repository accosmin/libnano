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

        feature_storage_t& istorage(tensor_size_t feature);
        feature_storage_t& tstorage() { return m_target; }

        template <typename tvalue>
        void set(tensor_size_t sample, const tvalue& value)
        {
            this->set(m_target, m_target_mask, sample, value);
        }

        template <typename tvalue>
        void set(tensor_size_t sample, tensor_size_t feature, const tvalue& value)
        {
            this->set(istorage(feature), m_inputs_mask.tensor(feature), sample, value);
        }

    private:

        using inputs_t = std::vector<feature_storage_t>;
        using target_t = feature_storage_t;

        using inputs_mask_t = tensor_mem_t<uint8_t, 2>;
        using target_mask_t = tensor_mem_t<uint8_t, 1>;

        template <typename tscalar>
        static auto check_from_string(const char* type, const feature_t& feature, const string_t& value)
        {
            tscalar scalar;
            try
            {
                scalar = ::nano::from_string<tscalar>(value);
            }
            catch (std::exception& e)
            {
                critical0("cannot set ", type, " feature <", feature.name(), ">: caught exception <", e.what(), ">!");
            }
            return scalar;
        }

        template <typename tscalar, size_t trank, typename tvalue>
        static void set(
            const feature_t& feature, const tensor_map_t<tscalar, trank>& tensor, tensor_size_t sample, const tvalue& value)
        {
            // single-label feature
            if constexpr (trank == 1)
            {
                const auto samples = tensor.template size<0>();
                critical(
                    sample < 0 || sample >= samples,
                    "cannot set single-label feature <", feature.name(),
                    ">: invalid sample ", sample, " not in [0, ", samples, ")!");

                tensor_size_t label;
                if constexpr (std::is_same<tvalue, string_t>::value)
                {
                    label = memory_dataset_t::check_from_string<tensor_size_t>("single-label", feature, value);
                }
                else if constexpr (std::is_arithmetic<tvalue>::value)
                {
                    label = static_cast<tensor_size_t>(value);
                }
                else
                {
                    critical0("cannot set single-label feature <", feature.name(), ">!");
                }

                const auto labels = static_cast<tensor_size_t>(feature.labels().size());
                critical(
                    label < 0 || label >= labels,
                    "cannot set single-label feature <", feature.name(),
                    ">: invalid label ", label, " not in [0, ", labels, ")!");

                tensor(sample) = static_cast<tscalar>(value);
            }

            // multi-label feature
            else if constexpr (trank == 2)
            {
                const auto samples = tensor.template size<0>();
                critical(
                    sample < 0 || sample >= samples,
                    "cannot set multi-label feature <", feature.name(),
                    ">: invalid sample ", sample, " not in [0, ", samples, ")!");

                if constexpr (::nano::is_tensor<tvalue>::value)
                {
                    if constexpr (tvalue::rank() == 1)
                    {
                        const auto labels = static_cast<tensor_size_t>(feature.labels().size());

                        critical(
                            value.size() != labels,
                            "cannot set multi-label feature <", feature.name(),
                            ">: invalid number of labels ", value.size(), " vs. ", labels, "!");

                        tensor.vector(sample) = value.vector().template cast<tscalar>();
                    }
                    else
                    {
                        critical0("cannot set multi-label feature <", feature.name(), ">!");
                    }
                }
                else
                {
                    critical0("cannot set multi-label feature <", feature.name(), ">!");
                }
            }

            // continuous feature
            else
            {
                const auto samples = tensor.template size<0>();
                critical(
                    sample < 0 || sample >= samples,
                    "cannot set scalar feature <", feature.name(),
                    ">: invalid sample ", sample, " not in [0, ", samples, ")!");

                if constexpr (std::is_same<tvalue, string_t>::value)
                {
                    critical(
                        ::nano::size(feature.dims()) != 1,
                        "cannot set scalar feature <", feature.name(),
                        ">: invalid tensor dimensions ", feature.dims(), "!");

                    tensor(sample) = memory_dataset_t::check_from_string<tscalar>("scalar", feature, value);
                }
                else if constexpr (std::is_arithmetic<tvalue>::value)
                {
                    critical(
                        ::nano::size(feature.dims()) != 1,
                        "cannot set scalar feature <", feature.name(),
                        ">: invalid tensor dimensions ", feature.dims(), "!");

                    tensor(sample) = static_cast<tscalar>(value);
                }
                else if constexpr (tvalue::rank() == 1)
                {
                    critical(
                        ::nano::size(feature.dims()) != value.size(),
                        "cannot set scalar feature <", feature.name(),
                        ">: invalid tensor dimensions ", feature.dims(), " vs. ", value.dims(), "!");

                    tensor.vector(sample) = value.vector().template cast<tscalar>();
                }
                else if constexpr (tvalue::rank() == 3)
                {
                    critical(
                        feature.dims() != value.dims(),
                        "cannot set scalar feature <", feature.name(),
                        ">: invalid tensor dimensions ", feature.dims(), " vs. ", value.dims(), "!");

                    tensor.vector(sample) = value.vector().template cast<tscalar>();
                }
                else
                {
                    critical0("cannot set scalar feature <", feature.name(), ">!");
                }
            }
        }

        template <typename tvalue>
        void set(feature_storage_t& fs, tensor_map_t<uint8_t, 1>&& mask, tensor_size_t sample, const tvalue& value)
        {
            fs.visit([&] (const auto& tensor)
            {
                memory_dataset_t::set(fs.feature(), tensor, sample, value);
            });

            ::nano::setbit(mask, sample);
        }

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
