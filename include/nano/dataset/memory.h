#pragma once

#include <variant>
#include <nano/util.h>
#include <nano/logger.h>
#include <nano/dataset/dataset.h>

namespace nano
{
    // continuous feature: (sample, dim1, dim2, dim3)
    template <typename tscalar>
    using scalar_storage_t = tensor_mem_t<tscalar, 4>;

    // single-label discrete feature: (sample) = label index
    template <typename tscalar>
    using sclass_storage_t = tensor_mem_t<tscalar, 1>;

    // multi-label discrete feature: (sample, label_index) = 1 if the label index is active, otherwise 0
    template <typename tscalar>
    using mclass_storage_t = tensor_mem_t<tscalar, 2>;

    // bitwise mask for a feature: (sample) = 1 if the feature value is available, otherwise 0
    using mask_t = tensor_mem_t<uint8_t, 1>;
    using mask_map_t = tensor_map_t<uint8_t, 1>;
    using mask_cmap_t = tensor_cmap_t<uint8_t, 1>;

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
    inline void setbit(const mask_map_t& mask, tensor_size_t sample)
    {
        assert(sample >= 0 && sample < (8 * mask.size()));
        mask(sample / 8) |= static_cast<uint8_t>(0x01 << (7 - (sample % 8)));
    }

    ///
    /// \brief check if a feature value exists for a particular sample.
    ///
    inline bool getbit(const mask_cmap_t& mask, tensor_size_t sample)
    {
        assert(sample >= 0 && sample < (8 * mask.size()));
        return (mask(sample / 8) & (0x01 << (7 - (sample % 8)))) != 0x00;
    }

    ///
    /// \brief returns true if the feature is optional (aka some samples haven't been set).
    ///
    inline bool optional(const mask_cmap_t& mask, tensor_size_t samples)
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
    /// \brief per-feature statistics for continuous feature values
    ///     (e.g. useful for normalizing inputs).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct feature_scalar_stats_t
    {
        tensor_size_t   m_count{0};         ///< total number of samples
        tensor3d_t      m_min, m_max;       ///<
        tensor3d_t      m_mean, m_stdev;    ///<
    };

    ///
    /// \brief per-feature statistics for single-label and multi-label discrete feature values
    ///     (e.g. useful for fixing unbalanced classification problems).
    ///
    /// NB: missing feature values are ignored when computing these statistics.
    ///
    struct feature_sclass_stats_t
    {
        indices_t       m_class_counts{0};  ///< number of samples per class (label)
    };
    struct feature_mclass_stats_t
    {
        indices_t       m_class_counts{0};  ///< number of samples per class (label)
    };

    ///
    /// \brief store the feature values of a collection of samples as compact as possible.
    ///
    class NANO_PUBLIC feature_storage_t
    {
    public:

        using storage_t = std::variant
        <
            scalar_storage_t<float>,
            scalar_storage_t<double>,
            scalar_storage_t<int8_t>,
            scalar_storage_t<int16_t>,
            scalar_storage_t<int32_t>,
            scalar_storage_t<int64_t>,
            scalar_storage_t<uint8_t>,
            scalar_storage_t<uint16_t>,
            scalar_storage_t<uint32_t>,
            scalar_storage_t<uint64_t>,

            sclass_storage_t<uint8_t>,
            sclass_storage_t<uint16_t>,

            mclass_storage_t<uint8_t>
        >;

        feature_storage_t() = default;
        feature_storage_t(feature_t, tensor_size_t samples);

        ///
        /// \brief access functions
        ///
        const auto& dims() const { return m_feature.dims(); }
        const auto& name() const { return m_feature.name(); }
        const feature_t& feature() const { return m_feature; }
        const storage_t& storage() const { return m_storage; }
        tensor_size_t labels() const { return static_cast<tensor_size_t>(m_feature.labels().size()); }

        ///
        /// \brief mutable access to the internal storage.
        ///
        template <typename toperator>
        auto visit(const toperator& op)
        {
            return std::visit(overloaded{
                [&] (scalar_storage_t<float>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<double>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<int8_t>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<int16_t>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<int32_t>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<int64_t>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<uint8_t>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<uint16_t>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<uint32_t>& tensor) { return op(tensor.tensor()); },
                [&] (scalar_storage_t<uint64_t>& tensor) { return op(tensor.tensor()); },
                [&] (sclass_storage_t<uint8_t>& tensor) { return op(tensor.tensor()); },
                [&] (sclass_storage_t<uint16_t>& tensor) { return op(tensor.tensor()); },
                [&] (mclass_storage_t<uint8_t>& tensor) { return op(tensor.tensor()); },
            }, m_storage);
        }

        ///
        /// \brief constant access to the internal storage.
        ///
        template <typename toperator>
        auto visit(const toperator& op) const
        {
            return std::visit(overloaded{
                [&] (const scalar_storage_t<float>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<double>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<int8_t>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<int16_t>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<int32_t>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<int64_t>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<uint8_t>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<uint16_t>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<uint32_t>& tensor) { return op(tensor); },
                [&] (const scalar_storage_t<uint64_t>& tensor) { return op(tensor); },
                [&] (const sclass_storage_t<uint8_t>& tensor) { return op(tensor); },
                [&] (const sclass_storage_t<uint16_t>& tensor) { return op(tensor); },
                [&] (const mclass_storage_t<uint8_t>& tensor) { return op(tensor); },
            }, m_storage);
        }

        ///
        /// \brief returns the number of stored samples.
        ///
        tensor_size_t samples() const
        {
            return visit([] (const auto& tensor) { return tensor.template size<0>(); });
        }

        ///
        /// \brief returns the feature-wise statistics of the given samples.
        ///
        feature_scalar_stats_t scalar_stats(const indices_cmap_t& samples, const mask_cmap_t& mask) const;
        feature_sclass_stats_t sclass_stats(const indices_cmap_t& samples, const mask_cmap_t& mask) const;
        feature_mclass_stats_t mclass_stats(const indices_cmap_t& samples, const mask_cmap_t& mask) const;

        ///
        /// \brief set the feature value of a sample.
        ///
        template <typename tvalue>
        void set(tensor_size_t sample, const tvalue& value)
        {
            visit([&] (const auto& tensor)
            {
                const auto samples = tensor.template size<0>();
                critical(
                    sample < 0 || sample >= samples,
                    "cannot set feature <", name(),
                    ">: invalid sample ", sample, " not in [0, ", samples, ")!");

                this->set(tensor, sample, value);
            });
        }

        ///
        /// \brief get the feature values of the given samples.
        ///
        template <typename tvalue, size_t trank>
        void get(const indices_cmap_t& samples, const mask_cmap_t& mask, tensor_mem_t<tvalue, trank>& values) const
        {
            visit([&] (const auto& tensor)
            {
                const auto _samples = tensor.template size<0>();

                critical(
                    samples.min() < 0 || samples.max() >= _samples,
                    "cannot access feature <", name(),
                    "> invalid samples [", samples.min(), ", ", samples.max(), ") not in [0, ", _samples, ")!");

                this->get(tensor, samples, mask, values);
            });
        }

    private:

        template <typename tscalar>
        auto check_from_string(const char* type, const string_t& value) const
        {
            tscalar scalar;
            try
            {
                scalar = ::nano::from_string<tscalar>(value);
            }
            catch (std::exception& e)
            {
                critical0("cannot set ", type, " feature <", name(), ">: caught exception <", e.what(), ">!");
            }
            return scalar;
        }

        template <typename tscalar, typename tvalue>
        void set(const tensor_map_t<tscalar, 1>& tensor, tensor_size_t sample, const tvalue& value)
        {
            tensor_size_t label;
            if constexpr (std::is_same<tvalue, string_t>::value)
            {
                label = check_from_string<tensor_size_t>("single-label", value);
            }
            else if constexpr (std::is_arithmetic<tvalue>::value)
            {
                label = static_cast<tensor_size_t>(value);
            }
            else
            {
                critical0("cannot set single-label feature <", name(), ">!");
            }

            critical(
                label < 0 || label >= labels(),
                "cannot set single-label feature <", name(),
                ">: invalid label ", label, " not in [0, ", labels(), ")!");

            tensor(sample) = static_cast<tscalar>(label);
        }

        template <typename tscalar, typename tvalue>
        void set(const tensor_map_t<tscalar, 2>& tensor, tensor_size_t sample, const tvalue& value)
        {
            if constexpr (::nano::is_tensor<tvalue>::value)
            {
                if constexpr (tvalue::rank() == 1)
                {
                    critical(
                        value.size() != labels(),
                        "cannot set multi-label feature <", name(),
                        ">: invalid number of labels ", value.size(), " vs. ", labels(), "!");

                    tensor.vector(sample) = value.vector().template cast<tscalar>();
                }
                else
                {
                    critical0("cannot set multi-label feature <", name(), ">!");
                }
            }
            else
            {
                critical0("cannot set multi-label feature <", name(), ">!");
            }
        }

        template <typename tscalar, typename tvalue>
        void set(const tensor_map_t<tscalar, 4>& tensor, tensor_size_t sample, const tvalue& value)
        {
            if constexpr (std::is_same<tvalue, string_t>::value)
            {
                critical(
                    ::nano::size(dims()) != 1,
                    "cannot set scalar feature <", name(),
                    ">: invalid tensor dimensions ", dims(), "!");

                tensor(sample) = check_from_string<tscalar>("scalar", value);
            }
            else if constexpr (std::is_arithmetic<tvalue>::value)
            {
                critical(
                    ::nano::size(dims()) != 1,
                    "cannot set scalar feature <", name(),
                    ">: invalid tensor dimensions ", dims(), "!");

                tensor(sample) = static_cast<tscalar>(value);
            }
            else if constexpr (::nano::is_tensor<tvalue>())
            {
                critical(
                    ::nano::size(dims()) != value.size(),
                    "cannot set scalar feature <", name(),
                    ">: invalid tensor dimensions ", dims(), " vs. ", value.dims(), "!");

                tensor.vector(sample) = value.vector().template cast<tscalar>();
            }
            else
            {
                critical0("cannot set scalar feature <", name(), ">!");
            }
        }

        template <typename tscalar, typename tvalue, size_t trank>
        void get(const tensor_mem_t<tscalar, 1>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask,
            tensor_mem_t<tvalue, trank>& values) const
        {
            if constexpr (trank == 1)
            {
                values.resize(samples.size());
                values.constant(-1);

                for (tensor_size_t i = 0; i < samples.size(); ++ i)
                {
                    if (::nano::getbit(mask, samples(i)))
                    {
                        values(i) = static_cast<tvalue>(tensor(samples(i)));
                    }
                }
            }
            else
            {
                critical0("cannot access single-label feature <", name(), ">!");
            }
        }

        template <typename tscalar, typename tvalue, size_t trank>
        void get(const tensor_mem_t<tscalar, 2>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask,
            tensor_mem_t<tvalue, trank>& values) const
        {
            if constexpr (trank == 2)
            {
                values.resize(make_dims(samples.size(), labels()));
                values.constant(-1);

                for (tensor_size_t i = 0; i < samples.size(); ++ i)
                {
                    if (::nano::getbit(mask, samples(i)))
                    {
                        values.vector(i) = tensor.vector(samples(i)).template cast<tvalue>();
                    }
                }
            }
            else
            {
                critical0("cannot access multi-label feature <", name(), ">!");
            }
        }

        template <typename tscalar, typename tvalue, size_t trank>
        void get(const tensor_mem_t<tscalar, 4>& tensor, const indices_cmap_t& samples, const mask_cmap_t& mask,
            tensor_mem_t<tvalue, trank>& values) const
        {
            if constexpr (trank == 4)
            {
                values.resize(cat_dims(samples.size(), dims()));
                values.constant(std::numeric_limits<tvalue>::quiet_NaN());

                for (tensor_size_t i = 0; i < samples.size(); ++ i)
                {
                    if (::nano::getbit(mask, samples(i)))
                    {
                        values.vector(i) = tensor.vector(samples(i)).template cast<tvalue>();
                    }
                }
            }
            else if constexpr (trank == 1)
            {
                critical(
                    ::nano::size(dims()) != 1,
                    "cannot access scalar feature <", name(), ">!");

                values.resize(samples.size());
                values.constant(std::numeric_limits<tvalue>::quiet_NaN());

                for (tensor_size_t i = 0; i < samples.size(); ++ i)
                {
                    if (::nano::getbit(mask, samples(i)))
                    {
                        values(i) = static_cast<tvalue>(tensor(i));
                    }
                }
            }
            else
            {
                critical0("cannot access scalar feature <", name(), ">!");
            }
        }

        // attributes
        feature_t       m_feature;  ///<
        storage_t       m_storage;  ///<
    };

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

        size_t target() const
        {
            return m_target;
        }

        task_type type() const override
        {
            return  has_target() ?
                    static_cast<task_type>(tstorage().feature()) :
                    task_type::unsupervised;
        }

        tensor_size_t samples() const
        {
            return  m_storage.empty() ?
                    tensor_size_t(0) :
                    m_storage.begin()->samples();
        }

        bool has_target() const
        {
            return m_target < m_storage.size();
        }

        auto features() const
        {
            const auto total = static_cast<tensor_size_t>(m_storage.size());
            return has_target() ? (total - 1) : total;
        }

        auto tmask() const
        {
            assert(has_target());
            return m_mask.tensor(static_cast<tensor_size_t>(m_target));
        }

        auto imask(tensor_size_t feature) const
        {
            const auto ufeature = static_cast<size_t>(feature);
            return m_mask.tensor(static_cast<tensor_size_t>(ufeature >= m_target ? (ufeature + 1) : ufeature));
        }

        const feature_storage_t& tstorage() const
        {
            assert(has_target());
            return m_storage[m_target];
        }

        const feature_storage_t& istorage(tensor_size_t feature) const
        {
            const auto ufeature = static_cast<size_t>(feature);
            return m_storage[ufeature >= m_target ? (ufeature + 1) : ufeature];
        }

        mask_cmap_t mask(tensor_size_t feature) const
        {
            return m_mask.tensor(feature);
        }

        const auto& storage() const
        {
            return m_storage;
        }

        const feature_storage_t& storage(tensor_size_t feature) const
        {
            return m_storage[static_cast<size_t>(feature)];
        }

        rfeature_dataset_iterator_t feature_iterator(indices_t samples) const override;
        rflatten_dataset_iterator_t flatten_iterator(indices_t samples) const override;

    protected:

        void resize(tensor_size_t samples, const features_t& features);
        void resize(tensor_size_t samples, const features_t& features, size_t target);

        mask_map_t mask(tensor_size_t feature)
        {
            return m_mask.tensor(feature);
        }

        feature_storage_t& storage(tensor_size_t feature)
        {
            return m_storage[static_cast<size_t>(feature)];
        }

        template <typename tvalue>
        void set(tensor_size_t sample, tensor_size_t feature, const tvalue& value)
        {
            this->set(storage(feature), mask(feature), sample, value);
        }

    private:

        template <typename tvalue>
        static void set(feature_storage_t& fs, tensor_map_t<uint8_t, 1>&& mask, tensor_size_t sample, const tvalue& value)
        {
            fs.set(sample, value);
            ::nano::setbit(mask, sample);
        }

        using mask_t = tensor_mem_t<uint8_t, 2>;
        using storage_t = std::vector<feature_storage_t>;

        // attributes
        mask_t                  m_mask;         ///< feature value given if the bit (feature, sample) is 1
        storage_t               m_storage;      ///<
        size_t                  m_target{0};    ///<
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
        tensor4d_cmap_t targets(tensor4d_t& buffer) const override;

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
