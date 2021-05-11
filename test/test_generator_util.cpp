#include <utest/utest.h>
#include <nano/generator/util.h>

using namespace nano;

static auto make_features()
{
    return features_t
    {
        feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}),
        feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(2, 1, 2)),
        feature_t{"f64"}.scalar(feature_type::float64),
    };
}

class fixture_dataset_t final : public memory_dataset_t
{
public:

    fixture_dataset_t(tensor_size_t samples, size_t target) :
        m_samples(samples),
        m_features(make_features()),
        m_target(target)
    {
    }

    void load() override
    {
        resize(m_samples, m_features, m_target);

        tensor_mem_t<tensor_size_t, 1> hits(3);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 3)
        {
            hits(0) = sample % 2;
            hits(1) = 1 - (sample % 2);
            hits(2) = (sample % 6) == 0;
            set(sample, 0, hits);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 1, (sample % 3 == 0) ? 0 : 1);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 2, sample);
        }

        tensor_mem_t<tensor_size_t, 3> values(2, 1, 2);
        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            values.full(sample);
            values(0) = sample + 1;
            set(sample, 3, values);
        }

        for (tensor_size_t sample = 0; sample < m_samples; sample ++)
        {
            set(sample, 4, 1 - sample);
        }
    }

private:

    tensor_size_t   m_samples{0};
    features_t      m_features;
    size_t          m_target;
};

static auto make_dataset(tensor_size_t samples, size_t target)
{
    auto dataset = fixture_dataset_t{samples, target};
    UTEST_CHECK_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    return dataset;
}

UTEST_BEGIN_MODULE(test_generator_util)

UTEST_CASE(select_scalar_components)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, indices_t{});
        const auto expected = make_tensor<tensor_size_t>(make_dims(2, 2), 2, 0, 4, 0);
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(2));
        const auto expected = make_tensor<tensor_size_t>(make_dims(1, 2), 2, 0);
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(3));
        const auto expected = feature_mapping_t{make_dims(0, 2)};
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::off, make_indices(2, 3, 4));
        const auto expected = make_tensor<tensor_size_t>(make_dims(2, 2), 2, 0, 4, 0);
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, indices_t{});
        const auto expected = make_tensor<tensor_size_t>(make_dims(6, 2), 2, 0, 3, 0, 3, 1, 3, 2, 3, 3, 4, 0);
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, make_indices(1, 4));
        const auto expected = make_tensor<tensor_size_t>(make_dims(1, 2), 4, 0);
        UTEST_CHECK_EQUAL(indices, expected);
    }
    {
        const auto indices = select_scalar_components(dataset, struct2scalar::on, make_indices(1, 3,  4));
        const auto expected = make_tensor<tensor_size_t>(make_dims(5, 2), 3, 0, 3, 1, 3, 2, 3, 3, 4, 0);
        UTEST_CHECK_EQUAL(indices, expected);
    }
}

UTEST_CASE(for_each_scalar)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        std::vector<std::tuple<feature_t, tensor_size_t, tensor_size_t>> history;
        for_each_scalar(dataset, struct2scalar::off, [&] (const feature_t& feature, tensor_size_t ifeature, tensor_size_t icomponent)
        {
            history.emplace_back(feature, ifeature, icomponent);
        });

        UTEST_REQUIRE_EQUAL(history.size(), 2U);
        UTEST_CHECK_EQUAL(std::get<0>(history[0]), dataset.feature(2U));
        UTEST_CHECK_EQUAL(std::get<1>(history[0]), 2);
        UTEST_CHECK_EQUAL(std::get<2>(history[0]), -1);
        UTEST_CHECK_EQUAL(std::get<0>(history[1]), dataset.feature(4U));
        UTEST_CHECK_EQUAL(std::get<1>(history[1]), 4);
        UTEST_CHECK_EQUAL(std::get<2>(history[1]), -1);
    }
    {
        std::vector<std::tuple<feature_t, tensor_size_t, tensor_size_t>> history;
        for_each_scalar(dataset, struct2scalar::on, [&] (const feature_t& feature, tensor_size_t ifeature, tensor_size_t icomponent)
        {
            history.emplace_back(feature, ifeature, icomponent);
        });

        UTEST_REQUIRE_EQUAL(history.size(), 6U);
        UTEST_CHECK_EQUAL(std::get<0>(history[0]), dataset.feature(2U));
        UTEST_CHECK_EQUAL(std::get<1>(history[0]), 2);
        UTEST_CHECK_EQUAL(std::get<2>(history[0]), -1);
        UTEST_CHECK_EQUAL(std::get<0>(history[1]), dataset.feature(3U));
        UTEST_CHECK_EQUAL(std::get<1>(history[1]), 3);
        UTEST_CHECK_EQUAL(std::get<2>(history[1]), 0);
        UTEST_CHECK_EQUAL(std::get<0>(history[2]), dataset.feature(3U));
        UTEST_CHECK_EQUAL(std::get<1>(history[2]), 3);
        UTEST_CHECK_EQUAL(std::get<2>(history[2]), 1);
        UTEST_CHECK_EQUAL(std::get<0>(history[3]), dataset.feature(3U));
        UTEST_CHECK_EQUAL(std::get<1>(history[3]), 3);
        UTEST_CHECK_EQUAL(std::get<2>(history[3]), 2);
        UTEST_CHECK_EQUAL(std::get<0>(history[4]), dataset.feature(3U));
        UTEST_CHECK_EQUAL(std::get<1>(history[4]), 3);
        UTEST_CHECK_EQUAL(std::get<2>(history[4]), 3);
        UTEST_CHECK_EQUAL(std::get<0>(history[5]), dataset.feature(4U));
        UTEST_CHECK_EQUAL(std::get<1>(history[5]), 4);
        UTEST_CHECK_EQUAL(std::get<2>(history[5]), -1);
    }
}

UTEST_CASE(for_each_struct)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        std::vector<std::tuple<feature_t, tensor_size_t, tensor_size_t>> history;
        for_each_struct(dataset, [&] (const feature_t& feature, tensor_size_t ifeature, tensor_size_t icomponent)
        {
            history.emplace_back(feature, ifeature, icomponent);
        });

        UTEST_REQUIRE_EQUAL(history.size(), 1U);
        UTEST_CHECK_EQUAL(std::get<0>(history[0]), dataset.feature(3U));
        UTEST_CHECK_EQUAL(std::get<1>(history[0]), 3);
        UTEST_CHECK_EQUAL(std::get<2>(history[0]), -1);
    }
}

UTEST_CASE(for_each_sclass)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        std::vector<std::tuple<feature_t, tensor_size_t, tensor_size_t>> history;
        for_each_sclass(dataset, sclass2binary::off, [&] (const feature_t& feature, tensor_size_t ifeature, tensor_size_t icomponent)
        {
            history.emplace_back(feature, ifeature, icomponent);
        });

        UTEST_REQUIRE_EQUAL(history.size(), 1U);
        UTEST_CHECK_EQUAL(std::get<0>(history[0]), dataset.feature(1U));
        UTEST_CHECK_EQUAL(std::get<1>(history[0]), 1);
        UTEST_CHECK_EQUAL(std::get<2>(history[0]), -1);
    }
    {
        std::vector<std::tuple<feature_t, tensor_size_t, tensor_size_t>> history;
        for_each_sclass(dataset, sclass2binary::on, [&] (const feature_t& feature, tensor_size_t ifeature, tensor_size_t icomponent)
        {
            history.emplace_back(feature, ifeature, icomponent);
        });

        UTEST_REQUIRE_EQUAL(history.size(), 2U);
        UTEST_CHECK_EQUAL(std::get<0>(history[0]), dataset.feature(1U));
        UTEST_CHECK_EQUAL(std::get<1>(history[0]), 1);
        UTEST_CHECK_EQUAL(std::get<2>(history[0]), 0);
        UTEST_CHECK_EQUAL(std::get<0>(history[1]), dataset.feature(1U));
        UTEST_CHECK_EQUAL(std::get<1>(history[1]), 1);
        UTEST_CHECK_EQUAL(std::get<2>(history[1]), 1);
    }
}

UTEST_CASE(for_each_mclass)
{
    const auto dataset = make_dataset(10, string_t::npos);
    {
        std::vector<std::tuple<feature_t, tensor_size_t, tensor_size_t>> history;
        for_each_mclass(dataset, mclass2binary::off, [&] (const feature_t& feature, tensor_size_t ifeature, tensor_size_t icomponent)
        {
            history.emplace_back(feature, ifeature, icomponent);
        });

        UTEST_REQUIRE_EQUAL(history.size(), 1U);
        UTEST_CHECK_EQUAL(std::get<0>(history[0]), dataset.feature(0U));
        UTEST_CHECK_EQUAL(std::get<1>(history[0]), 0);
        UTEST_CHECK_EQUAL(std::get<2>(history[0]), -1);
    }
    {
        std::vector<std::tuple<feature_t, tensor_size_t, tensor_size_t>> history;
        for_each_mclass(dataset, mclass2binary::on, [&] (const feature_t& feature, tensor_size_t ifeature, tensor_size_t icomponent)
        {
            history.emplace_back(feature, ifeature, icomponent);
        });

        UTEST_REQUIRE_EQUAL(history.size(), 3U);
        UTEST_CHECK_EQUAL(std::get<0>(history[0]), dataset.feature(0U));
        UTEST_CHECK_EQUAL(std::get<1>(history[0]), 0);
        UTEST_CHECK_EQUAL(std::get<2>(history[0]), 0);
        UTEST_CHECK_EQUAL(std::get<0>(history[1]), dataset.feature(0U));
        UTEST_CHECK_EQUAL(std::get<1>(history[1]), 0);
        UTEST_CHECK_EQUAL(std::get<2>(history[1]), 1);
        UTEST_CHECK_EQUAL(std::get<0>(history[2]), dataset.feature(0U));
        UTEST_CHECK_EQUAL(std::get<1>(history[2]), 0);
        UTEST_CHECK_EQUAL(std::get<2>(history[2]), 2);
    }
}


UTEST_END_MODULE()
