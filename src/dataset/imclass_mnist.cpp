#include <fstream>
#include <nano/core/logger.h>
#include <nano/dataset/imclass_mnist.h>

using namespace nano;

base_mnist_dataset_t::base_mnist_dataset_t(string_t dir, string_t name, feature_t target) :
    m_dir(std::move(dir)),
    m_name(std::move(name)),
    m_target(std::move(target))
{
}

void base_mnist_dataset_t::load()
{
    const auto parts =
    {
        std::make_tuple(
            m_dir + "/train-images-idx3-ubyte",
            m_dir + "/train-labels-idx1-ubyte",
            tensor_size_t(0), tensor_size_t(60000)),
        std::make_tuple(
            m_dir + "/t10k-images-idx3-ubyte",
            m_dir + "/t10k-labels-idx1-ubyte",
            tensor_size_t(60000), tensor_size_t(10000))
    };

    const auto features = std::vector<feature_t>
    {
        feature_t("image").scalar(feature_type::uint8, make_dims(1, 28, 28)),
        m_target
    };
    resize(70000, features, 1U);

    tensor_size_t sample = 0;
    for (const auto& part : parts)
    {
        const auto& ifile = std::get<0>(part);
        const auto& tfile = std::get<1>(part);
        const auto offset = std::get<2>(part);
        const auto expected = std::get<3>(part);

        log_info() << m_name << ": loading file <" << ifile << ">...";
        critical(
            !iread(ifile, offset, expected),
            m_name, ": failed to load file <", ifile, ">!");

        log_info() << m_name << ": loading file <" << tfile << ">...";
        critical(
            !tread(tfile, offset, expected),
            m_name, ": failed to load file <", tfile, ">!");

        sample += expected;
        log_info() << m_name << ": loaded " << sample << " samples.";
    }

    dataset_t::testing({make_range(60000, 70000)});
}

bool base_mnist_dataset_t::iread(const string_t& path, tensor_size_t sample, tensor_size_t expected)
{
    char buffer[16];
    tensor_mem_t<uint8_t, 3> image(1, 28, 28);

    std::ifstream stream(path);
    if (!stream.read(buffer, 16))
    {
        return false;
    }

    for ( ; expected > 0; -- expected, ++ sample)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        if (!stream.read(reinterpret_cast<char*>(image.data()), image.size())) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        {
            return false;
        }
        set(sample, 0, image);
    }

    return expected == 0;
}

bool base_mnist_dataset_t::tread(const string_t& path, tensor_size_t sample, tensor_size_t expected)
{
    std::ifstream stream(path);

    char buffer[8];
    if (!stream.read(buffer, 8))
    {
        return false;
    }

    tensor_vector_t<char> labels(expected);
    if (!stream.read(labels.data(), labels.size()))
    {
        return false;
    }

    for (auto i = 0; expected > 0; -- expected, ++ sample, ++ i)
    {
        set(sample, 1, static_cast<tensor_size_t>(static_cast<unsigned char>(labels(i))));
    }

    return true;
}

mnist_dataset_t::mnist_dataset_t() : base_mnist_dataset_t(
    scat(std::getenv("HOME"), "/libnano/datasets/mnist"), "MNIST",
    feature_t("digit").sclass(strings_t
    {"digit0", "digit1", "digit2", "digit3", "digit4", "digit5", "digit6", "digit7", "digit8", "digit9"}))
{
}

fashion_mnist_dataset_t::fashion_mnist_dataset_t() : base_mnist_dataset_t(
    scat(std::getenv("HOME"), "/libnano/datasets/fashion-mnist"), "Fashion-MNIST",
    feature_t("article").sclass(strings_t
    {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"}))
{
}
