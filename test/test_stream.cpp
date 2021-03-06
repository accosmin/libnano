#include <mutex>
#include <fstream>
#include <sstream>
#include <utest/utest.h>
#include <nano/stream.h>
#include <nano/factory.h>

using namespace nano;

struct object_t;
using object_factory_t = factory_t<object_t>;
using robject_t = object_factory_t::trobject;

struct object_t : public serializable_t
{
    object_t() = default;
    ~object_t() override = default;
    static object_factory_t& all();
    object_t(object_t&&) = default;
    object_t(const object_t&) = default;
    object_t& operator=(object_t&&) = default;
    object_t& operator=(const object_t&) = default;
    virtual int get() const = 0;
    virtual robject_t clone() const = 0;
};

template <int tv>
struct objectx_t final : public object_t
{
    objectx_t() = default;
    int get() const override { return tv; }
    robject_t clone() const override { return std::make_unique<objectx_t<tv>>(*this); }
};

using object1_t = objectx_t<1>;
using object2_t = objectx_t<2>;
using object3_t = objectx_t<3>;

object_factory_t& object_t::all()
{
    static object_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<object1_t>("id1", "desc1");
        manager.add<object2_t>("id2", "desc2");
        manager.add<object3_t>("id3", "desc3");
    });

    return manager;
}

auto to_string(const serializable_t& object)
{
    std::ostringstream stream;
    UTEST_REQUIRE_NOTHROW(object.write(stream));
    UTEST_REQUIRE(stream);
    return stream.str();
}

UTEST_BEGIN_MODULE(test_stream)

UTEST_CASE(string)
{
    for (const auto& string : {std::string{}, std::string("stream strings")})
    {
        std::ostringstream ostream;
        UTEST_REQUIRE_NOTHROW(::nano::write(ostream, string));
        UTEST_REQUIRE(ostream);

        const auto ostring = ostream.str();
        UTEST_CHECK_EQUAL(ostring.size(), string.size() + 4);

        std::string istring;
        std::istringstream istream(ostring);
        UTEST_REQUIRE(istream);
        UTEST_REQUIRE_NOTHROW(::nano::read(istream, istring));
        UTEST_REQUIRE(istream);

        UTEST_CHECK_EQUAL(string, istring);

        std::string ifstring;
        std::ifstream ifstream;
        UTEST_REQUIRE(ifstream);
        UTEST_REQUIRE_NOTHROW(::nano::read(ifstream, ifstring));
        UTEST_REQUIRE(!ifstream);
    }
}

UTEST_CASE(serializable_default)
{
    const auto object = serializable_t{};
    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version);
}

UTEST_CASE(serializable_read_const)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4));

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(object.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version);
}

UTEST_CASE(serializable_read_major)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[0] = ::nano::major_version - 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(object.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version - 1);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version - 0);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version - 0);
}

UTEST_CASE(serializable_read_minor)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[1] = ::nano::minor_version - 2; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(object.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version - 0);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version - 2);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version - 0);
}

UTEST_CASE(serializable_read_patch)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = ::nano::patch_version - 3; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(object.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version - 0);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version - 0);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version - 3);
}

UTEST_CASE(serializable_write_fail)
{
    const auto object = serializable_t{};

    std::ofstream stream;
    UTEST_CHECK_THROW(object.write(stream), std::runtime_error);
}

UTEST_CASE(serializable_read_fail_major)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[0] = ::nano::major_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(object.read(stream), std::runtime_error);
}

UTEST_CASE(serializable_read_fail_minor)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[1] = ::nano::minor_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(object.read(stream), std::runtime_error);
}

UTEST_CASE(serializable_read_fail_patch)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = ::nano::patch_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(object.read(stream), std::runtime_error);
}

UTEST_CASE(identifiable_default)
{
    auto object = identifiable_t<object_t>{};

    std::ostringstream stream;
    UTEST_CHECK_EQUAL(object.id(), "");
    UTEST_CHECK_EQUAL(static_cast<bool>(object), false);
    UTEST_CHECK_THROW(object.write(stream), std::runtime_error);
}

UTEST_CASE(identifiable_read_write)
{
    auto object = identifiable_t<object_t>{"id2", std::make_unique<object2_t>()};

    std::string str;
    {
        std::ostringstream stream;
        UTEST_CHECK_EQUAL(object.id(), "id2");
        UTEST_CHECK_EQUAL(static_cast<bool>(object), true);
        UTEST_CHECK_EQUAL(object.get().get(), 2);
        UTEST_CHECK_NOTHROW(object.write(stream));
        str = stream.str();
    }
    {
        object = identifiable_t<object_t>{};
        UTEST_CHECK_EQUAL(object.id(), "");
        UTEST_CHECK_EQUAL(static_cast<bool>(object), false);

        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(object.read(stream));
        UTEST_CHECK_EQUAL(object.id(), "id2");
        UTEST_CHECK_EQUAL(static_cast<bool>(object), true);
        UTEST_CHECK_EQUAL(object.get().get(), 2);
    }
}

UTEST_CASE(identifiable_invalid_id)
{
    auto object = identifiable_t<object_t>{"invalid_id", std::make_unique<object1_t>()};

    std::string str;
    {
        std::ostringstream stream;
        UTEST_CHECK_EQUAL(object.id(), "invalid_id");
        UTEST_CHECK_EQUAL(static_cast<bool>(object), true);
        UTEST_CHECK_EQUAL(object.get().get(), 1);
        UTEST_CHECK_NOTHROW(object.write(stream));
        str = stream.str();
    }
    {
        object = identifiable_t<object_t>{};
        UTEST_CHECK_EQUAL(object.id(), "");
        UTEST_CHECK_EQUAL(static_cast<bool>(object), false);

        std::istringstream stream(str);
        UTEST_CHECK_THROW(object.read(stream), std::runtime_error);
    }
}

UTEST_CASE(identifiable_read_write_many)
{
    auto objects = std::vector<identifiable_t<object_t>>
    {
        {"id2", std::make_unique<object2_t>()},
        {"id1", std::make_unique<object1_t>()},
        {"id3", std::make_unique<object3_t>()},
    };

    std::string str;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(::nano::write(stream, objects));
        str = stream.str();
    }
    {
        objects = std::vector<identifiable_t<object_t>>{};
        UTEST_CHECK_EQUAL(objects.empty(), true);

        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(::nano::read(stream, objects));
        UTEST_REQUIRE_EQUAL(objects.size(), 3U);
        UTEST_CHECK_EQUAL(objects[0].id(), "id2");
        UTEST_CHECK_EQUAL(objects[1].id(), "id1");
        UTEST_CHECK_EQUAL(objects[2].id(), "id3");
        UTEST_CHECK_EQUAL(static_cast<bool>(objects[0]), true);
        UTEST_CHECK_EQUAL(static_cast<bool>(objects[1]), true);
        UTEST_CHECK_EQUAL(static_cast<bool>(objects[2]), true);
        UTEST_CHECK_EQUAL(objects[0].get().get(), 2);
        UTEST_CHECK_EQUAL(objects[1].get().get(), 1);
        UTEST_CHECK_EQUAL(objects[2].get().get(), 3);
    }
}

UTEST_END_MODULE()
