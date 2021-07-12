#include <fstream>
#include <utest/utest.h>
#include <nano/core/cmdline.h>

UTEST_BEGIN_MODULE(test_core_cmdline)

UTEST_CASE(empty)
{
    nano::cmdline_t cmdline("unit testing");

    UTEST_CHECK(!cmdline.has("h"));
    UTEST_CHECK(!cmdline.has("help"));
    UTEST_CHECK_THROW(cmdline.has("v"), std::runtime_error);

    UTEST_CHECK_THROW(cmdline.get<int>("version"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.get<std::string>("f"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.get<std::string>("file"), std::runtime_error);
}

UTEST_CASE(usage)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version number", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations"));

    std::stringstream os;
    cmdline.usage(os);

    UTEST_CHECK_EQUAL(os.str(), R"XXX(unit testing
  -h,--help            usage
  -v,--version(0.3)    version number
  --iterations         number of iterations

)XXX");
}

UTEST_CASE(parse_chars)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "trials", "number of trials"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations"));

    const int argc = 3;
    const char* argv[] = { "", "-v", "0.3.0" };

    UTEST_CHECK_NOTHROW(cmdline.process(argc, argv));

    UTEST_CHECK(cmdline.has("v"));
    UTEST_CHECK(cmdline.has("version"));
    UTEST_CHECK(!cmdline.has("trials"));
    UTEST_CHECK(!cmdline.has("iterations"));
    UTEST_CHECK(!cmdline.has("h"));
    UTEST_CHECK(!cmdline.has("help"));

    UTEST_CHECK_EQUAL(cmdline.get<std::string>("v"), "0.3.0");
    UTEST_CHECK_THROW(cmdline.get<int>("trials"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.get<int>("iterations"), std::runtime_error);
}

UTEST_CASE(parse_string)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", 127));

    UTEST_CHECK_NOTHROW(cmdline.process("-v --iterations 7"));

    UTEST_CHECK(cmdline.has("v"));
    UTEST_CHECK(cmdline.has("version"));
    UTEST_CHECK(cmdline.has("iterations"));
    UTEST_CHECK(!cmdline.has("h"));
    UTEST_CHECK(!cmdline.has("help"));

    UTEST_CHECK_EQUAL(cmdline.get<int>("iterations"), 7);
    UTEST_CHECK_EQUAL(cmdline.get<std::string>("v"), "0.3");
}

UTEST_CASE(error_invalid_arg1)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int argc = 4;
    const char* argv[] = { "", "v", "--version", "7" };

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_invalid_arg2)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int argc = 4;
    const char* argv[] = { "", "--version", "7", "-", "--version", "13" };

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_invalid_arg3)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int argc = 4;
    const char* argv[] = { "", "--version", "11", "--" };

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_unknown_arg)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int argc = 4;
    const char* argv[] = { "", "-v", "--what", "7" };

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(parse_config_file)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const std::string path = "config";
    {
        std::ofstream out(path.c_str());
        out << "-v\n";
        out << "--iterations 29";
    }

    UTEST_CHECK_NOTHROW(cmdline.process_config_file(path));

    UTEST_CHECK(cmdline.has("v"));
    UTEST_CHECK(cmdline.has("version"));
    UTEST_CHECK(cmdline.has("iterations"));
    UTEST_CHECK(!cmdline.has("h"));
    UTEST_CHECK(!cmdline.has("help"));

    UTEST_CHECK_EQUAL(cmdline.get<std::string>("v"), "0.3");
    UTEST_CHECK_EQUAL(cmdline.get<int>("iterations"), 29);

    std::remove(path.c_str());
}

UTEST_CASE(error_invalid_options)
{
    nano::cmdline_t cmdline("unit testing");

    UTEST_CHECK_THROW(cmdline.add("v", "", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("v", "-", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("v", "--", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("v", "--version", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-", "version", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("vv", "version", ""), std::runtime_error);
}

UTEST_CASE(error_duplicate_options)
{
    nano::cmdline_t cmdline("unit testing");

    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", ""));
    UTEST_CHECK_THROW(cmdline.add("v", "version", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("v", "wersion", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("w", "version", ""), std::runtime_error);
}

UTEST_END_MODULE()
