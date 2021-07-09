#include <sstream>
#include <iomanip>
#include <utest/utest.h>
#include <nano/string.h>
#include <nano/core/logger.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_core_logger)

UTEST_CASE(log)
{
    UTEST_CHECK_NOTHROW(log_info() << "info message");
    UTEST_CHECK_NOTHROW(log_error() << "error message");
    UTEST_CHECK_NOTHROW(log_warning() << "warning message");
    UTEST_CHECK_NOTHROW(logger_t(static_cast<logger_t::type>(42)) << "what message");
}

UTEST_CASE(info)
{
    std::ostringstream stream_cout, stream_cerr;
    log_info(&stream_cout, &stream_cerr) << std::flush << "info message" << '\n' << std::endl;
    UTEST_CHECK(ends_with(stream_cout.str(), ": info message\n\n\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");

    stream_cout.str("");
    stream_cerr.str("");
    log_info(&stream_cout, nullptr) << std::flush << "info message" << '\n' << std::endl;
    UTEST_CHECK(ends_with(stream_cout.str(), ": info message\n\n\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");

    stream_cout.str("");
    stream_cerr.str("");
    log_info(nullptr, &stream_cerr) << std::flush << "info message" << '\n' << std::endl;
    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_CASE(error)
{
    std::ostringstream stream_cout, stream_cerr;
    stream_cerr << std::setprecision(3);
    log_error(&stream_cout, &stream_cerr) << std::setprecision(7) << "error message";
    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK(ends_with(stream_cerr.str(), ": error message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.precision(), 3);

    stream_cout.str("");
    stream_cerr.str("");
    log_error(&stream_cout, nullptr) << std::setprecision(4) << "error message";
    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
    UTEST_CHECK_EQUAL(stream_cerr.precision(), 3);

    stream_cout.str("");
    stream_cerr.str("");
    log_error(nullptr, &stream_cerr) << std::setprecision(12) << "error message";
    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK(ends_with(stream_cerr.str(), ": error message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.precision(), 3);
}

UTEST_CASE(warning)
{
    std::ostringstream stream_cout, stream_cerr;
    log_warning(&stream_cout, &stream_cerr) << "warning message";
    UTEST_CHECK(ends_with(stream_cout.str(), ": warning message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");

    stream_cout.str("");
    stream_cerr.str("");
    log_warning(&stream_cout, nullptr) << "warning message";
    UTEST_CHECK(ends_with(stream_cout.str(), ": warning message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");

    stream_cout.str("");
    stream_cerr.str("");
    log_warning(nullptr, &stream_cerr) << "warning message";
    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_CASE(unknown)
{
    std::ostringstream stream_cout, stream_cerr;
    logger_t(static_cast<logger_t::type>(42), &stream_cout, &stream_cerr) << "unknown message";
    UTEST_CHECK(ends_with(stream_cout.str(), ": unknown message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");

    stream_cout.str("");
    stream_cerr.str("");
    logger_t(static_cast<logger_t::type>(42), nullptr, &stream_cerr) << "unknown message";
    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");

    stream_cout.str("");
    stream_cerr.str("");
    logger_t(static_cast<logger_t::type>(42), &stream_cout, nullptr) << "unknown message";
    UTEST_CHECK(ends_with(stream_cout.str(), ": unknown message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_CASE(critical)
{
    UTEST_CHECK_NOTHROW(critical(false, "message"));
    UTEST_CHECK_THROW(critical(true, "message"), std::runtime_error);
}

UTEST_CASE(main)
{
    const auto op_ok = [] (int, const char* [])
    {
        return EXIT_SUCCESS;
    };

    const auto op_unknown = [] (int, const char* [])
    {
        throw 42; // NOLINT(hicpp-exception-baseclass)
        return EXIT_SUCCESS;
    };

    const auto op_exception = [] (int, const char* [])
    {
        throw std::runtime_error("runtime error");
        return EXIT_SUCCESS;
    };

    const auto argc = 1;
    const char* argv[] = {"main"};

    UTEST_CHECK_NOTHROW(nano::main(op_ok, argc, argv));
    UTEST_CHECK_NOTHROW(nano::main(op_unknown, argc, argv));
    UTEST_CHECK_NOTHROW(nano::main(op_exception, argc, argv));

    UTEST_CHECK_EQUAL(nano::main(op_ok, argc, argv), EXIT_SUCCESS);
    UTEST_CHECK_EQUAL(nano::main(op_unknown, argc, argv), EXIT_FAILURE);
    UTEST_CHECK_EQUAL(nano::main(op_exception, argc, argv), EXIT_FAILURE);
}

UTEST_END_MODULE()
