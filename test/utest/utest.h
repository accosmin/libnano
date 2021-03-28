#pragma once

#include <cmath>
#include <atomic>
#include <string>
#include <iomanip>
#include <iostream>

#define UTEST_STRINGIFY_(x) #x
#define UTEST_STRINGIFY(x) UTEST_STRINGIFY_(x)

static std::string utest_case_name;
static std::string utest_module_name;

static std::size_t utest_n_cases = 0;
static std::atomic<std::size_t> utest_n_checks = {0};
static std::atomic<std::size_t> utest_n_failures = {0};

enum class exception_status
{
    none,
    expected,
    unexpected
};

template <typename texception, typename toperator>
static exception_status check_throw(const toperator& op)
{
    try
    {
        op();
        return exception_status::none;
    }
    catch (texception&)
    {
        return exception_status::expected;
    }
    catch (...)
    {
        return exception_status::unexpected;
    }
}

#define UTEST_BEGIN_MODULE(name) \
int main(int, char* []) /*NOLINT(hicpp-function-size,readability-function-size)*/ \
{ \
try \
{ \
    utest_module_name = #name;

#define UTEST_CASE(name) \
    ++ utest_n_cases; \
    utest_case_name = #name; \
    std::cout << "running test case [" << utest_module_name << "/" << utest_case_name << "] ..." << std::endl;

#define UTEST_END_MODULE() \
    if (utest_n_failures > 0) \
    { \
        std::cout << "  failed with " << utest_n_failures << " errors in " << utest_n_checks \
              << " check" << (utest_n_checks > 0 ? "s" : "") << "!" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
    else \
    { \
        std::cout << "  no errors detected in " << utest_n_checks \
              << " check" << (utest_n_checks > 0 ? "s" : "") << "." << std::endl; \
        exit(EXIT_SUCCESS); \
    } \
} \
catch (std::exception& e) \
{ \
    std::cout << " failed with uncaught exception <" << e.what() << ">!" << std::endl; \
    exit(EXIT_FAILURE); \
} \
catch (...) \
{ \
    std::cout << " failed with uncaught unknown exception!" << std::endl; \
    exit(EXIT_FAILURE); \
} \
}

#define UTEST_HANDLE_CRITICAL(critical) \
    if (critical) \
    { \
        exit(EXIT_FAILURE); \
    }
#define UTEST_HANDLE_FAILURE() \
    ++ utest_n_failures; \
    std::cout << __FILE__ << ":" << __LINE__ << std::fixed << std::setprecision(12) << ": [" << utest_module_name << "/" << utest_case_name

#define UTEST_EVALUATE(check, critical) \
    ++ utest_n_checks; \
    if (!(check)) \
    { \
        UTEST_HANDLE_FAILURE() \
            << "]: check {" << UTEST_STRINGIFY(check) << "} failed!" << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    }
#define UTEST_CHECK(check) \
    UTEST_EVALUATE(check, false)
#define UTEST_REQUIRE(check) \
    UTEST_EVALUATE(check, true)

#define UTEST_THROW(call, exception, critical) \
    ++ utest_n_checks; \
    switch (check_throw<exception>([&] () { (void)(call); })) \
    { \
    case exception_status::none: \
        UTEST_HANDLE_FAILURE() \
            << "]: call {" << UTEST_STRINGIFY(call) << "} does not throw!" << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    case exception_status::expected: \
        break; \
    case exception_status::unexpected: \
        UTEST_HANDLE_FAILURE() \
            << "]: call {" << UTEST_STRINGIFY(call) << "} does not throw {" \
            << UTEST_STRINGIFY(exception) << "}!" << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    }
#define UTEST_CHECK_THROW(call, exception) \
    UTEST_THROW(call, exception, false)
#define UTEST_REQUIRE_THROW(call, exception) \
    UTEST_THROW(call, exception, true)

#define UTEST_NOTHROW(call, critical) \
    ++ utest_n_checks; \
    switch (check_throw<std::exception>([&] () { (void)(call); })) \
    { \
    case exception_status::none: \
        break; \
    case exception_status::expected: \
    case exception_status::unexpected: \
        UTEST_HANDLE_FAILURE() \
            << "]: call {" << UTEST_STRINGIFY(call) << "} throws!" << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    }
#define UTEST_CHECK_NOTHROW(call) \
    UTEST_NOTHROW(call, false)
#define UTEST_REQUIRE_NOTHROW(call) \
    UTEST_NOTHROW(call, true)

#define UTEST_EVALUATE_BINARY_OP(left, right, op, critical) \
{ \
    ++ utest_n_checks; \
    const auto res_left = (left); /* NOLINT */ \
    const auto res_right = (right); /* NOLINT */ \
    if (!(res_left op res_right)) \
    { \
        UTEST_HANDLE_FAILURE() \
            << "]: check {" << UTEST_STRINGIFY(left op right) \
            << "} failed {" << res_left << " " << UTEST_STRINGIFY(op) << " " << res_right << "}!" << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    } \
}

#define UTEST_EVALUATE_EQUAL(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, ==, critical)
#define UTEST_CHECK_EQUAL(left, right) \
    UTEST_EVALUATE_EQUAL(left, right, false)
#define UTEST_REQUIRE_EQUAL(left, right) \
    UTEST_EVALUATE_EQUAL(left, right, true)

#define UTEST_EVALUATE_NOT_EQUAL(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, !=, critical)
#define UTEST_CHECK_NOT_EQUAL(left, right) \
    UTEST_EVALUATE_NOT_EQUAL(left, right, false)
#define UTEST_REQUIRE_NOT_EQUAL(left, right) \
    UTEST_EVALUATE_NOT_EQUAL(left, right, true)

#define UTEST_EVALUATE_LESS(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, <, critical)
#define UTEST_CHECK_LESS(left, right) \
    UTEST_EVALUATE_LESS(left, right, false)
#define UTEST_REQUIRE_LESS(left, right) \
    UTEST_EVALUATE_LESS(left, right, true)

#define UTEST_EVALUATE_LESS_EQUAL(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, <=, critical)
#define UTEST_CHECK_LESS_EQUAL(left, right) \
    UTEST_EVALUATE_LESS_EQUAL(left, right, false)
#define UTEST_REQUIRE_LESS_EQUAL(left, right) \
    UTEST_EVALUATE_LESS_EQUAL(left, right, true)

#define UTEST_EVALUATE_GREATER(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, >, critical);
#define UTEST_CHECK_GREATER(left, right) \
    UTEST_EVALUATE_GREATER(left, right, false);
#define UTEST_REQUIRE_GREATER(left, right) \
    UTEST_EVALUATE_GREATER(left, right, true);

#define UTEST_EVALUATE_GREATER_EQUAL(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, >=, critical);
#define UTEST_CHECK_GREATER_EQUAL(left, right) \
    UTEST_EVALUATE_GREATER_EQUAL(left, right, false);
#define UTEST_REQUIRE_GREATER_EQUAL(left, right) \
    UTEST_EVALUATE_GREATER_EQUAL(left, right, true);

#define UTEST_EVALUATE_CLOSE(left, right, epsilon, critical) \
    UTEST_EVALUATE_LESS(std::fabs((left) - (right)), epsilon * (1 + std::fabs((left)) + std::fabs((right))), critical)
#define UTEST_CHECK_CLOSE(left, right, epsilon) \
    UTEST_EVALUATE_CLOSE(left, right, epsilon, false);
#define UTEST_REQUIRE_CLOSE(left, right, epsilon) \
    UTEST_EVALUATE_CLOSE(left, right, epsilon, true);

#ifdef UTEST_WITH_EIGEN

#define UTEST_EVALUATE_EIGEN_CLOSE(left, right, epsilon, critical) \
    UTEST_REQUIRE_EQUAL((left).size(), (right).size()); \
    UTEST_EVALUATE_LESS((((left) - (right)).array().abs().maxCoeff()), \
        epsilon * (1 + (left).array().abs().maxCoeff() + (right).array().abs().maxCoeff()), critical)

#define UTEST_CHECK_EIGEN_CLOSE(left, right, epsilon) \
    UTEST_EVALUATE_EIGEN_CLOSE(left, right, epsilon, false);
#define UTEST_REQUIRE_EIGEN_CLOSE(left, right, epsilon) \
    UTEST_EVALUATE_EIGEN_CLOSE(left, right, epsilon, true);

#define UTEST_EVALUATE_EIGEN_EQUAL(left, right, critical) \
    UTEST_REQUIRE_EQUAL((left).size(), (right).size()); \
    UTEST_EVALUATE_EQUAL((left), (right), critical);

#define UTEST_CHECK_EIGEN_EQUAL(left, right) \
    UTEST_EVALUATE_EIGEN_EQUAL(left, right, false);
#define UTEST_REQUIRE_EIGEN_EQUAL(left, right) \
    UTEST_EVALUATE_EIGEN_EQUAL(left, right, true);

#endif

#ifdef UTEST_WITH_TENSOR

#define UTEST_EVALUATE_TENSOR_CLOSE(left, right, epsilon, critical) \
    UTEST_REQUIRE_EQUAL((left).dims(), (right).dims()); \
    ++ utest_n_checks; \
    if (!close((left), (right), epsilon)) \
    { \
        UTEST_HANDLE_FAILURE() \
            << "]: check {" << UTEST_STRINGIFY(left <> right) \
            << "} failed {" << (left) << " <> " << (right) << "}!" << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    }

#define UTEST_CHECK_TENSOR_CLOSE(left, right, epsilon) \
    UTEST_EVALUATE_TENSOR_CLOSE(left, right, epsilon, false);
#define UTEST_REQUIRE_TENSOR_CLOSE(left, right, epsilon) \
    UTEST_EVALUATE_TENSOR_CLOSE(left, right, epsilon, true);

#define UTEST_EVALUATE_TENSOR_EQUAL(left, right, critical) \
    UTEST_REQUIRE_EQUAL((left).dims(), (right).dims()); \
    UTEST_EVALUATE_EQUAL((left), (right), critical);

#define UTEST_CHECK_TENSOR_EQUAL(left, right) \
    UTEST_EVALUATE_TENSOR_EQUAL(left, right, false);
#define UTEST_REQUIRE_TENSOR_EQUAL(left, right) \
    UTEST_EVALUATE_TENSOR_EQUAL(left, right, true);

#endif
