# function to setup compilation flags for various targets
function(target_compile_configure target)
    target_compile_options(${target}
        PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic -Wsign-conversion>
        PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -pedantic -Wsign-conversion>
        PRIVATE $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -pedantic -Wsign-conversion>)
    target_compile_features(${target}
        PUBLIC cxx_std_17
        PRIVATE cxx_std_17)
endfunction()

# function to create a unit test application
function(make_test test libs)
    add_executable(${test} ${test}.cpp)
    target_compile_configure(${test})
    target_compile_definitions(${test}
        PRIVATE UTEST_WITH_EIGEN UTEST_WITH_TENSOR)
    target_include_directories(${test}
        SYSTEM PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/test>)
    target_link_libraries(${test}
        PRIVATE ${libs})
    add_test(${test} ${test})
endfunction()

# function to create an executable
function(make_app app libs)
    add_executable(${app} ${app}.cpp)
    target_compile_configure(${app})
    target_include_directories(${app}
        PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
    target_link_libraries(${app}
        PRIVATE ${libs})
    install(TARGETS ${app}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()
