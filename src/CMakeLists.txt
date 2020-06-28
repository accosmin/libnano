# create library

if(NOT APPLE)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

add_library(nano)

target_sources(nano PRIVATE
    table.cpp
    stream.cpp
    dataset/tabular_adult.cpp
    dataset/imclass.cpp
    dataset/imclass_mnist.cpp
    dataset/tabular_abalone.cpp
    dataset/tabular_breast_cancer.cpp
    dataset/tabular_forest_fires.cpp
    dataset/tabular_bank_marketing.cpp
    dataset/tabular.cpp
    dataset/tabular_iris.cpp
    dataset/tabular_wine.cpp
    dataset/imclass_cifar.cpp
    lsearchk/backtrack.cpp
    lsearchk/lemarechal.cpp
    lsearchk/morethuente.cpp
    lsearchk/cgdescent.cpp
    lsearchk/fletcher.cpp
    linear/function.cpp
    linear/model.cpp
    lsearchk.cpp
    solver/lbfgs.cpp
    solver/cgd.cpp
    solver/quasi.cpp
    solver/gd.cpp
    function.cpp
    lsearch0.cpp
    cmdline.cpp
    loss.cpp
    lsearch0/quadratic.cpp
    lsearch0/linear.cpp
    lsearch0/cgdescent.cpp
    lsearch0/constant.cpp
    solver.cpp
    mlearn/train.cpp
    mlearn/cluster.cpp
    gboost/wlearner.cpp
    gboost/wlearner_dtree.cpp
    gboost/wlearner_stump.cpp
    gboost/wlearner_table.cpp
    gboost/wlearner_affine.cpp
    gboost/wlearner_feature1.cpp
    gboost/model.cpp
    gboost/bias.cpp
    gboost/scale.cpp)

add_library(NANO::nano ALIAS nano)
target_compile_configure(nano)

target_include_directories(nano
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(nano
    PUBLIC Eigen3::Eigen)

if(NOT WIN32)
    target_link_libraries(nano
        PUBLIC Threads::Threads
        PRIVATE Threads::Threads)
endif()

# install library
install(TARGETS nano
    EXPORT nano-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(nano PROPERTIES EXPORT_NAME NANO)
