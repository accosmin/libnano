#!/usr/bin/env bash

basedir=$(pwd)
installdir=${basedir}/install
libnanodir=${basedir}/build/libnano
exampledir=${basedir}/build/example
clang_tidy_suffix=""
build_type="RelWithDebInfo"
cmake_options=""

generator=""
build_shared="ON"

cores=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu || echo "$NUMBER_OF_PROCESSORS")
threads=$((cores+1))

export PATH="${PATH}:${installdir}"
export CXXFLAGS="${CXXFLAGS} -Werror"

function asan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=address -fno-omit-frame-pointer"
}

function lsan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=leak -fno-omit-frame-pointer"
}

function usan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=undefined -fno-omit-frame-pointer"
}

function msan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=memory -fno-omit-frame-pointer"
}

function tsan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=thread -fno-omit-frame-pointer"
}

function gold {
    export CXXFLAGS="${CXXFLAGS} -fuse-ld=gold"
    export LDFLAGS="${LDFLAGS} -fuse-ld=gold"
}

function native {
    export CXXFLAGS="${CXXFLAGS} -mtune=native -march=native"
}

function libcpp {
    export CXXFLAGS="${CXXFLAGS} -stdlib=libc++"
    export LDFLAGS="${LDFLAGS} -lc++abi"
}

function coverage {
    export CXXFLAGS="${CXXFLAGS} -fno-inline -fno-omit-frame-pointer -fno-inline-small-functions -fno-default-inline"
    export CXXFLAGS="${CXXFLAGS} -coverage -O0"
}

function suffix {
    installdir=${basedir}/install/$1
    libnanodir=${basedir}/build/libnano/$1
    exampledir=${basedir}/build/example/$1

    export PATH="${PATH}:${installdir}"
}

function config {
    cd ${basedir}
    cmake ${generator} -H. -B${libnanodir} ${cmake_options} \
        -DCMAKE_BUILD_TYPE=${build_type} \
        -DBUILD_SHARED_LIBS=${build_shared} \
        -DCMAKE_INSTALL_RPATH=${installdir}/lib \
        -DCMAKE_INSTALL_PREFIX=${installdir} || return 1
}

function build {
    cd ${libnanodir}
    command=$(grep test_mlearn.cpp compile_commands.json | grep command)
    printf "${command}\n"
    cmake --build ${libnanodir} -- -j ${threads} || return 1
}

function tests {
    cd ${libnanodir}
    ctest --output-on-failure -j ${threads} || return 1
}

function install {
    cd ${libnanodir}
    cmake --build ${libnanodir} --target install || return 1
}

function build_example {
    cd ${basedir}
    cmake ${generator} -Hexample -B${exampledir} -DCMAKE_BUILD_TYPE=Debug || return 1
    cd ${exampledir}
    cmake --build ${exampledir} -- -j ${threads} || return 1
}

function cppcheck {
    cd ${libnanodir}

    version=2.1
    installed_version=$(/tmp/cppcheck/bin/cppcheck --version)

    if [ "${installed_version}" != "Cppcheck ${version}" ]; then
        wget -N https://github.com/danmar/cppcheck/archive/${version}.tar.gz || return 1
        tar -xf ${version}.tar.gz > /dev/null || return 1

        OLD_CXXFLAGS=${CXXFLAGS}
        export CXXFLAGS=""
        cd cppcheck-${version} && rm -rf build && mkdir build && cd build
        cmake .. ${generator} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/cppcheck > config.log 2>&1 || return 1
        cmake --build . -- -j ${threads} > build.log 2>&1 || return 1
        cmake --build . --target install > install.log 2>&1 || return 1
        cd ../../
        export CXXFLAGS="${OLD_CXXFLAGS}"
    fi

    /tmp/cppcheck/bin/cppcheck --version || return 1

    # NB: the warnings are not fatal (exitcode=0) as they are usually false alarms!
    #--suppress=shadowFunction
    #--suppress=shadowVar
    #--suppress=unusedFunction
    /tmp/cppcheck/bin/cppcheck \
        --project=compile_commands.json \
        --enable=all --quiet --std=c++17 --error-exitcode=0 --inline-suppr --force \
        --template='{file}:{line},{severity},{id},{message}' \
        --suppress=unknownMacro \
        --suppress=shadowFunction \
        --suppress=unmatchedSuppression || return 1
}

function install_json {
    bash ${basedir}/scripts/install_json.sh || return 1
}

function codecov {
    cd ${basedir}

    local output=${basedir}/coverage.info

    lcov --directory . --gcov-tool ${GCOV} --capture --output-file ${output} || return 1
    lcov --remove ${output} '/usr/*' "${HOME}"'/.cache/*' '*/test/*' '*/external/*' --output-file ${output} || return 1
    lcov --list ${output} || return 1
    genhtml --output lcovhtml ${output} || return 1
    bash <(curl -s https://codecov.io/bash) -f ${output} || return 1
    #bash <(curl -s https://codecov.io/bash) -R ${basedir} -f '!*test_*' || return 1
}

function coveralls {
    cd ${basedir}

    coveralls --root ${basedir} --gcov-options '\-lp' || return 1
}

function build_valgrind {
    version=3.16.0
    installed_version=$(/tmp/valgrind/bin/valgrind --version)

    if [ "${installed_version}" != "valgrind-${version}" ]; then
        wget -N https://sourceware.org/pub/valgrind/valgrind-${version}.tar.bz2 || return 1
        tar -xf valgrind-${version}.tar.bz2 > /dev/null || return 1

        OLD_CXXFLAGS=${CXXFLAGS}
        export CXXFLAGS=""
        cd valgrind-${version}
        ./autogen.sh > autogen.log 2>&1 || return 1
        ./configure --prefix=/tmp/valgrind > config.log 2>&1 || return 1
        make -j > build.log 2>&1 || return 1
        make install > install.log 2>&1 || return 1
        cd ..
        export CXXFLAGS="${OLD_CXXFLAGS}"
    fi

    /tmp/valgrind/bin/valgrind --version || return 1
}

function memcheck {
    cd ${libnanodir}

    build_valgrind || return 1

    # NB: not using ctest directly:
    #   - to use custom build if present
    #   - to pass options to memcheck
    #ctest --output-on-failure -T memcheck

    returncode=0
    utests=$(ls test/test_* | grep -v .log)
    for utest in ${utests}
    do
        printf "Running memcheck@%s ...\n" ${utest}
        log=memcheck_${utest/test\//}.log
        /tmp/valgrind/bin/valgrind --tool=memcheck \
            --leak-check=yes --show-reachable=yes --num-callers=50 --error-exitcode=1 \
            --log-file=${log} ${utest}

        if [[ $? -gt 0 ]]
        then
            cat ${log}
            returncode=1
        fi
        printf "\n"
    done

    return ${returncode}
}

function helgrind {
    cd ${libnanodir}

    build_valgrind || return 1

    returncode=0
    utests="test/test_tpool"
    for utest in ${utests}
    do
        printf "Running helgrind@%s ...\n" ${utest}
        log=helgrind_${utest/test\//}.log
        /tmp/valgrind/bin/valgrind --tool=helgrind \
            --error-exitcode=1 \
            --log-file=${log} ${utest}

        if [[ $? -gt 0 ]]
        then
            cat ${log}
            # NB: ignore for now the warnings reported by helgrind!
            returncode=1
        fi
        printf "\n"
    done

    return ${returncode}
}

function clang_tidy {
    cd ${libnanodir}

    check=$1

    printf "Running $check ...\n"
    log=clang_tidy_${check}.log
    log=${log//\*/ALL}
    log=${log//,-/_NOT}
    printf "Logging to ${log} ...\n"

    wrapper=run-clang-tidy${clang_tidy_suffix}
    wrapper=$(which ${wrapper} || which ${wrapper}.py)
    printf "Using wrapper ${wrapper} ...\n"
    ${wrapper} -clang-tidy-binary clang-tidy${clang_tidy_suffix} \
        -header-filter=.* -checks=-*,${check} -quiet > $log 2>&1

    if [[ $? -ne 0 ]]; then
        cat ${log}
        return 1
    fi

    printf "\n"
    started="0"
    while read line; do
        if [[ $line == *"Enabled checks:"* ]]; then
            printf "${line}\n"
            started="1"
        elif [[ -z ${line} ]]; then
            started="0"
        elif [[ ${started} == "1" ]]; then
            printf "    ${line}\n";
        fi
    done < ${log}

    printf "\n"
    cat $log | grep -E "warning:|error:" | grep -oE "[^ ]+$" | sort | uniq -c
    printf "\n"

    # show log only if any warning or error is detected
    warnings=$(cat $log | grep -E "warning:|error:" | sort -u | grep -v Eigen | wc -l)
    if [[ $warnings -gt 0 ]]
    then
        grep -E "warning:|error:" $log | sort -u
    fi

    # decide if should exit with failure
    if [[ $warnings -gt 0 ]]
    then
        printf "failed with $warnings warnings and errors!\n\n"
        return 1
    else
        printf "passed.\n"
        return 0
    fi
}

function clang_tidy_misc {
    clang_tidy "misc*,-misc-non-private-member-variables-in-classes"
}

function clang_tidy_cert {
    clang_tidy "cert*"
}

function clang_tidy_hicpp {
    checks="hicpp*"
    checks="${checks},-hicpp-avoid-c-arrays"
    checks="${checks},-hicpp-no-array-decay"
    checks="${checks},-hicpp-signed-bitwise"
    clang_tidy ${checks}
}

function clang_tidy_bugprone {
    clang_tidy "bugprone*"
}

function clang_tidy_modernize {
    checks="modernize*"
    checks="${checks},-modernize-avoid-c-arrays"
    checks="${checks},-modernize-use-trailing-return-type"
    clang_tidy ${checks}
}

function clang_tidy_performance {
    clang_tidy "performance*"
}

function clang_tidy_portability {
    clang_tidy "portability*"
}

function clang_tidy_readability {
    checks="readability*"
    checks="${checks},-readability-magic-numbers"
    checks="${checks},-readability-named-parameter"
    checks="${checks},-readability-isolate-declaration"
    checks="${checks},-readability-else-after-return"
    checks="${checks},-readability-convert-member-functions-to-static"
    clang_tidy ${checks}
}

function clang_tidy_clang_analyzer {
    clang_tidy "clang-analyzer*"
}

function clang_tidy_cppcoreguidelines {
    checks="cppcoreguidelines*"
    checks="${checks},-cppcoreguidelines-macro-usage"
    checks="${checks},-cppcoreguidelines-avoid-c-arrays"
    checks="${checks},-cppcoreguidelines-avoid-magic-numbers"
    checks="${checks},-cppcoreguidelines-pro-bounds-pointer-arithmetic"
    checks="${checks},-cppcoreguidelines-pro-bounds-array-to-pointer-decay"
    clang_tidy ${checks}
}

function usage {
    cat <<EOF
usage: $0 [OPTIONS]

options:
    -h,--help
        print usage
    --asan
        setup compiler and linker flags to use the address sanitizer
    --lsan
        setup compiler and linker flags to use the leak sanitizer
    --usan
        setup compiler and linker flags to use the undefined behaviour sanitizer
    --tsan
        setup compiler and linker flags to use the thread sanitizer
    --msan
        setup compiler and linker flags to use the memory sanitizer
    --gold
        setup compiler and linker flags to use the gold linker
    --native
        setup compiler flags to optimize for the native platform
    --libcpp
        setup compiler and linker flags to use libc++
    --coverage
        setup compiler and linker flags to setup code coverage
    --suffix <string>
        suffix for the build and installation directories
    --build-type [Debug,Release,RelWithDebInfo,MinSizeRel]
        build type as defined by CMake
    --config
        generate build using CMake
    --build
        compile the library, the unit tests and the command line applications
    --test
        run the unit tests
    --install
        install the library and the command line applications
    --cppcheck
        run cppcheck (static code analyzer)
    --codecov
        upload code coverage results to codecov.io
    --coveralls
        upload code coverage results to coveralls.io
    --memcheck
        run the unit tests through memcheck (e.g. detects unitialized variales, memory leaks, invalid memory accesses)
    --helgrind
        run the unit tests through helgrind (e.g. detects data races)
    --clang-tidy-check <check name>
        run a particular clang-tidy check (e.g. misc, cert)
    --clang-tidy-suffix <string>
        suffix for the clang-tidy binaries (e.g. -6.0)
    --clang-tidy-misc
    --clang-tidy-cert
    --clang-tidy-hicpp
    --clang-tidy-bugprone
    --clang-tidy-modernize
    --clang-tidy-performance
    --clang-tidy-portability
    --clang-tidy-readability
    --clang-tidy-clang-analyzer
    --clang-tidy-cppcoreguidelines
    --build-example
        build example project
    --generator
        overwrite the default build generator (e.g. --generator Ninja to use Ninja as the build system)
    --shared
        build libnano as a shared library (default)
    --static
        build libnano as a static library
EOF
    exit 1
}

if [ "$1" == "" ]; then
    usage
fi

while [ "$1" != "" ]; do
    case $1 in
        -h | --help)                    usage;;
        --asan)                         asan;;
        --lsan)                         lsan;;
        --usan)                         usan;;
        --tsan)                         tsan;;
        --msan)                         msan;;
        --gold)                         gold;;
        --native)                       native;;
        --libcpp)                       libcpp;;
        --coverage)                     coverage;;
        --suffix)                       shift; suffix $1;;
        --build-type)                   shift; build_type=$1;;
        --config)                       config || exit 1;;
        --build)                        build || exit 1;;
        --test)                         tests || exit 1;;
        --install)                      install || exit 1;;
        --cppcheck)                     cppcheck || exit 1;;
        --codecov)                      codecov || exit 1;;
        --coveralls)                    coveralls || exit 1;;
        --memcheck)                     memcheck || exit 1;;
        --helgrind)                     helgrind || exit 1;;
        --clang-tidy-check)             shift; clang_tidy $1 || exit 1;;
        --clang-tidy-suffix)            shift; clang_tidy_suffix=$1;;
        --clang-tidy-misc)              clang_tidy_misc || exit 1;;
        --clang-tidy-cert)              clang_tidy_cert || exit 1;;
        --clang-tidy-hicpp)             clang_tidy_hicpp || exit 1;;
        --clang-tidy-bugprone)          clang_tidy_bugprone || exit 1;;
        --clang-tidy-modernize)         clang_tidy_modernize || exit 1;;
        --clang-tidy-performance)       clang_tidy_performance || exit 1;;
        --clang-tidy-portability)       clang_tidy_portability || exit 1;;
        --clang-tidy-readability)       clang_tidy_readability || exit 1;;
        --clang-tidy-clang-analyzer)    clang_tidy_clang_analyzer || exit 1;;
        --clang-tidy-cppcoreguidelines) clang_tidy_cppcoreguidelines || exit 1;;
        --build-example)                build_example || exit 1;;
        --generator)                    shift; generator="-G$1";;
        --shared)                       build_shared="ON";;
        --static)                       build_shared="OFF";;
        -D*)                            cmake_options="${cmake_options} $1";;
        *)                              echo "unrecognized option $1"; echo; usage;;
    esac
    shift
done

exit 0
