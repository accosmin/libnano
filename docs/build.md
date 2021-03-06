# Build instructions


#### Structure

The project is organized as follows:
* [include/nano](../include/nano) - C++ headers files representing the library's interface.
* [src](../src) - C++ source files with implementation details.
* [app](../app) - command line C++ utilities mostly used for benchmarking various components.
* [test](../test) - C++ unit tests run with [ctest](https://cmake.org/cmake/help/v3.16/manual/ctest.1.html#ctest-1).
* [docs](../docs) - documentation written in Markdown.
* [cmake](../cmake) - various CMake utilities (e.g. to simplify building unit tests).
* [scripts](../scripts) - various Bash utilities (e.g. to simplify configuring and building of the library, to download datasets or to run experiments).
* [example](../example) - example programs used for testing the setup of the library and for showcasing some of its functionality.


#### Dependencies

System:
* compiler supporting [C++17](https://isocpp.org/wiki/faq/cpp17)
* [CMake](https://cmake.org)
* [Eigen3](https://tuxfamily.org) - high-performance linear-algebra C++ library

Git submodules:
* [utest](https://github.com/accosmin/utest) - (micro) header-only unit test library

Libnano is tested on Ubuntu using gcc (version 5+) and clang (version 4+) and on OSX using AppleClang (version 7+). It may work with minor changes on other platforms as well or using other compiler versions.


#### How to build

The easiest way to build, test and install the library on Linux and OSX is to call [scripts/build.sh](../scripts/build.sh) with the appropriate command line arguments. This script is invoked to run various tests on continuous integrations plaforms like [Travis CI](https://travis-ci.org/accosmin/libnano/builds) and [Codecov](https://codecov.io/gh/accosmin/libnano). See [.travis.yml](../.travis.yml) for examples.


Otherwise, users can also invoke the main CMake script directly for other platforms or for custom builds.


The following command displays the command line arguments supported by [scripts/build.sh](../scripts/build.sh):
```
usage: scripts/build.sh [OPTIONS]

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
    --build-example
        build example project
    --generator
        overwrite the default build generator (e.g. --generator Ninja to use Ninja as the build system)
    --shared
        build libnano as a shared library (default)
    --static
        build libnano as a static library
```

The order of the command line arguments matter: the configuration parameters should be first and the actions should follow in the right order (e.g. configuration, then compilation, then testing and then installation). The build script and the CMake scripts do not override environmental variables like ```CXXFLAGS``` or ```LDFLAGS``` and as such the library can be easily wrapped by a package manager.


Examples:
* build the Debug build and run the unit tests in the folder ```build/libnano/debug```:
```
bash scripts/build.sh --suffix debug --build-type Debug --config --build --test
```

* build the natively-optimized release build in the folder ```build/libnano/release```, run the unit tests, install the library in ```install/release``` and build the examples in the folder ```build/example/release```:
```
bash scripts/build.sh --suffix release --build-type Release --native --config --build --test --install --build-example
```

* build the Debug build and run the unit tests in the folder ```build/libnano/debug``` with memcheck:
```
bash scripts/build.sh --suffix debug --build-type Debug --config --build --memcheck
```


NB: Use the ```--native``` flag when compiling for ```Release``` builds to maximize performance on a given machine. This is because Eigen3 uses vectorization internally for the linear algebra operations. Please note that the resulting binaries may not be usable on another platform.
