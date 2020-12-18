#!/usr/bin/env bash

CXX=g++ bash scripts/build.sh --suffix gcc-debug --build-type Debug \
    --generator Ninja --config --build --test --install --build-example

CXX=g++ bash scripts/build.sh --suffix gcc-release --build-type Release --native \
    --generator Ninja --config --build --test --install --build-example

CXX=g++ bash scripts/build.sh --suffix cppcheck --config --cppcheck

CXX=g++ bash scripts/build.sh --suffix memcheck --build-type RelWithDebInfo --config --build --memcheck

CXX=g++ bash scripts/build.sh --suffix gcc-asan --build-type Debug --asan \
    --generator Ninja --config --build --test

CXX=g++ bash scripts/build.sh --suffix gcc-lsan --build-type Debug --lsan \
    --generator Ninja --config --build --test

CXX=g++ bash scripts/build.sh --suffix gcc-usan --build-type Debug --usan \
    --generator Ninja --config --build --test

CXX=g++ bash scripts/build.sh --suffix gcc-tsan --build-type Debug --tsan \
    --generator Ninja --config --build --test

CXX=clang++ bash scripts/build.sh --suffix clang-tidy \
    --generator Ninja --config --build  --clang-tidy-cert

CXX=clang++ bash scripts/build.sh --suffix clang-debug --build-type Debug \
    --generator Ninja --config --build --test --install --build-example

