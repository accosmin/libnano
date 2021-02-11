#pragma once

// see https://en.cppreference.com/w/cpp/utility/variant/visit
// FIXME: this won't be needed in C++20
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
