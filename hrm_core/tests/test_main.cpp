// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include <iostream>
#include <stdexcept>

int main() {
    try {
        extern void test_signature();
        extern void test_payload();
        extern void test_router_index();
        test_signature();
        test_payload();
        test_router_index();
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}

