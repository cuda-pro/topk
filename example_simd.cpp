// g++ -mavx2 -O3 -Wall --std=c++11 -march=native example_simd.cpp -o bin/example_simd
/*
-mavx                           -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2 and AVX built-in functions and code generation
-mavx2                          -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation
-mavx256-split-unaligned-load   -- split 32-byte AVX unaligned load
-mavx256-split-unaligned-store  -- split 32-byte AVX unaligned store
-mavx5124fmaps                  -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512F and AVX5124FMAPS built- in functions and code gener
-mavx5124vnniw                  -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512F and AVX5124VNNIW built- in functions and code gener
-mavx512bf16                    -- avx512bf16
-mavx512bitalg                  -- avx512bitalg
-mavx512bw                      -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F and AVX512BW built- in functions and code genera
-mavx512cd                      -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F and AVX512CD built- in functions and code genera
-mavx512dq                      -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F and AVX512DQ built- in functions and code genera
-mavx512er                      -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F and AVX512ER built- in functions and code genera
-mavx512f                       -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F built-in functions and code generation
-mavx512ifma                    -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F and AVX512IFMA built-in functions and code gener
-mavx512pf                      -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F and AVX512PF built- in functions and code genera
-mavx512vbmi                    -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F and AVX512VBMI built-in functions and code gener
-mavx512vbmi2                   -- avx512vbmi2
-mavx512vl                      -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX512F and AVX512VL built- in functions and code genera
-mavx512vnni                    -- avx512vnni
-mavx512vp2intersect            -- avx512vp2intersect
-mavx512vpopcntdq               -- support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512F and AVX512VPOPCNTDQ built-in functions and code gen
-mavxvnni                       -- avxvnni
*/
/*
- https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
- https://lemire.me/blog/2023/09/22/parsing-integers-quickly-with-avx-512/
- https://github.com/simdjson/simdjson/issues/10
- https://github.com/Tencent/rapidjson/issues/499
- https://github.com/Highload-fun/platform/wiki
*/
// #include <bits/stdc++.h>
#include <immintrin.h>

#include <chrono>
#include <iostream>
#include <vector>

typedef std::chrono::high_resolution_clock Clock;
void out(int D[]) {
    for (int i = 0; i < 8; i++) {
        std::cout << D[i] << " ";
    }
    std::cout << std::endl;
}

void vec_add_AVX2_case1() {
    // define AVX2 cpu register int __m256i
    __m256i x;
    __m256i y;
    __m256i z;

    // define array on stack
    int A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int ans[8];

    // align load to register
    x = _mm256_loadu_si256((__m256i *)&A[0]);
    y = _mm256_loadu_si256((__m256i *)&B[0]);

    // x add y to z register
    z = _mm256_add_epi64(x, y);

    // store z to ans on stack from register
    _mm256_storeu_si256((__m256i *)&ans[0], z);

    out(A);
    out(B);
    out(ans);
}

int main() {
    vec_add_AVX2_case1();
    return 0;
}