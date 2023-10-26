#include <chrono>
#include <iostream>
#include <vector>

#include "threadpool.h"

// g++ -g -O3 -o bin/threadpool_example threadpool_example.cpp -std=c++11 -pthread
// low gcc version need add -pthread

int case1() {
    int concurrency = std::thread::hardware_concurrency();
    std::cout << "hardware concurrency:" << concurrency << std::endl;

    ThreadPool pool(concurrency);
    std::vector<std::future<int> > results;

    for (int i = 0; i < concurrency; ++i) {
        results.emplace_back(
            pool.enqueue([i] {
                // std::cout << "hello " << i << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                // std::cout << "world " << i << std::endl;
                return i * i;
            }));
    }

    for (auto&& result : results)
        printf("result:%d\n", result.get());
    return 0;
}

int main() {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    case1();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
    return 0;
}