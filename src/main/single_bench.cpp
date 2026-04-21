#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "relu.h"
#include "bitwise.h"
#include "graph.h"
#include "filter_gradient.h"


int main() {
    std::uint32_t seed = 12345u;
    //relu
    //constexpr size_t relu_size = 1024000;
    //relu_args relu_args_naive;
    //relu_args relu_args_stu;
    //initialize_relu(&relu_args_naive, relu_size, seed);
    //initialize_relu(&relu_args_stu, relu_size, seed);
    //std::println("\tReLU: vector length={}", relu_size);

    //bitwise
    //constexpr size_t bitwise_size = 1024000;
    //bitwise_args bitwise_args_naive;
    //bitwise_args bitwise_args_stu;
    //initialize_bitwise(&bitwise_args_naive, bitwise_size, seed);
    //initialize_bitwise(&bitwise_args_stu, bitwise_size, seed);
    //std::println("\tBitwise: vector length={}", bitwise_size);

    //filter_gradient
    //constexpr size_t width = 1024;
    //constexpr size_t height = 1024;
    //filter_gradient_args filter_args;
    //initialize_filter_gradient(&filter_args, width, height, seed);
    //std::println("\tFilterGradient: {} x {}", width, height);

    //graph
    constexpr size_t node_count = 1024000;
    constexpr int avg_degree = 8;
    graph_args graph_args_naive;
    graph_args graph_args_stu;
    initialize_graph(&graph_args_naive, node_count, avg_degree, seed);
    initialize_graph(&graph_args_stu, node_count, avg_degree, seed);
    std::println("\tGraph: node_count={}, avg_degree={}", node_count, avg_degree);

    std::vector<bench_t> benchmarks = {
                //{"ReLU (Naive)",
                //naive_relu_wrapper,
                //naive_relu_wrapper,
                //relu_check,
                //&relu_args_naive,
                //&relu_args_naive,
                //BASELINE_RELU},

                //{"ReLU (Stu)",
                //stu_relu_wrapper,
                //naive_relu_wrapper,
                //relu_check,
                //&relu_args_stu,
                //&relu_args_naive,
                //BASELINE_RELU},

                //{"Bitwise (Naive)",
                //naive_bitwise_wrapper,
                //naive_bitwise_wrapper,
                //bitwise_check,
                //&bitwise_args_naive,
                //&bitwise_args_naive,
                //BASELINE_BITWISE},

                //{"Bitwise (Stu)",
                //stu_bitwise_wrapper,
                //naive_bitwise_wrapper,
                //bitwise_check,
                //&bitwise_args_stu,
                //&bitwise_args_naive,
                //BASELINE_BITWISE},

                //{"FilterGradient",
                //stu_filter_gradient_wrapper,
                //naive_filter_gradient_wrapper,
                //filter_gradient_check,
                //&filter_args,
                //&filter_args,
                //BASELINE_FILTER_GRADIENT}

                {"Graph (Naive)",
                naive_graph_wrapper,
                naive_graph_wrapper,
                graph_check,
                &graph_args_naive,
                &graph_args_naive,
                BASELINE_GRAPH},

                {"Graph (Stu)",
                stu_graph_wrapper,
                naive_graph_wrapper,
                graph_check,
                &graph_args_stu,
                &graph_args_naive,
                BASELINE_GRAPH}
                
    };
    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << "\n";
    std::cout << "--------------------------------------------------------\n";

    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};
        const int k_best = 20;

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });

            avg_time += elapsed;
            debug_log("\tDEBUG: {}-th measurement: {} ns\n",
                      i,
                      static_cast<std::uint64_t>(elapsed.count()));
        }
        avg_time /= static_cast<uint64_t>(k_best);

        bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << "\n";
            std::cout
                << "  Error: Results do not match naive implementation!\n";
        } else {
            std::cout << "\033[1;32mPASSED\033[0m" << std::right
                      << std::setw(15) << avg_time.count() << " ns";
            if (avg_time.count() > bench.baseline_time.count() * 1.1) {
                std::cout << " (SLOW)";
            }
            std::cout << "\n";
        }
    }

    return 0;
}