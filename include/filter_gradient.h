#ifndef FILTER_GRADIENT_H
#define FILTER_GRADIENT_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

inline constexpr std::chrono::nanoseconds BASELINE_FILTER_GRADIENT{25000000};

struct data_struct {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> e;
    std::vector<float> f;
    std::vector<float> g;
    std::vector<float> h;
    std::vector<float> i;
};

struct alignas(64) PixelData {
    float a, b, c, d, e, f, g, h, i;
};

struct optimized_data {
    std::vector<PixelData> pixels;
    std::size_t width;
    std::size_t height;
};

struct filter_gradient_args {
    data_struct data; 
    optimized_data opt_data;
    bool is_converted = false;
    std::size_t width;
    std::size_t height;
    float out;
    double epsilon;

    explicit filter_gradient_args(double epsilon_in = 1e-6)
        : width(0), height(0), out(0.0f), epsilon(epsilon_in) {}
};

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height);
void stu_filter_gradient(float& out, const optimized_data& opt_data,
                   std::size_t width, std::size_t height);

void naive_filter_gradient_wrapper(void* ctx);
void stu_filter_gradient_wrapper(void* ctx);

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed);

                        
bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);
void convert_to_optimized(optimized_data& opt, const data_struct& data, 
                          std::size_t width, std::size_t height);
#endif // filter_gradient_H