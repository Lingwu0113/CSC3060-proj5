#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count);
    args->data.b.resize(count);
    args->data.c.resize(count);
    args->data.d.resize(count);
    args->data.e.resize(count);
    args->data.f.resize(count);
    args->data.g.resize(count);
    args->data.h.resize(count);
    args->data.i.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen);
        args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen);
        args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen);
        args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen);
        args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);
    }
}

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    float total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {

            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += data.a[idx];
                    sum_b += data.b[idx];
                    sum_c += data.c[idx];
                }
            }
            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;

            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx =
                -data.d[ym1 + xm1] + data.d[ym1 + xp1]
                -2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1]
                -data.d[yp1 + xm1] + data.d[yp1 + xp1];

            const float sobel_ex =
                -data.e[ym1 + xm1] + data.e[ym1 + xp1]
                -2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1]
                -data.e[yp1 + xm1] + data.e[yp1 + xp1];

            const float sobel_fx =
                -data.f[ym1 + xm1] + data.f[ym1 + xp1]
                -2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1]
                -data.f[yp1 + xm1] + data.f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -data.g[ym1 + xm1] - 2.0f * data.g[ym1 + x0] - data.g[ym1 + xp1]
                + data.g[yp1 + xm1] + 2.0f * data.g[yp1 + x0] + data.g[yp1 + xp1];

            const float sobel_hy =
                -data.h[ym1 + xm1] - 2.0f * data.h[ym1 + x0] - data.h[ym1 + xp1]
                + data.h[yp1 + xm1] + 2.0f * data.h[yp1 + x0] + data.h[yp1 + xp1];

            const float sobel_iy =
                -data.i[ym1 + xm1] - 2.0f * data.i[ym1 + x0] - data.i[ym1 + xp1]
                + data.i[yp1 + xm1] + 2.0f * data.i[yp1 + x0] + data.i[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = total;
}

void stu_filter_gradient(float& out, const optimized_data& opt_data,
                         std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    const PixelData* pixels = opt_data.pixels.data();
    float total = 0.0;
    
    for (std::size_t y = 1; y + 1 < H; ++y) {
        const std::size_t row_ym1 = (y - 1) * W;
        const std::size_t row_y0  = y * W;
        const std::size_t row_yp1 = (y + 1) * W;

        const PixelData* row_ym1_ptr = pixels + row_ym1;
        const PixelData* row_y0_ptr  = pixels + row_y0;
        const PixelData* row_yp1_ptr = pixels + row_yp1;
        
        const PixelData* left_top = &row_ym1_ptr[0];
        const PixelData* left_mid = &row_y0_ptr[0];
        const PixelData* left_bot = &row_yp1_ptr[0];

        const PixelData* mid_top = &row_ym1_ptr[1];
        const PixelData* mid_mid = &row_y0_ptr[1];
        const PixelData* mid_bot = &row_yp1_ptr[1];

        const PixelData* right_top = &row_ym1_ptr[2];
        const PixelData* right_mid = &row_y0_ptr[2];
        const PixelData* right_bot = &row_yp1_ptr[2];

        float colA_left  = left_top->a  + left_mid->a  + left_bot->a;
        float colA_mid   = mid_top->a   + mid_mid->a   + mid_bot->a;
        float colA_right = right_top->a + right_mid->a + right_bot->a;

        float colB_left  = left_top->b  + left_mid->b  + left_bot->b;
        float colB_mid   = mid_top->b   + mid_mid->b   + mid_bot->b;
        float colB_right = right_top->b + right_mid->b + right_bot->b;

        float colC_left  = left_top->c  + left_mid->c  + left_bot->c;
        float colC_mid   = mid_top->c   + mid_mid->c   + mid_bot->c;
        float colC_right = right_top->c + right_mid->c + right_bot->c;

        for (std::size_t x = 1; x + 1 < W; ++x) {
            const float sum_a = colA_left + colA_mid + colA_right;
            const float sum_b = colB_left + colB_mid + colB_right;
            const float sum_c = colC_left + colC_mid + colC_right;

            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;
            
            const float sobel_dx =
                -left_top->d  + right_top->d
                -2.0f * left_mid->d + 2.0f * right_mid->d
                -left_bot->d  + right_bot->d;

            const float sobel_ex =
                -left_top->e  + right_top->e
                -2.0f * left_mid->e + 2.0f * right_mid->e
                -left_bot->e  + right_bot->e;

            const float sobel_fx =
                -left_top->f  + right_top->f
                -2.0f * left_mid->f + 2.0f * right_mid->f
                -left_bot->f  + right_bot->f;

            const float p2 = sobel_dx * sobel_ex + sobel_fx;
            
            const float sobel_gy =
                -left_top->g - 2.0f * mid_top->g - right_top->g
                + left_bot->g + 2.0f * mid_bot->g + right_bot->g;

            const float sobel_hy =
                -left_top->h - 2.0f * mid_top->h - right_top->h
                + left_bot->h + 2.0f * mid_bot->h + right_bot->h;

            const float sobel_iy =
                -left_top->i - 2.0f * mid_top->i - right_top->i
                + left_bot->i + 2.0f * mid_bot->i + right_bot->i;

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
            
            if (x + 2 < W) {
                left_top = mid_top;
                left_mid = mid_mid;
                left_bot = mid_bot;

                mid_top = right_top;
                mid_mid = right_mid;
                mid_bot = right_bot;

                right_top = &row_ym1_ptr[x + 2];
                right_mid = &row_y0_ptr[x + 2];
                right_bot = &row_yp1_ptr[x + 2];

                colA_left  = colA_mid;
                colA_mid   = colA_right;
                colA_right = right_top->a + right_mid->a + right_bot->a;

                colB_left  = colB_mid;
                colB_mid   = colB_right;
                colB_right = right_top->b + right_mid->b + right_bot->b;

                colC_left  = colC_mid;
                colC_mid   = colC_right;
                colC_right = right_top->c + right_mid->c + right_bot->c;
            }
        }
    }
    out = total;
}

void convert_to_optimized(optimized_data& opt, const data_struct& data,
                          std::size_t width, std::size_t height) {
    opt.width = width;
    opt.height = height;
    std::size_t count = width * height;
    opt.pixels.resize(count);
    
    for (std::size_t i = 0; i < count; ++i) {
        opt.pixels[i] = {data.a[i], data.b[i], data.c[i],
                         data.d[i], data.e[i], data.f[i],
                         data.g[i], data.h[i], data.i[i]};
    }
}
void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}
void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    if (!args.is_converted) {
        convert_to_optimized(args.opt_data, args.data, args.width, args.height);
        args.is_converted = true;
    }
    args.out = 0.0f;
    stu_filter_gradient(args.out, args.opt_data, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;
    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;
    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}