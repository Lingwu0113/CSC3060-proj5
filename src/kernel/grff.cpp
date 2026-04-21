#include "grff.h"
#include <algorithm>
#include <cmath>
#include <random>

void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    args->a_features.resize(size);
    args->b_features.resize(size);
    args->c_features.resize(size);
    args->f_output.resize(size);

    for (size_t i = 0; i < size; ++i) {
        args->a_features[i] = dist(gen);
        args->b_features[i] = dist(gen);
        args->c_features[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Naive Implementation (A Simplified Gated Residual Feature Fusion (GRFF))
// -------------------------------------------------------------------------
void naive_grff(grff_args& args) {
    size_t n = args.a_features.size();
    
    // Intermediate buffers
    std::vector<float> G(n), A_prime(n), Smooth_A(n), B_prime(n), C_prime(n), H(n), E(n);

    // Stage 1: Gate
    for (size_t i = 0; i < n; ++i) 
        G[i] = 0.5f * ((args.a_features[i] * args.b_features[i]) / (1.0f + std::abs(args.a_features[i] * args.b_features[i])) + 1.0f);

    // Stage 2: Update A (Residual)
    for (size_t i = 0; i < n; ++i) 
        A_prime[i] = args.a_features[i] + G[i];

    // Stage 3: Global Feature Scaling
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_a += A_prime[i];
    }
    float avg_a = sum_a / static_cast<float>(n);

    // Stage 4: Update A (Smooth)
    Smooth_A[0] = A_prime[0];
    for (size_t i = 1; i < n; ++i) {
        Smooth_A[i] = (A_prime[i] + A_prime[i-1]) * 0.5f; 
    }

    // Stage 5: Update B (Suppression)
    for (size_t i = 0; i < n; ++i) 
        B_prime[i] = args.b_features[i] * (1.0f - G[i]) * avg_a;

    // Stage 6: Context Integration 
    for (size_t i = 0; i < n; ++i) 
        C_prime[i] = args.c_features[i] + (Smooth_A[i] / (1.0f + std::abs(Smooth_A[i])));

    // Stage 7: Hidden Interaction
    for (size_t i = 0; i < n; ++i) 
        H[i] = Smooth_A[i] * C_prime[i];

    // Stage 8: Normalization
    for (size_t i = 0; i < n; ++i) 
        E[i] = (H[i] + B_prime[i]) / (1.0f + std::abs(Smooth_A[i]));

    // Stage 9: Final Output (ReLU)
    for (size_t i = 0; i < n; ++i) {
        float result = C_prime[i] - E[i];
        args.f_output[i] = std::max(result, 0.0f);
    }
}

// -------------------------------------------------------------------------
// TODO: Student Implementation
// -------------------------------------------------------------------------
void stu_grff(grff_args& args) {
const size_t n = args.a_features.size();
    
    const float* a = args.a_features.data();
    const float* b = args.b_features.data();
    const float* c = args.c_features.data();
    float* out = args.f_output.data();
    
    static thread_local std::vector<float> G, A_prime, B_prime, C_prime, Smooth_A;
    G.resize(n);
    A_prime.resize(n);
    B_prime.resize(n);
    C_prime.resize(n);
    Smooth_A.resize(n);
    
    float* g = G.data();
    float* ap = A_prime.data();
    float* bp = B_prime.data();
    float* cp = C_prime.data();
    float* sm = Smooth_A.data();
    
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float ai = a[i];
        float bi = b[i];
        float ab = ai * bi;
        float g_val = 0.5f * (ab / (1.0f + std::abs(ab)) + 1.0f);
        g[i] = g_val;
        float ap_val = ai + g_val;
        ap[i] = ap_val;
        sum_a += ap_val;
    }
    const float avg_a = sum_a / static_cast<float>(n);
    
    sm[0] = ap[0];
    for (size_t i = 1; i < n; ++i) {
        sm[i] = (ap[i] + ap[i-1]) * 0.5f;
    }
    
    for (size_t i = 0; i < n; ++i) {
        bp[i] = b[i] * (1.0f - g[i]) * avg_a;
    }
    
    for (size_t i = 0; i < n; ++i) {
        float smooth = sm[i];
        cp[i] = c[i] + (smooth / (1.0f + std::abs(smooth)));
    }
    
    for (size_t i = 0; i < n; ++i) {
        float smooth = sm[i];
        float h_val = smooth * cp[i];
        float e_val = (h_val + bp[i]) / (1.0f + std::abs(smooth));
        float result = cp[i] - e_val;
        out[i] = (result > 0.0f) ? result : 0.0f;
    }
}

// -------------------------------------------------------------------------
// Wrappers and Checker
// -------------------------------------------------------------------------
void naive_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    naive_grff(args);
}

void stu_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    stu_grff(args);
}

bool grff_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<grff_args *>(stu_ctx);
    auto &ref_args = *static_cast<grff_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;
    const double atol = 1e-6;

    if (stu_args.f_output.size() != ref_args.f_output.size()) return false;

    for (size_t i = 0; i < ref_args.f_output.size(); ++i) {
        double r = static_cast<double>(ref_args.f_output[i]);
        double s = static_cast<double>(stu_args.f_output[i]);
        double err = std::abs(s - r);

        if (err > (atol + eps * std::abs(r))) {
            debug_log("DEBUG: GRFF fail at %zu: ref=%f stu=%f\n", i, r, s);
            return false;
        }
    }
    return true;
}
