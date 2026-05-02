#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// TODO: Optimize the bitwise function
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    constexpr std::uint8_t  kBase8  = 0xA5u;
    constexpr std::uint8_t  kMask8  = 0x99u;
    constexpr std::uint64_t kBase64 = 0xA5A5A5A5A5A5A5A5ULL;
    constexpr std::uint64_t kMask64 = 0x9999999999999999ULL;

    const std::size_t n = std::min(result.size(), std::min(a.size(), b.size()));

    const std::int8_t* a_ptr = a.data();
    const std::int8_t* b_ptr = b.data();
    std::int8_t* r_ptr = result.data();

    std::size_t i = 0;

    const std::size_t n64 = n - (n % 64);
    for (; i < n64; i += 64) {
        std::uint64_t a0, a1, a2, a3, a4, a5, a6, a7;
        std::uint64_t b0, b1, b2, b3, b4, b5, b6, b7;

        std::memcpy(&a0, a_ptr + i,      8);
        std::memcpy(&a1, a_ptr + i + 8,  8);
        std::memcpy(&a2, a_ptr + i + 16, 8);
        std::memcpy(&a3, a_ptr + i + 24, 8);
        std::memcpy(&a4, a_ptr + i + 32, 8);
        std::memcpy(&a5, a_ptr + i + 40, 8);
        std::memcpy(&a6, a_ptr + i + 48, 8);
        std::memcpy(&a7, a_ptr + i + 56, 8);

        std::memcpy(&b0, b_ptr + i,      8);
        std::memcpy(&b1, b_ptr + i + 8,  8);
        std::memcpy(&b2, b_ptr + i + 16, 8);
        std::memcpy(&b3, b_ptr + i + 24, 8);
        std::memcpy(&b4, b_ptr + i + 32, 8);
        std::memcpy(&b5, b_ptr + i + 40, 8);
        std::memcpy(&b6, b_ptr + i + 48, 8);
        std::memcpy(&b7, b_ptr + i + 56, 8);

        const std::uint64_t r0 = kBase64 ^ ((a0 | b0) & kMask64);
        const std::uint64_t r1 = kBase64 ^ ((a1 | b1) & kMask64);
        const std::uint64_t r2 = kBase64 ^ ((a2 | b2) & kMask64);
        const std::uint64_t r3 = kBase64 ^ ((a3 | b3) & kMask64);
        const std::uint64_t r4 = kBase64 ^ ((a4 | b4) & kMask64);
        const std::uint64_t r5 = kBase64 ^ ((a5 | b5) & kMask64);
        const std::uint64_t r6 = kBase64 ^ ((a6 | b6) & kMask64);
        const std::uint64_t r7 = kBase64 ^ ((a7 | b7) & kMask64);

        std::memcpy(r_ptr + i,      &r0, 8);
        std::memcpy(r_ptr + i + 8,  &r1, 8);
        std::memcpy(r_ptr + i + 16, &r2, 8);
        std::memcpy(r_ptr + i + 24, &r3, 8);
        std::memcpy(r_ptr + i + 32, &r4, 8);
        std::memcpy(r_ptr + i + 40, &r5, 8);
        std::memcpy(r_ptr + i + 48, &r6, 8);
        std::memcpy(r_ptr + i + 56, &r7, 8);
    }

    const std::size_t n8 = n - ((n - i) % 8);
    for (; i < n8; i += 8) {
        std::uint64_t va, vb;
        std::memcpy(&va, a_ptr + i, 8);
        std::memcpy(&vb, b_ptr + i, 8);

        const std::uint64_t res = kBase64 ^ ((va | vb) & kMask64);
        std::memcpy(r_ptr + i, &res, 8);
    }

    for (; i < n; ++i) {
        const std::uint8_t x =
            static_cast<std::uint8_t>(a_ptr[i]) |
            static_cast<std::uint8_t>(b_ptr[i]);
        r_ptr[i] = static_cast<std::int8_t>(kBase8 ^ (x & kMask8));
    }
}
void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    // Call your verion here
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}
