#include "graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

void initialize_graph(graph_args* args,
                      std::size_t node_count,
                      int avg_degree,
                      std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node_count) - 1);

    const std::size_t total_edges =
        node_count * static_cast<std::size_t>(avg_degree);

    // Naive linked-list representation (for reference implementation)
    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(total_edges);

    // Optimized contiguous representation
    args->flat_to.clear();
    args->flat_to.resize(total_edges);
    args->offsets.clear();
    args->offsets.resize(node_count + 1, 0);

    std::size_t edge_pos = 0;
    args->offsets[0] = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        const std::size_t base = edge_pos;

        // Generate neighbors directly into contiguous storage
        for (int k = 0; k < avg_degree; ++k) {
            args->flat_to[base + static_cast<std::size_t>(k)] = dist(gen);
        }

        // Build the naive linked list using the same values
        Edge* head = nullptr;
        for (int k = avg_degree; k-- > 0;) {
            Edge& e = args->edge_storage[base + static_cast<std::size_t>(k)];
            e.to = args->flat_to[base + static_cast<std::size_t>(k)];
            e.next = head;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
        args->offsets[u + 1] = edge_pos;
    }

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();
    args->graph.flat_to = args->flat_to.data();
    args->graph.offsets = args->offsets.data();
    args->graph.edge_count = total_edges;

    args->out = 0;
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

void stu_graph(std::uint64_t& out, const Graph& graph) {
    const int* p = graph.flat_to;
    const std::size_t m = graph.edge_count;

    if (p == nullptr || m == 0) {
        out = 0;
        return;
    }

    // Unrolled sequential scan over contiguous memory
    std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::uint64_t s4 = 0, s5 = 0, s6 = 0, s7 = 0;

    std::size_t i = 0;
    for (; i + 8 <= m; i += 8) {
        s0 += static_cast<std::uint64_t>(p[i]);
        s1 += static_cast<std::uint64_t>(p[i + 1]);
        s2 += static_cast<std::uint64_t>(p[i + 2]);
        s3 += static_cast<std::uint64_t>(p[i + 3]);
        s4 += static_cast<std::uint64_t>(p[i + 4]);
        s5 += static_cast<std::uint64_t>(p[i + 5]);
        s6 += static_cast<std::uint64_t>(p[i + 6]);
        s7 += static_cast<std::uint64_t>(p[i + 7]);
    }

    std::uint64_t sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);

    for (; i < m; ++i) {
        sum += static_cast<std::uint64_t>(p[i]);
    }

    out = sum;
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    stu_graph(args.out, args.graph);
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);
    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}