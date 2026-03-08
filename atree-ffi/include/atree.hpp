#ifndef ATREE_HPP
#define ATREE_HPP

#include "atree.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace atree {

template <size_t D>
class ATree {
    static_assert(D >= 2 && D <= 16, "ATree dimension must be in [2, 16]");

public:
    /** Construct from a flat row-major array of num_points * D floats. */
    ATree(const float* positions, size_t num_points)
        : handle_(atree_create(positions, num_points, D)) {
        assert(handle_ && "atree_create failed");
    }

    /** Construct from a vector of floats (must be divisible by D). */
    explicit ATree(const std::vector<float>& positions)
        : ATree(positions.data(), positions.size() / D) {
        assert(positions.size() % D == 0);
    }

    ~ATree() {
        if (handle_) atree_destroy(handle_);
    }

    // Move-only
    ATree(ATree&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    ATree& operator=(ATree&& other) noexcept {
        if (this != &other) {
            if (handle_) atree_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ATree(const ATree&) = delete;
    ATree& operator=(const ATree&) = delete;

    /** Query all points within radius. Returns a vector of point IDs. */
    std::vector<uint64_t> query_radius(const float* pos, double radius) const {
        uint64_t* ids = nullptr;
        size_t count = 0;
        atree_query_radius_alloc(handle_, pos, radius, &ids, &count);
        std::vector<uint64_t> result(ids, ids + count);
        atree_free_results(ids, count);
        return result;
    }

    /**
     * Query into a caller-provided buffer. Returns the total number of results
     * (may exceed capacity, in which case only capacity results are written).
     */
    size_t query_radius(const float* pos, double radius,
                        uint64_t* out_ids, size_t capacity) const {
        return ::atree_query_radius(handle_, pos, radius, out_ids, capacity);
    }

    size_t dim() const { return D; }
    size_t point_count() const { return atree_point_count(handle_); }

private:
    ATreeHandle* handle_;
};

} // namespace atree

#endif /* ATREE_HPP */
