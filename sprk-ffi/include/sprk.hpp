#ifndef SPRK_HPP
#define SPRK_HPP

#include "sprk.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace sprk {

template <size_t D>
class Sprk {
    static_assert(D >= 2 && D <= 16, "Sprk dimension must be in [2, 16]");

public:
    /** Construct from a flat row-major array of num_points * D floats. */
    Sprk(const float* positions, size_t num_points)
        : handle_(sprk_create(positions, num_points, D)) {
        assert(handle_ && "sprk_create failed");
    }

    /** Construct from a vector of floats (must be divisible by D). */
    explicit Sprk(const std::vector<float>& positions)
        : Sprk(positions.data(), positions.size() / D) {
        assert(positions.size() % D == 0);
    }

    ~Sprk() {
        if (handle_) sprk_destroy(handle_);
    }

    // Move-only
    Sprk(Sprk&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Sprk& operator=(Sprk&& other) noexcept {
        if (this != &other) {
            if (handle_) sprk_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    Sprk(const Sprk&) = delete;
    Sprk& operator=(const Sprk&) = delete;

    /** Query all points within radius. Returns a vector of point IDs. */
    std::vector<uint64_t> query_radius(const float* pos, double radius) const {
        uint64_t* ids = nullptr;
        size_t count = 0;
        sprk_query_radius(handle_, pos, radius, &ids, &count);
        std::vector<uint64_t> result(ids, ids + count);
        sprk_free_results(ids, count);
        return result;
    }

    size_t dim() const { return D; }
    size_t point_count() const { return sprk_point_count(handle_); }

private:
    SprkHandle* handle_;
};

} // namespace sprk

#endif /* SPRK_HPP */
