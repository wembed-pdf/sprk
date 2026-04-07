#ifndef SPRK_H
#define SPRK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SprkHandle SprkHandle;

/**
 * Create a SPRK-tree from a flat array of positions.
 * @param positions  Flat array of num_points * dim floats (row-major).
 * @param num_points Number of points.
 * @param dim        Dimensionality (must be in [2, 16]).
 * @return Handle to the tree, or NULL on error.
 */
SprkHandle* sprk_create(const float* positions, size_t num_points, size_t dim);

/** Destroy a tree and free its memory. */
void sprk_destroy(SprkHandle* handle);

/**
 * Query all points within radius, allocating the result array.
 * Caller must free with sprk_free_results().
 */
void sprk_query_radius(const SprkHandle* handle, const float* pos,
                       double radius, uint64_t** out_ids, size_t* out_count);

/** Free a result array from sprk_query_radius(). */
void sprk_free_results(uint64_t* ids, size_t count);

/** Return the dimensionality of the tree. */
size_t sprk_dim(const SprkHandle* handle);

/** Return the number of points in the tree. */
size_t sprk_point_count(const SprkHandle* handle);

#ifdef __cplusplus
}
#endif

#endif /* SPRK_H */
