#ifndef ATREE_H
#define ATREE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ATreeHandle ATreeHandle;

/**
 * Create an ATree from a flat array of positions.
 * @param positions  Flat array of num_points * dim floats (row-major).
 * @param num_points Number of points.
 * @param dim        Dimensionality (must be in [2, 16]).
 * @return Handle to the tree, or NULL on error.
 */
ATreeHandle* atree_create(const float* positions, size_t num_points, size_t dim);

/** Destroy a tree and free its memory. */
void atree_destroy(ATreeHandle* handle);

/**
 * Query all points within radius, writing into a caller-provided buffer.
 * @param out_ids      Buffer for result IDs.
 * @param out_capacity Size of the buffer.
 * @return Number of results found (may exceed out_capacity).
 */
size_t atree_query_radius(const ATreeHandle* handle, const float* pos,
                          double radius, uint64_t* out_ids, size_t out_capacity);

/**
 * Query all points within radius, allocating the result array.
 * Caller must free with atree_free_results().
 */
void atree_query_radius_alloc(const ATreeHandle* handle, const float* pos,
                              double radius, uint64_t** out_ids, size_t* out_count);

/** Free a result array from atree_query_radius_alloc(). */
void atree_free_results(uint64_t* ids, size_t count);

/** Return the dimensionality of the tree. */
size_t atree_dim(const ATreeHandle* handle);

/** Return the number of points in the tree. */
size_t atree_point_count(const ATreeHandle* handle);

#ifdef __cplusplus
}
#endif

#endif /* ATREE_H */
