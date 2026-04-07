# sprk-ffi

C/C++ bindings for [SPRK-tree](../README.md), a high-performance spatial index for radius queries.

## Building

```bash
cargo build --release -p sprk-ffi
```

The build produces both a shared library (`libsprk_ffi.so` / `.dylib` / `.dll`) and a static library (`libsprk_ffi.a`).

Headers are in the [`include/`](include/) directory:
- `sprk.h` — C API
- `sprk.hpp` — C++ RAII wrapper (header-only, includes `sprk.h`)

## C API

```c
#include "sprk.h"

float positions[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 5.0f, 5.0f};

// Create a 2D tree with 4 points
SprkHandle* tree = sprk_create(positions, 4, 2);

// Query all points within radius 1.5 of (0.5, 0.5)
float query[] = {0.5f, 0.5f};
uint64_t* ids;
size_t count;
sprk_query_radius(tree, query, 1.5, &ids, &count);

// Use results...
for (size_t i = 0; i < count; i++) {
    printf("found point %llu\n", ids[i]);
}

sprk_free_results(ids, count);
sprk_destroy(tree);
```

### Functions

| Function | Description |
|----------|-------------|
| `sprk_create(positions, num_points, dim)` | Build a tree from flat row-major `float` data. Returns `NULL` on error. |
| `sprk_destroy(handle)` | Free a tree handle. |
| `sprk_query_radius(handle, pos, radius, &ids, &count)` | Query and allocate results. Free with `sprk_free_results`. |
| `sprk_free_results(ids, count)` | Free a result array from `sprk_query_radius`. |
| `sprk_dim(handle)` | Return the tree's dimensionality. |
| `sprk_point_count(handle)` | Return the number of points. |

## C++ API

The header-only C++ wrapper provides a type-safe RAII interface with the dimension as a template parameter:

```cpp
#include "sprk.hpp"

float positions[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 5.0f, 5.0f};
sprk::Sprk<2> tree(positions, 4);

float query[] = {0.5f, 0.5f};
auto results = tree.query_radius(query, 1.5);
// results is std::vector<uint64_t>
```

The C++ wrapper is move-only and automatically destroys the tree when it goes out of scope.

## Dimensionality

Dimensions 2-16 (plus 18, 20, 22, 24) use monomorphized code paths with SIMD acceleration. Other dimensions use a dynamic fallback.

## License

MIT
