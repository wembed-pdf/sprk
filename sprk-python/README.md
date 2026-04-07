# sprk-python

Python bindings for [SPRK-tree](../README.md), a high-performance spatial index for radius queries.

## Installation

Requires Python >= 3.8 and a Rust toolchain.

```bash
pip install maturin
maturin develop --release
```

## Usage

```python
import numpy as np
from sprk import Sprk

# Build a tree from an (N, D) float32 array
positions = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [5.0, 5.0],
], dtype=np.float32)

tree = Sprk(positions)

# Radius query — returns an array of point indices
query_point = np.array([0.5, 0.5], dtype=np.float32)
indices = tree.query_radius(query_point, radius=1.5)
print(indices)  # [0, 1, 2]

# Properties
print(tree.dim)          # 2
print(tree.point_count)  # 4
```

## Dimensionality

The dimension is inferred from the input array's second axis. Dimensions 2-16 use compile-time-specialized code paths with SIMD acceleration. Higher dimensions fall back to a dynamic implementation.

## API

### `Sprk(positions)`

Construct a tree from a 2D NumPy array of shape `(num_points, dim)` with dtype `float32`.

### `tree.query_radius(pos, radius) -> np.ndarray[u64]`

Return the indices of all points within `radius` of `pos`. `pos` must be a 1D `float32` array matching the tree's dimensionality.

### `tree.dim -> int`

The dimensionality of the tree.

### `tree.point_count -> int`

The number of points in the tree.

## License

MIT
