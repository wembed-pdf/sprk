use crate::dvec::DVec;

pub mod dvec;

const LEAFSIZE: usize = 150;
pub type NodeId = u64;

#[derive(Clone)]
pub struct ATree<const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub positions_sorted: Vec<DVec<D>>,
    pub node_ids: Vec<u64>,
    pub d_pos: Vec<f32>,
    layers: Vec<Layer>,
}

#[derive(Clone, Debug)]
struct Node {
    split: f32,
}

#[derive(Clone, Debug, Default)]
struct Snn {
    dpos_offset: usize,
    lut: Vec<usize>,
    min: f32,
    resolution: f32,
}

#[derive(Clone, Debug)]
enum Layer {
    Node(Node),
    Leaf(Snn),
}

fn children(index: usize) -> (usize, usize) {
    (index * 2 + 1, index * 2 + 2)
}

impl<const D: usize> ATree<D> {
    // --- Construction ---

    pub fn new(positions: Vec<DVec<D>>) -> Self {
        let mut atree = ATree {
            positions: positions.clone(),
            positions_sorted: Vec::new(),
            node_ids: Vec::new(),
            d_pos: Vec::new(),
            layers: vec![Layer::Node(Node { split: 0. }); positions.len()],
        };
        if !atree.positions.is_empty() {
            atree.update_positions(&positions, None);
        }
        atree
    }

    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        if self.positions.len() != positions.len() {
            self.positions = positions.to_vec();
        } else {
            for (old_pos, pos) in self.positions.iter_mut().zip(positions.iter()) {
                *old_pos = *pos;
            }
        }

        let mut node_ids: Vec<_> = (0..positions.len() as u64).collect();

        let num_leafs = node_ids.len().max(LEAFSIZE).ilog2() - LEAFSIZE.ilog2();
        let mut d_pos =
            vec![f32::INFINITY; node_ids.len() + (1usize << num_leafs as usize) * 2 * 4];
        let mut layers = std::mem::take(&mut self.layers);
        if layers.len() < node_ids.len() {
            layers = vec![Layer::Leaf(Snn::default()); node_ids.len()];
        }
        self.init_layers(
            node_ids.as_mut_slice(),
            d_pos.as_mut_slice(),
            &mut layers,
            0,
            0,
            0,
            0,
        );
        self.layers = layers;
        self.node_ids = node_ids;
        self.positions_sorted = self.node_ids.iter().map(|id| self.position(*id)).collect();
        for _ in 0..4 {
            d_pos.push(f32::INFINITY);
            self.positions_sorted.push(DVec::splat(f32::INFINITY));
        }
        self.d_pos = d_pos;
    }

    // --- Query ---

    pub fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        let radius_squared = radius.powi(2);
        self.query_recursive(
            pos,
            0,
            0,
            radius_squared as f32,
            radius_squared,
            DVec::zero(),
            results,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn query_recursive(
        &self,
        pos: DVec<D>,
        depth: usize,
        layer_id: usize,
        dim_radius_squared: f32,
        original_radius_squared: f64,
        mut distances: DVec<D>,
        results: &mut Vec<NodeId>,
    ) {
        let layer = &self.layers[layer_id];
        let own_pos = pos[depth];
        let new_depth = (depth + 1) % D;
        match layer {
            Layer::Node(node) => {
                let (left, right) = children(layer_id);
                let (own, other) = if own_pos < node.split {
                    (left, right)
                } else {
                    (right, left)
                };
                self.query_recursive(
                    pos,
                    new_depth,
                    own,
                    dim_radius_squared,
                    original_radius_squared,
                    distances,
                    results,
                );
                let mut reduced_radius = dim_radius_squared;
                let dist = own_pos - node.split;
                let d_2 = dist - distances[depth];
                let x = 2. * distances[depth] * d_2 + d_2.powi(2);
                distances[depth] = dist;
                reduced_radius -= x;
                if reduced_radius <= 0. {
                    return;
                }

                self.query_recursive(
                    pos,
                    new_depth,
                    other,
                    reduced_radius,
                    original_radius_squared,
                    distances,
                    results,
                );
            }
            Layer::Leaf(snn) => {
                let dim_diff_squared = distances[depth].powi(2);
                let radius_sqrt = (dim_radius_squared + dim_diff_squared).sqrt();
                let min = own_pos - radius_sqrt;
                let max = own_pos + radius_sqrt;
                let idx =
                    (((min - snn.min) * snn.resolution) as usize).min(snn.lut.len().max(1) - 1);
                if snn.lut.is_empty() {
                    return;
                }
                let min_i = snn.lut[idx];

                let mut i = min_i;
                loop {
                    results.reserve(4);
                    let p = self.d_pos[i + snn.dpos_offset];
                    let b = p > max;
                    for j in 0..2 {
                        let i = i + j;
                        let other_pos = self.positions_sorted[i];
                        if !b && pos.distance_squared(&other_pos) <= original_radius_squared as f32
                        {
                            results.push(self.node_ids[i]);
                        }
                    }
                    i += 2;
                    if b {
                        break;
                    }
                }
            }
        }
    }

    // --- Tree building internals ---

    fn init_layers(
        &self,
        nodes: &mut [NodeId],
        d_pos: &mut [f32],
        layers: &mut [Layer],
        depth: usize,
        layer_id: usize,
        offset: usize,
        dpos_offset: usize,
    ) {
        if nodes.len() <= LEAFSIZE {
            self.init_leaf(nodes, d_pos, layers, depth, layer_id, offset, dpos_offset);
        } else {
            self.init_node(nodes, d_pos, layers, depth, layer_id, offset, dpos_offset);
        }
    }

    fn init_leaf(
        &self,
        nodes: &mut [NodeId],
        d_pos: &mut [f32],
        layers: &mut [Layer],
        depth: usize,
        layer_id: usize,
        offset: usize,
        dpos_offset: usize,
    ) {
        nodes.sort_unstable_by_key(|i| self.sort_key(i, depth));

        for (d_pos, pos) in d_pos
            .iter_mut()
            .zip(nodes.iter().map(|id| self.position(*id)))
        {
            *d_pos = pos[depth];
        }

        let min = d_pos[0].floor();
        let slack = d_pos.len() - nodes.len();
        let max = d_pos.iter().rev().nth(slack).unwrap_or(&min).ceil();
        let resolution = 50. / (max - min);

        let mut lut = vec![];
        for i in 0..(((max - min) * resolution) as i32) {
            let pos_idx = d_pos
                .iter()
                .take_while(|&&x| x < ((i as f32 / resolution) + min))
                .count();
            lut.push(pos_idx + offset);
        }

        layers[layer_id] = Layer::Leaf(Snn {
            dpos_offset: dpos_offset - offset,
            lut,
            min,
            resolution,
        });
    }

    fn init_node(
        &self,
        nodes: &mut [NodeId],
        d_pos: &mut [f32],
        layers: &mut [Layer],
        depth: usize,
        layer_id: usize,
        offset: usize,
        dpos_offset: usize,
    ) {
        let median_idx = nodes.len() / 2;
        nodes.select_nth_unstable_by_key(median_idx, |i| self.sort_key(i, depth));

        // Move elements equal to pivot to the right side for strict partitioning
        let split = self.position(nodes[median_idx])[depth];
        let mut split_pos = median_idx;

        let mut i = 0;
        while i < split_pos {
            if self.position(nodes[i])[depth] == split {
                split_pos -= 1;
                nodes.swap(i, split_pos);
            } else {
                i += 1;
            }
        }

        let slack = d_pos.len() - nodes.len();
        let (a_ids, b_ids) = nodes.split_at_mut(split_pos);
        let (a_dpos, b_dpos) = d_pos.split_at_mut(split_pos + slack / 2);
        let (a_id, b_id) = children(layer_id);
        let depth = (depth + 1) % D;

        self.init_layers(a_ids, a_dpos, layers, depth, a_id, offset, dpos_offset);
        self.init_layers(
            b_ids,
            b_dpos,
            layers,
            depth,
            b_id,
            offset + split_pos,
            dpos_offset + split_pos + slack / 2,
        );

        layers[layer_id] = Layer::Node(Node { split });
    }

    // --- Helpers ---

    fn position(&self, index: NodeId) -> DVec<D> {
        self.positions[index as usize]
    }

    fn sort_key(&self, id: &NodeId, depth: usize) -> i32 {
        i32::from_ne_bytes(self.position(*id)[depth].to_ne_bytes())
    }
}
