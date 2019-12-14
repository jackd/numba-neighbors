from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba_neighbors.kd_tree import KDTree

N = 1024
query_r = 0.2
out_size = 512
k_query = 256
leaf_size = 64

coords = np.random.normal(size=(N, 3)).astype(np.float32)
coords /= np.linalg.norm(coords, axis=-1)[:, np.newaxis]

r2 = query_r**2
tree = KDTree(coords, leaf_size=leaf_size)
sample_result, query_result = tree.rejection_ifp_sample_query(
    r2, r2, tree.get_node_indices(), out_size, k_query)
# sample_result, query_result = tree.ifp_sample_query(r2, tree.get_node_indices(),
#                                                     out_size, k_query)

print(query_result.dists[0, :query_result.counts[0]])
import trimesh
red = [[255, 0, 0]]
green = [[0, 255, 0]]
blue = [[0, 0, 255]]

sampled_coords = coords[sample_result.indices]

scene = trimesh.scene.Scene()
pc = trimesh.PointCloud(coords, colors=np.full((N, 3), 255, dtype=np.uint8))
scene = pc.scene()
scene.add_geometry(
    trimesh.PointCloud(sampled_coords, colors=np.tile(blue, (out_size, 1))))
rl = query_result.counts[0]
scene.add_geometry(
    trimesh.PointCloud(coords[query_result.indices[0, :rl]],
                       colors=np.tile(green, (rl, 1))))
scene.add_geometry(trimesh.PointCloud(sampled_coords[:1], colors=red))
# scene.add_geometry(
#     trimesh.primitives.Sphere(radius=radius, center=sampled_coords[0]))
# scene.add_geometry(trimesh.primitives.Sphere(radius=1.1, center=[0, 0, 0]))
scene.show(background=np.zeros((3,), dtype=np.uint8))
