import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from panorama.alignment import Edge, align_spherical, intrinsics, spanning_tree
from panorama.geometry import project


def test_disconnected_graph_does_not_silently_drop_images():
    edge = Edge(0, 1, np.eye(3), np.zeros((20, 2)), np.zeros((20, 2)), 0.1)
    with pytest.raises(ValueError, match="未连通"):
        spanning_tree(3, [edge])


def test_rotation_bundle_recovers_known_camera_geometry():
    images = [np.zeros((480, 640, 3), np.uint8) for _ in range(3)]
    k = intrinsics(images[0].shape, 0.8)
    rotations = [
        Rotation.from_euler("yx", [yaw, pitch], degrees=True).as_matrix()
        for yaw, pitch in [(-20, 2), (0, 0), (22, -2)]
    ]
    rng = np.random.default_rng(4)
    points = rng.uniform([160, 100], [480, 380], (60, 2))
    edges = []
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        h = k @ rotations[j].T @ rotations[i] @ np.linalg.inv(k)
        edges.append(Edge(i, j, h / h[2, 2], points, project(h, points), 0.01))
    root, tree = spanning_tree(3, edges)
    found, ratio, info = align_spherical(images, edges, root, tree)
    assert abs(ratio - 0.8) < 0.002
    estimated_h = k @ found[2].T @ found[0] @ np.linalg.inv(k)
    np.testing.assert_allclose(project(estimated_h, points), project(edges[-1].h, points), atol=0.1)
    assert info["bundle_final_ray_rmse_px"] < 0.01
