"""Subdivision surface MVP (Catmull-Clark on quad meshes).

This script implements Catmull-Clark subdivision from scratch using NumPy,
then runs two refinement iterations on a cube mesh and prints topology and
geometry statistics.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class QuadMesh:
    """Simple quad mesh container."""

    vertices: np.ndarray  # shape: (V, 3)
    faces: List[Tuple[int, int, int, int]]


@dataclass
class MeshTopology:
    """Adjacency and edge mapping for a mesh."""

    edge_vertices: List[Tuple[int, int]]
    edge_faces: List[List[int]]
    edge_to_index: Dict[Tuple[int, int], int]
    vertex_faces: List[List[int]]
    vertex_edges: List[List[int]]


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def build_cube_mesh() -> QuadMesh:
    """Build a closed quad cube centered at origin with side length 2."""
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],  # 0
            [1.0, -1.0, -1.0],  # 1
            [-1.0, 1.0, -1.0],  # 2
            [1.0, 1.0, -1.0],  # 3
            [-1.0, -1.0, 1.0],  # 4
            [1.0, -1.0, 1.0],  # 5
            [-1.0, 1.0, 1.0],  # 6
            [1.0, 1.0, 1.0],  # 7
        ],
        dtype=float,
    )

    # All faces are quads; orientation consistency is not required for this demo.
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    return QuadMesh(vertices=vertices, faces=faces)


def build_topology(mesh: QuadMesh) -> MeshTopology:
    """Build edge and vertex adjacency structures."""
    v_count = mesh.vertices.shape[0]
    edge_to_index: Dict[Tuple[int, int], int] = {}
    edge_vertices: List[Tuple[int, int]] = []
    edge_faces: List[List[int]] = []
    vertex_faces: List[List[int]] = [[] for _ in range(v_count)]
    vertex_edges: List[List[int]] = [[] for _ in range(v_count)]

    for f_idx, face in enumerate(mesh.faces):
        n = len(face)
        if n != 4:
            raise ValueError("This MVP expects quad faces only")

        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            k = _edge_key(a, b)
            if k not in edge_to_index:
                e_idx = len(edge_vertices)
                edge_to_index[k] = e_idx
                edge_vertices.append(k)
                edge_faces.append([])
            e_idx = edge_to_index[k]
            edge_faces[e_idx].append(f_idx)
            vertex_edges[a].append(e_idx)
            vertex_edges[b].append(e_idx)

        for vid in face:
            vertex_faces[vid].append(f_idx)

    # Deduplicate adjacency lists while preserving sorted deterministic order.
    vertex_faces = [sorted(set(ids)) for ids in vertex_faces]
    vertex_edges = [sorted(set(ids)) for ids in vertex_edges]

    return MeshTopology(
        edge_vertices=edge_vertices,
        edge_faces=edge_faces,
        edge_to_index=edge_to_index,
        vertex_faces=vertex_faces,
        vertex_edges=vertex_edges,
    )


def catmull_clark_subdivide(mesh: QuadMesh) -> QuadMesh:
    """Perform one Catmull-Clark subdivision step on a quad mesh."""
    verts = mesh.vertices
    topo = build_topology(mesh)
    v_count = verts.shape[0]
    e_count = len(topo.edge_vertices)
    f_count = len(mesh.faces)

    # 1) Face points
    face_points = np.empty((f_count, 3), dtype=float)
    for f_idx, face in enumerate(mesh.faces):
        face_points[f_idx] = verts[np.array(face, dtype=int)].mean(axis=0)

    # 2) Edge points
    edge_points = np.empty((e_count, 3), dtype=float)
    for e_idx, (a, b) in enumerate(topo.edge_vertices):
        adjacent_faces = topo.edge_faces[e_idx]
        if len(adjacent_faces) == 2:
            f0, f1 = adjacent_faces
            edge_points[e_idx] = (verts[a] + verts[b] + face_points[f0] + face_points[f1]) * 0.25
        elif len(adjacent_faces) == 1:
            # Boundary-safe fallback (not used by the cube demo).
            edge_points[e_idx] = (verts[a] + verts[b]) * 0.5
        else:
            # Non-manifold fallback: average endpoints + all adjacent face points.
            accum = verts[a] + verts[b]
            for fid in adjacent_faces:
                accum += face_points[fid]
            edge_points[e_idx] = accum / float(2 + len(adjacent_faces))

    # 3) Updated old vertices
    updated_vertices = np.empty_like(verts)
    for vid in range(v_count):
        touching_faces = topo.vertex_faces[vid]
        touching_edges = topo.vertex_edges[vid]
        n = len(touching_faces)

        if n == 0:
            updated_vertices[vid] = verts[vid]
            continue

        is_boundary = any(len(topo.edge_faces[eid]) == 1 for eid in touching_edges)
        if is_boundary:
            # Boundary rule fallback: keep corners stable and smooth with boundary neighbors.
            boundary_neighbors: List[int] = []
            for eid in touching_edges:
                if len(topo.edge_faces[eid]) != 1:
                    continue
                a, b = topo.edge_vertices[eid]
                boundary_neighbors.append(b if a == vid else a)
            uniq_neighbors = sorted(set(boundary_neighbors))
            if len(uniq_neighbors) >= 2:
                updated_vertices[vid] = (
                    0.75 * verts[vid]
                    + 0.125 * verts[uniq_neighbors[0]]
                    + 0.125 * verts[uniq_neighbors[1]]
                )
            else:
                updated_vertices[vid] = verts[vid]
            continue

        f_avg = face_points[np.array(touching_faces, dtype=int)].mean(axis=0)

        edge_midpoints = []
        for eid in touching_edges:
            a, b = topo.edge_vertices[eid]
            edge_midpoints.append((verts[a] + verts[b]) * 0.5)
        r_avg = np.mean(np.array(edge_midpoints, dtype=float), axis=0)

        updated_vertices[vid] = (f_avg + 2.0 * r_avg + (n - 3.0) * verts[vid]) / float(n)

    # 4) Assemble new vertex array: [updated old vertices | edge points | face points]
    new_vertices = np.vstack([updated_vertices, edge_points, face_points])
    edge_offset = v_count
    face_offset = v_count + e_count

    # 5) Build new quads
    new_faces: List[Tuple[int, int, int, int]] = []
    for f_idx, face in enumerate(mesh.faces):
        n = len(face)
        for i in range(n):
            v_cur = face[i]
            v_next = face[(i + 1) % n]
            v_prev = face[(i - 1) % n]

            e_cur = topo.edge_to_index[_edge_key(v_cur, v_next)]
            e_prev = topo.edge_to_index[_edge_key(v_prev, v_cur)]

            new_faces.append(
                (
                    v_cur,
                    edge_offset + e_cur,
                    face_offset + f_idx,
                    edge_offset + e_prev,
                )
            )

    return QuadMesh(vertices=new_vertices, faces=new_faces)


def unique_edges(mesh: QuadMesh) -> List[Tuple[int, int]]:
    """Collect unique undirected edges from face list."""
    edges = set()
    for face in mesh.faces:
        for i in range(4):
            edges.add(_edge_key(face[i], face[(i + 1) % 4]))
    return sorted(edges)


def edge_length_stats(mesh: QuadMesh) -> Dict[str, float]:
    """Compute min/mean/max/std edge lengths."""
    edges = unique_edges(mesh)
    lengths = []
    for a, b in edges:
        lengths.append(float(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])))
    arr = np.array(lengths, dtype=float)
    return {
        "edge_count": float(arr.size),
        "edge_len_min": float(arr.min()),
        "edge_len_mean": float(arr.mean()),
        "edge_len_max": float(arr.max()),
        "edge_len_std": float(arr.std()),
    }


def radius_stats(mesh: QuadMesh) -> Dict[str, float]:
    """Distance of vertices to mesh centroid, used as smoothness proxy."""
    c = mesh.vertices.mean(axis=0)
    r = np.linalg.norm(mesh.vertices - c[None, :], axis=1)
    return {
        "radius_min": float(r.min()),
        "radius_mean": float(r.mean()),
        "radius_max": float(r.max()),
        "radius_std": float(r.std()),
    }


def euler_characteristic(mesh: QuadMesh) -> int:
    """Compute Euler characteristic V - E + F."""
    v = mesh.vertices.shape[0]
    e = len(unique_edges(mesh))
    f = len(mesh.faces)
    return int(v - e + f)


def predicted_counts_for_closed_quad(mesh: QuadMesh) -> Dict[str, int]:
    """Catmull-Clark count formulas for a closed quad manifold."""
    v = mesh.vertices.shape[0]
    e = len(unique_edges(mesh))
    f = len(mesh.faces)
    return {
        "next_v": v + e + f,
        "next_f": 4 * f,
    }


def print_iteration_report(name: str, mesh: QuadMesh) -> None:
    """Print topology and geometry summary."""
    v = mesh.vertices.shape[0]
    e = len(unique_edges(mesh))
    f = len(mesh.faces)
    chi = euler_characteristic(mesh)

    edge_stats = edge_length_stats(mesh)
    rad_stats = radius_stats(mesh)

    print(f"[{name}]")
    print(f"V={v}, E={e}, F={f}, Euler(V-E+F)={chi}")
    print(
        "EdgeLen min/mean/max/std = "
        f"{edge_stats['edge_len_min']:.6f} / {edge_stats['edge_len_mean']:.6f} "
        f"/ {edge_stats['edge_len_max']:.6f} / {edge_stats['edge_len_std']:.6f}"
    )
    print(
        "Radius  min/mean/max/std = "
        f"{rad_stats['radius_min']:.6f} / {rad_stats['radius_mean']:.6f} "
        f"/ {rad_stats['radius_max']:.6f} / {rad_stats['radius_std']:.6f}"
    )


def main() -> None:
    mesh = build_cube_mesh()
    print("Catmull-Clark Subdivision Surface MVP")
    print("Input: closed quad cube mesh, 2 subdivision iterations")
    print()

    print_iteration_report("iter=0 (original)", mesh)
    print()

    for i in range(1, 3):
        predicted = predicted_counts_for_closed_quad(mesh)
        mesh = catmull_clark_subdivide(mesh)

        print_iteration_report(f"iter={i}", mesh)
        print(
            "Predicted counts check "
            f"(V' = V+E+F, F' = 4F): "
            f"V_expected={predicted['next_v']}, F_expected={predicted['next_f']}, "
            f"match={mesh.vertices.shape[0] == predicted['next_v'] and len(mesh.faces) == predicted['next_f']}"
        )
        print()

    sample = mesh.vertices[:8]
    print("First 8 refined vertices (after iter=2):")
    for idx, p in enumerate(sample):
        print(f"  v[{idx}] = ({p[0]: .6f}, {p[1]: .6f}, {p[2]: .6f})")


if __name__ == "__main__":
    main()
