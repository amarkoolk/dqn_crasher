"""
Dead-simple MTV-based collision classification based on the
`new_implementation.md` design document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sat_collision import SATResult, separating_axis_theorem

# Vertex indices (CCW from rear-right)
VERTEX_NAMES = {
    0: "rear-left corner",   # was rear-right
    1: "rear-right corner",  # was rear-left
    2: "front-right corner", # was front-left
    3: "front-left corner",  # was front-right
}

EDGE_NAMES = {
    0: "rear edge",    # unchanged
    1: "right edge",   # was left
    2: "front edge",   # unchanged
    3: "left edge",    # was right
}

# Which vertices form which edges
EDGE_VERTICES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
)


@dataclass(frozen=True)
class CollisionClassification:
    """Simple collision classification result."""

    contact_type: str  # "edge-edge", "vertex-edge", "vertex-vertex"
    collision_type: str  # "rear-end", "side-swipe", "head-on"
    ego_feature: str  # "front edge", "rear-right corner", etc.
    npc_feature: str
    ego_vertices: Tuple[int, ...]  # Vertex indices
    npc_vertices: Tuple[int, ...]
    ego_edges: Tuple[int, ...]  # Edge indices
    npc_edges: Tuple[int, ...]


@dataclass(frozen=True)
class MTVClassificationResult:
    """Bundle of SAT output alongside the derived classification."""

    sat_result: SATResult
    classification: CollisionClassification
    collision_axis: np.ndarray
    minimum_overlap: float

    @property
    def collision_detected(self) -> bool:
        return self.sat_result.intersecting

    @property
    def penetration_depth(self) -> float:
        return self.minimum_overlap


def detect_and_classify_collision(
    ego_polygon: np.ndarray,
    npc_polygon: np.ndarray,
    ego_velocity: Optional[np.ndarray] = None,
    npc_velocity: Optional[np.ndarray] = None,
    debug: bool = False,  # maintained for compatibility with prior callers
) -> Optional[MTVClassificationResult]:
    """
    Convenience wrapper that runs SAT and then classifies the collision if one occurred.
    """

    del debug

    if ego_velocity is None:
        ego_velocity = np.zeros(2, dtype=float)
    if npc_velocity is None:
        npc_velocity = np.zeros(2, dtype=float)

    sat_result = separating_axis_theorem(ego_polygon, npc_polygon, ego_velocity, npc_velocity)
    if not sat_result.intersecting:
        return None

    classification, axis, overlap = classify_collision(
        sat_result,
        ego_polygon,
        npc_polygon,
    )

    return MTVClassificationResult(
        sat_result=sat_result,
        classification=classification,
        collision_axis=axis,
        minimum_overlap=overlap,
    )


def classify_collision(
    sat_result: SATResult,
    ego_polygon: np.ndarray,
    npc_polygon: np.ndarray,
) -> Tuple[CollisionClassification, np.ndarray, float]:
    """
    Classify collision by finding minimum overlap axis from edge normals.

    SAT's MTV uses velocity-adjusted projections for swept collision detection.
    For classification, we need static projections (current geometry only).
    """

    if not sat_result.intersecting:
        raise ValueError("No collision to classify")

    # Get vertices (remove duplicate if 5th vertex = 1st vertex)
    ego_verts = _get_vertices(ego_polygon)
    npc_verts = _get_vertices(npc_polygon)

    # Compute all 8 edge normals and find minimum overlap axis (true collision axis)
    axis, min_overlap, ego_proj, npc_proj = _minimum_overlap_axis(ego_verts, npc_verts)

    # Find extremal vertices on the minimum overlap axis
    ego_vertices, npc_vertices = _find_extremal_vertices(ego_proj, npc_proj)

    # Find which edges (if any) are formed by these vertices
    ego_edges = _find_edges(ego_vertices)
    npc_edges = _find_edges(npc_vertices)

    # Classify based on vertex/edge geometry
    contact_type, collision_type, ego_feature, npc_feature = _classify(
        ego_vertices, npc_vertices, ego_edges, npc_edges
    )

    classification = CollisionClassification(
        contact_type=contact_type,
        collision_type=collision_type,
        ego_feature=ego_feature,
        npc_feature=npc_feature,
        ego_vertices=tuple(ego_vertices),
        npc_vertices=tuple(npc_vertices),
        ego_edges=tuple(ego_edges),
        npc_edges=tuple(npc_edges),
    )

    return classification, axis, min_overlap


def print_classification(result: CollisionClassification) -> None:
    """Print collision classification."""

    print(f"\n{'=' * 60}")
    print(f"COLLISION: {result.collision_type.upper()}")
    print(f"{'=' * 60}")
    print(f"Contact type: {result.contact_type}")
    print(f"EGO: {result.ego_feature}")
    print(f"NPC: {result.npc_feature}")
    print(f"EGO vertices: {result.ego_vertices}")
    print(f"NPC vertices: {result.npc_vertices}")
    print(f"EGO edges: {result.ego_edges}")
    print(f"NPC edges: {result.npc_edges}")
    print(f"{'=' * 60}\n")


def _get_vertices(polygon: np.ndarray) -> np.ndarray:
    """Extract 4 unique vertices."""

    verts = np.asarray(polygon, dtype=float)
    if verts.shape[0] > 4 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]
    if verts.shape[0] != 4:
        raise ValueError(f"Expected 4 vertices, got {verts.shape[0]}")
    return verts


def _minimum_overlap_axis(
    ego_verts: np.ndarray,
    npc_verts: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    min_overlap = float("inf")
    best_axis: Optional[np.ndarray] = None
    best_ego_proj: Optional[np.ndarray] = None
    best_npc_proj: Optional[np.ndarray] = None

    for verts in (ego_verts, npc_verts):
        for edge_id in range(4):
            axis = _edge_normal(verts, edge_id)
            if axis is None:
                continue

            ego_proj = ego_verts @ axis
            npc_proj = npc_verts @ axis
            ego_min, ego_max = float(ego_proj.min()), float(ego_proj.max())
            npc_min, npc_max = float(npc_proj.min()), float(npc_proj.max())

            overlap = min(ego_max, npc_max) - max(ego_min, npc_min)
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = axis
                best_ego_proj = ego_proj
                best_npc_proj = npc_proj

    if best_axis is None or best_ego_proj is None or best_npc_proj is None:
        raise ValueError("Could not find collision axis")

    return best_axis, min_overlap, best_ego_proj, best_npc_proj


def _edge_normal(verts: np.ndarray, edge_id: int) -> Optional[np.ndarray]:
    v1_idx, v2_idx = EDGE_VERTICES[edge_id]
    edge_vec = verts[v2_idx] - verts[v1_idx]
    normal = np.array([-edge_vec[1], edge_vec[0]], dtype=float)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return None
    return normal / norm


def _find_extremal_vertices(
    ego_proj: np.ndarray,
    npc_proj: np.ndarray,
) -> Tuple[List[int], List[int]]:
    """
    Find extremal vertices on projection axis.

    The vehicles overlap in the interval [max(min_ego, min_npc), min(max_ego, max_npc)].
    The extremal vertices are those at the boundaries of this overlap.
    """

    ego_min, ego_max = float(ego_proj.min()), float(ego_proj.max())
    npc_min, npc_max = float(npc_proj.min()), float(npc_proj.max())

    if ego_min < npc_min:
        # EGO's max side touches NPC's min side
        ego_target = ego_max
        npc_target = npc_min
    else:
        # EGO's min side touches NPC's max side
        ego_target = ego_min
        npc_target = npc_max

    # Find vertices at extremal positions (tight tolerance for zero penetration)
    ego_vertices = _vertices_at_value(ego_proj, ego_target, tolerance=0.001)
    npc_vertices = _vertices_at_value(npc_proj, npc_target, tolerance=0.001)

    return ego_vertices, npc_vertices


def _vertices_at_value(projections: np.ndarray, target: float, tolerance: float) -> List[int]:
    """Return vertex indices whose projection is at the target value."""

    distances = np.abs(projections - target)
    min_dist = float(distances.min())
    return [int(i) for i, d in enumerate(distances) if d <= min_dist + tolerance]


def _find_edges(vertex_ids: List[int]) -> List[int]:
    """Find which edges are formed by the given vertices."""

    if len(vertex_ids) < 2:
        return []

    v_set = set(vertex_ids)
    return [
        edge_id
        for edge_id, (v1, v2) in enumerate(EDGE_VERTICES)
        if v1 in v_set and v2 in v_set
    ]


def _classify(
    ego_v: List[int],
    npc_v: List[int],
    ego_e: List[int],
    npc_e: List[int],
) -> Tuple[str, str, str, str]:
    """Classify collision from vertex/edge indices."""

    # Edge-edge: both vehicles have an edge at contact
    if ego_e and npc_e:
        e_edge, n_edge = ego_e[0], npc_e[0]

        if e_edge == 2 and n_edge == 0:
            coll_type = "rear-end"
        elif e_edge == 0 and n_edge == 2:
            coll_type = "rear-ended"
        elif e_edge == 2 and n_edge == 2:
            coll_type = "head-on"
        else:
            coll_type = "side-swipe"

        return "edge-edge", coll_type, EDGE_NAMES[e_edge], EDGE_NAMES[n_edge]

    # Vertex-edge: NPC vertex hits EGO edge
    if ego_e and npc_v:
        e_edge = ego_e[0]
        n_vertex = npc_v[0]

        if e_edge == 2:
            coll_type = "rear-end"
        elif e_edge == 0:
            coll_type = "rear-ended"
        else:
            coll_type = "side-swipe"

        return "vertex-edge", coll_type, EDGE_NAMES[e_edge], VERTEX_NAMES[n_vertex]

    # Vertex-edge: EGO vertex hits NPC edge
    if npc_e and ego_v:
        n_edge = npc_e[0]
        e_vertex = ego_v[0]

        if n_edge == 0:
            coll_type = "rear-end"
        elif n_edge == 2:
            coll_type = "rear-ended"
        else:
            coll_type = "side-swipe"

        return "vertex-edge", coll_type, VERTEX_NAMES[e_vertex], EDGE_NAMES[n_edge]

    # Vertex-vertex: single vertices touching
    if ego_v and npc_v:
        e_v, n_v = ego_v[0], npc_v[0]

        e_front = e_v in (2, 3)
        n_front = n_v in (2, 3)
        e_left = e_v in (0, 3)
        n_left = n_v in (0, 3)

        if e_left != n_left:
            coll_type = "side-swipe"
        elif e_front and not n_front:
            coll_type = "rear-end"
        elif not e_front and n_front:
            coll_type = "rear-ended"
        elif e_front and n_front:
            coll_type = "head-on"
        else:
            coll_type = "angled"

        return "vertex-vertex", coll_type, VERTEX_NAMES[e_v], VERTEX_NAMES[n_v]

    # Fallback
    return "complex", "angled", f"{len(ego_v)} verts", f"{len(npc_v)} verts"


__all__ = [
    "CollisionClassification",
    "MTVClassificationResult",
    "classify_collision",
    "detect_and_classify_collision",
    "print_classification",
]
