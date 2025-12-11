from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

Vector = np.ndarray


@dataclass(frozen=True)
class SATResult:
    """Outcome of a SAT collision test between two convex polygons."""

    intersecting: bool
    will_intersect: bool
    minimum_translation_vector: Optional[np.ndarray]


def project_polygon(polygon: Vector, axis: Vector) -> tuple[float, float]:
    """
    Project polygon vertices onto an axis and return the [min, max] interval.
    """
    projections = polygon @ axis
    return float(np.min(projections)), float(np.max(projections))


def interval_distance(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    """
    Distance between two intervals. Negative values mean the intervals overlap.
    """
    return min_b - max_a if min_a < min_b else min_a - max_b


def separating_axis_theorem(
    polygon_a: Vector,
    polygon_b: Vector,
    displacement_a: Vector,
    displacement_b: Vector,
) -> SATResult:
    """
    Separating Axis Theorem for convex polygons returning MTV information.

    Parameters
    ----------
    polygon_a, polygon_b:
        Arrays of shape (N, 2) containing polygon vertices. The first vertex
        should be repeated at the end so that ``zip(polygon, polygon[1:])`` lists
        every edge.
    displacement_a, displacement_b:
        Linear displacements applied to each polygon for the time-step being
        tested. Passing the vehicles' velocity * dt reproduces highway-env's
        swept collision test.
    """
    intersecting = True
    will_intersect = True
    min_distance = np.inf
    translation_axis: Optional[np.ndarray] = None

    for polygon in (polygon_a, polygon_b):
        for p1, p2 in zip(polygon, polygon[1:]):
            edge = np.asarray(p2) - np.asarray(p1)
            normal = np.array([-edge[1], edge[0]], dtype=float)
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal /= norm

            min_a, max_a = project_polygon(polygon_a, normal)
            min_b, max_b = project_polygon(polygon_b, normal)

            if interval_distance(min_a, max_a, min_b, max_b) > 0:
                intersecting = False

            velocity_projection = float(normal.dot(displacement_a - displacement_b))
            if velocity_projection < 0:
                min_a += velocity_projection
            else:
                max_a += velocity_projection

            distance = interval_distance(min_a, max_a, min_b, max_b)
            if distance > 0:
                will_intersect = False
            if not intersecting and not will_intersect:
                break

            abs_distance = abs(distance)
            if abs_distance < min_distance:
                min_distance = abs_distance
                center_delta = polygon_a[:-1].mean(axis=0) - polygon_b[:-1].mean(axis=0)
                translation_axis = normal if center_delta.dot(normal) > 0 else -normal
        if not intersecting and not will_intersect:
            break

    translation = (
        min_distance * translation_axis if will_intersect and translation_axis is not None else None
    )
    return SATResult(
        intersecting=intersecting,
        will_intersect=will_intersect,
        minimum_translation_vector=translation,
    )

