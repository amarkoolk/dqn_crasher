"""
Unified collision detection interface for comparing different methods.

This module provides:
- Abstract base class for collision detection algorithms
- Unified result data structures
- Common utilities for all methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time

import numpy as np


@dataclass(frozen=True)
class ContactPoint:
    """Represents a single contact point between two polygons."""
    vertex_id: int  # Which vertex (0-3 for rectangles)
    edge_id: int    # Which edge it contacts (0-3 for rectangles)
    position: np.ndarray  # World position of contact
    normal: np.ndarray    # Contact normal (pointing from B to A)
    penetration: float    # Penetration depth at this point


@dataclass(frozen=True)
class CollisionResult:
    """
    Unified collision detection result.
    
    All methods must provide at minimum:
    - method: name of the detection method
    - intersecting: boolean collision flag
    - computation_time_ns: time taken in nanoseconds
    
    Optional rich data (method-dependent):
    - collision_normal: direction to separate polygons
    - penetration_depth: how deep the penetration is
    - contact_points: detailed contact geometry
    - closest_points: closest points on each polygon (for distance queries)
    - broad_phase_rejected: whether broad phase culled the test
    """
    method: str
    intersecting: bool
    computation_time_ns: int
    
    # Common collision data
    collision_normal: Optional[np.ndarray] = None
    penetration_depth: Optional[float] = None
    
    # Method-specific rich data
    contact_points: Optional[List[ContactPoint]] = None
    closest_points: Optional[Tuple[np.ndarray, np.ndarray]] = None
    broad_phase_rejected: Optional[bool] = None
    
    # Additional metadata
    will_intersect: bool = False  # For swept/continuous collision detection
    metadata: dict = field(default_factory=dict)  # Method-specific extras
    
    def get_mtv(self) -> Optional[np.ndarray]:
        """
        Get the Minimum Translation Vector (MTV).
        MTV = normal * depth, representing smallest displacement to separate.
        """
        if self.collision_normal is not None and self.penetration_depth is not None:
            return self.collision_normal * self.penetration_depth
        return None
    
    def data_richness_score(self) -> int:
        """
        Score representing how much data this result provides.
        Higher = more information available for analysis.
        """
        score = 0
        if self.collision_normal is not None:
            score += 1
        if self.penetration_depth is not None:
            score += 1
        if self.contact_points is not None:
            score += len(self.contact_points)
        if self.closest_points is not None:
            score += 1
        if self.metadata:
            score += len(self.metadata)
        return score


class CollisionDetector(ABC):
    """
    Abstract base class for collision detection algorithms.
    
    All detectors must implement the detect() method which takes two
    polygons and their velocities, returning a CollisionResult.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._call_count = 0
        self._total_time_ns = 0
    
    @abstractmethod
    def detect(
        self,
        polygon_a: np.ndarray,
        polygon_b: np.ndarray,
        displacement_a: np.ndarray,
        displacement_b: np.ndarray
    ) -> CollisionResult:
        """
        Detect collision between two convex polygons.
        
        Parameters
        ----------
        polygon_a, polygon_b : np.ndarray
            Arrays of shape (N, 2) containing polygon vertices.
            The first vertex should be repeated at the end for edge iteration.
        displacement_a, displacement_b : np.ndarray
            Linear displacement vectors for swept collision detection.
            Pass velocity * dt for continuous collision detection.
        
        Returns
        -------
        CollisionResult
            Unified result structure with collision data.
        """
        pass
    
    def detect_timed(
        self,
        polygon_a: np.ndarray,
        polygon_b: np.ndarray,
        displacement_a: np.ndarray,
        displacement_b: np.ndarray
    ) -> CollisionResult:
        """
        Wrapper that automatically times the detection.
        Use this for consistent benchmarking.
        """
        start = time.perf_counter_ns()
        result = self.detect(polygon_a, polygon_b, displacement_a, displacement_b)
        elapsed = time.perf_counter_ns() - start
        
        self._call_count += 1
        self._total_time_ns += elapsed
        
        # Replace the computation time with accurate measurement
        result = CollisionResult(
            method=result.method,
            intersecting=result.intersecting,
            computation_time_ns=elapsed,
            collision_normal=result.collision_normal,
            penetration_depth=result.penetration_depth,
            contact_points=result.contact_points,
            closest_points=result.closest_points,
            broad_phase_rejected=result.broad_phase_rejected,
            will_intersect=result.will_intersect,
            metadata=result.metadata
        )
        
        return result
    
    def get_stats(self) -> dict:
        """Get performance statistics for this detector."""
        return {
            'name': self.name,
            'call_count': self._call_count,
            'total_time_ns': self._total_time_ns,
            'avg_time_ns': self._total_time_ns / max(1, self._call_count)
        }
    
    def reset_stats(self):
        """Reset performance counters."""
        self._call_count = 0
        self._total_time_ns = 0


def compute_polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of a polygon.
    
    Parameters
    ----------
    polygon : np.ndarray
        Array of shape (N, 2) containing polygon vertices.
    
    Returns
    -------
    np.ndarray
        Centroid position [x, y].
    """
    # Exclude the repeated last vertex
    return polygon[:-1].mean(axis=0)


def compute_aabb(polygon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box for a polygon.
    
    Parameters
    ----------
    polygon : np.ndarray
        Array of shape (N, 2) containing polygon vertices.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (min_point, max_point) where each is [x, y].
    """
    min_point = polygon[:-1].min(axis=0)
    max_point = polygon[:-1].max(axis=0)
    return min_point, max_point


def aabb_overlap(min_a: np.ndarray, max_a: np.ndarray,
                 min_b: np.ndarray, max_b: np.ndarray) -> bool:
    """
    Check if two axis-aligned bounding boxes overlap.
    
    This is a fast O(1) broad-phase collision test.
    
    Parameters
    ----------
    min_a, max_a : np.ndarray
        Min and max corners of AABB for polygon A.
    min_b, max_b : np.ndarray
        Min and max corners of AABB for polygon B.
    
    Returns
    -------
    bool
        True if the AABBs overlap.
    """
    # No overlap if separated along any axis
    if max_a[0] < min_b[0] or max_b[0] < min_a[0]:
        return False
    if max_a[1] < min_b[1] or max_b[1] < min_a[1]:
        return False
    return True


def point_to_line_segment_distance(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray
) -> Tuple[float, np.ndarray, float]:
    """
    Compute the shortest distance from a point to a line segment.
    
    Parameters
    ----------
    point : np.ndarray
        Point coordinates [x, y].
    seg_start, seg_end : np.ndarray
        Line segment endpoints [x, y].
    
    Returns
    -------
    Tuple[float, np.ndarray, float]
        (distance, closest_point, t_param)
        - distance: shortest distance from point to segment
        - closest_point: point on segment closest to the input point
        - t_param: parametric position along segment [0, 1]
    """
    segment = seg_end - seg_start
    segment_length_sq = np.dot(segment, segment)
    
    if segment_length_sq < 1e-10:
        # Degenerate segment (point)
        closest_point = seg_start
        distance = np.linalg.norm(point - seg_start)
        return distance, closest_point, 0.0
    
    # Project point onto the infinite line
    to_point = point - seg_start
    t = np.dot(to_point, segment) / segment_length_sq
    
    # Clamp to segment
    t = max(0.0, min(1.0, t))
    
    closest_point = seg_start + t * segment
    distance = np.linalg.norm(point - closest_point)
    
    return distance, closest_point, t


def classify_collision_from_normal(
    normal: np.ndarray,
    ego_heading: float
) -> Tuple[str, float, float, float]:
    """
    Classify collision type based on collision normal and ego heading.
    
    This is the same classification logic used in manual_control_universal.py,
    but extracted for use by all detection methods.
    
    Parameters
    ----------
    normal : np.ndarray
        Collision normal vector (MTV direction or contact normal).
    ego_heading : float
        Ego vehicle's heading angle in radians.
    
    Returns
    -------
    Tuple[str, float, float, float]
        (classification, impact_angle_deg, longitudinal_component, lateral_component)
    """
    if np.linalg.norm(normal) < 1e-6:
        return None, float('nan'), 0.0, 0.0
    
    # Ego's reference frame
    ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    ego_left = np.array([-np.sin(ego_heading), np.cos(ego_heading)])
    
    # Project normal onto ego frame
    longitudinal = float(np.dot(normal, ego_forward))
    lateral = float(np.dot(normal, ego_left))
    
    # Impact angle in ego frame
    impact_angle_deg = float(np.degrees(np.arctan2(lateral, longitudinal)))
    
    # Classify based on dominant component
    if abs(longitudinal) >= abs(lateral):
        label = (
            "rear impact (other hit ego from behind)"
            if longitudinal > 0
            else "front impact (ego hit vehicle ahead)"
        )
    else:
        label = (
            "right-side impact (other on ego's right)"
            if lateral > 0
            else "left-side impact (other on ego's left)"
        )
    
    return label, impact_angle_deg, longitudinal, lateral



