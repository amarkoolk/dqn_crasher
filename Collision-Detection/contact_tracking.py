"""
Contact tracking collision detector.

This method directly tracks which vertices penetrate which edges,
providing richer contact information than SAT while potentially being faster.

Core idea (from professor's suggestion):
- Track active states: edges and vertices of both vehicles
- Identify which vertex hit which edge
- Compute contact normals directly from edge geometry
- Early exit when first contact found (optional optimization)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from collision_methods import (
    CollisionDetector,
    CollisionResult,
    ContactPoint,
    point_to_line_segment_distance,
    compute_polygon_centroid
)


class ContactTrackingDetector(CollisionDetector):
    """
    Vertex/edge contact tracking collision detector.
    
    Instead of testing all separating axes (SAT), this method:
    1. Tests each vertex of polygon A against each edge of polygon B
    2. Tests each vertex of polygon B against each edge of polygon A
    3. Records all contact points where vertices penetrate edges
    4. Optionally supports early exit for performance
    
    Advantages:
    - Richer data: exact contact geometry (which vertex hit which edge)
    - Potentially faster: can exit early
    - Simpler geometry: just point-to-edge tests
    
    Disadvantages:
    - May need edge-edge contact handling for parallel edges
    - Multiple contacts require aggregation
    """
    
    def __init__(self, early_exit: bool = False, penetration_threshold: float = 0.05):
        """
        Parameters
        ----------
        early_exit : bool
            If True, return immediately when first contact found.
            Faster but provides less complete contact information.
        penetration_threshold : float
            Distance threshold (in meters) to consider a vertex as penetrating.
            Typical: 0.05m = 5cm (allows slight numerical tolerance)
        """
        super().__init__("ContactTracking")
        self.early_exit = early_exit
        self.penetration_threshold = penetration_threshold
    
    def detect(
        self,
        polygon_a: np.ndarray,
        polygon_b: np.ndarray,
        displacement_a: np.ndarray,
        displacement_b: np.ndarray
    ) -> CollisionResult:
        """
        Detect collision by tracking vertex-edge contacts.
        
        Algorithm:
        1. For each vertex of A, test against all edges of B
        2. For each vertex of B, test against all edges of A
        3. Identify penetrations using signed distance (inside/outside test)
        4. Aggregate contact information
        """
        contacts: List[ContactPoint] = []
        
        # Test vertices of A against edges of B
        contacts_ab = self._test_vertices_against_edges(
            polygon_a, polygon_b, vertex_polygon_id='A', edge_polygon_id='B'
        )
        contacts.extend(contacts_ab)
        
        if self.early_exit and contacts:
            return self._build_result(True, contacts)
        
        # Test vertices of B against edges of A
        contacts_ba = self._test_vertices_against_edges(
            polygon_b, polygon_a, vertex_polygon_id='B', edge_polygon_id='A'
        )
        contacts.extend(contacts_ba)
        
        # Check for collision
        intersecting = len(contacts) > 0
        
        return self._build_result(intersecting, contacts)
    
    def _test_vertices_against_edges(
        self,
        vertex_polygon: np.ndarray,
        edge_polygon: np.ndarray,
        vertex_polygon_id: str,
        edge_polygon_id: str
    ) -> List[ContactPoint]:
        """
        Test all vertices of one polygon against all edges of another.
        
        Parameters
        ----------
        vertex_polygon : np.ndarray
            Polygon whose vertices we test.
        edge_polygon : np.ndarray
            Polygon whose edges we test against.
        vertex_polygon_id, edge_polygon_id : str
            Identifiers for tracking which polygon is which.
        
        Returns
        -------
        List[ContactPoint]
            All detected contact points.
        """
        contacts = []
        
        # Compute edge polygon centroid for inside/outside tests
        centroid = compute_polygon_centroid(edge_polygon)
        
        # Test each vertex (excluding the repeated last one)
        for vertex_id, vertex in enumerate(vertex_polygon[:-1]):
            # Test against each edge
            for edge_id in range(len(edge_polygon) - 1):
                edge_start = edge_polygon[edge_id]
                edge_end = edge_polygon[edge_id + 1]
                
                # Get closest point on edge and distance
                distance, closest_point, t_param = point_to_line_segment_distance(
                    vertex, edge_start, edge_end
                )
                
                # Compute edge normal (perpendicular, pointing outward)
                edge_vector = edge_end - edge_start
                edge_normal = np.array([-edge_vector[1], edge_vector[0]])
                edge_normal_length = np.linalg.norm(edge_normal)
                
                if edge_normal_length < 1e-10:
                    continue  # Degenerate edge
                
                edge_normal = edge_normal / edge_normal_length
                
                # Ensure normal points outward (away from centroid)
                edge_midpoint = (edge_start + edge_end) / 2
                to_centroid = centroid - edge_midpoint
                if np.dot(edge_normal, to_centroid) > 0:
                    edge_normal = -edge_normal
                
                # Check if vertex is on the inside (penetrating) side
                to_vertex = vertex - closest_point
                signed_distance = np.dot(to_vertex, edge_normal)
                
                # Penetration if vertex is inside (negative signed distance)
                # and close enough to the edge
                if signed_distance < self.penetration_threshold and distance < self.penetration_threshold:
                    penetration_depth = abs(signed_distance)
                    
                    contact = ContactPoint(
                        vertex_id=vertex_id,
                        edge_id=edge_id,
                        position=closest_point,
                        normal=-edge_normal,  # Normal points from edge to vertex (penetration direction)
                        penetration=penetration_depth
                    )
                    contacts.append(contact)
        
        return contacts
    
    def _build_result(self, intersecting: bool, contacts: List[ContactPoint]) -> CollisionResult:
        """
        Build unified CollisionResult from contact points.
        
        If multiple contacts exist, aggregate them to get a single
        collision normal and penetration depth.
        """
        collision_normal = None
        penetration_depth = None
        
        if contacts:
            # Aggregate multiple contacts
            # Strategy: Use the deepest penetration
            deepest_contact = max(contacts, key=lambda c: c.penetration)
            collision_normal = deepest_contact.normal / np.linalg.norm(deepest_contact.normal)
            penetration_depth = deepest_contact.penetration
            
            # Alternative strategy: Average all contact normals (weighted by penetration)
            # This could be explored in future research
        
        return CollisionResult(
            method=self.name,
            intersecting=intersecting,
            computation_time_ns=0,  # Filled by detect_timed
            collision_normal=collision_normal,
            penetration_depth=penetration_depth,
            contact_points=contacts if contacts else None,
            metadata={
                'num_contacts': len(contacts),
                'early_exit': self.early_exit
            }
        )
    
    def get_contact_summary(self, contacts: List[ContactPoint]) -> dict:
        """
        Generate human-readable summary of contact points.
        
        Useful for analysis and debugging.
        """
        if not contacts:
            return {'num_contacts': 0}
        
        # Count contacts by edge
        edge_counts = {}
        for contact in contacts:
            edge_counts[contact.edge_id] = edge_counts.get(contact.edge_id, 0) + 1
        
        # Identify dominant contact edge
        dominant_edge = max(edge_counts, key=edge_counts.get)
        
        # Edge labels (for rectangles: 0=rear, 1=left, 2=front, 3=right)
        edge_labels = {0: 'rear', 1: 'left', 2: 'front', 3: 'right'}
        
        return {
            'num_contacts': len(contacts),
            'edges_involved': list(edge_counts.keys()),
            'dominant_edge': dominant_edge,
            'dominant_edge_label': edge_labels.get(dominant_edge, f'edge_{dominant_edge}'),
            'max_penetration': max(c.penetration for c in contacts),
            'avg_penetration': sum(c.penetration for c in contacts) / len(contacts)
        }


