"""
Contact Tracking using SAT Normal Vector Projections

Core idea: Use SAT's edge normals to project vertices and find contacts.
Track which vertex-edge pairs have minimal overlap in projections.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sat_collision import separating_axis_theorem, project_polygon, interval_distance


@dataclass(frozen=True)
class Contact:
    """Single contact point - which EGO part hit which NPC part."""
    contact_type: str  # "vertex-edge" or "edge-edge"
    npc_edge_id: int    # 0=rear, 1=left, 2=front, 3=right
    distance: float     # Overlap distance
    position: np.ndarray  # Contact position
    signed_distance: float  # Signed distance (negative = penetrating)
    ego_vertex_id: Optional[int] = None  # None for edge-to-edge
    ego_edge_id: Optional[int] = None  # None for vertex-to-edge


def get_edge_normal(polygon: np.ndarray, edge_id: int) -> np.ndarray:
    """Get normalized normal vector for edge (pointing outward)."""
    edge_start = polygon[edge_id]
    edge_end = polygon[(edge_id + 1) % (len(polygon) - 1)]
    edge_vec = edge_end - edge_start
    normal = np.array([-edge_vec[1], edge_vec[0]])
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return np.array([1.0, 0.0])
    normal = normal / norm
    
    # Ensure normal points outward
    centroid = polygon[:-1].mean(axis=0)
    edge_mid = (edge_start + edge_end) / 2
    to_centroid = centroid - edge_mid
    if np.dot(normal, to_centroid) > 0:
        normal = -normal
    
    return normal


def find_primary_contact(ego_polygon: np.ndarray, npc_polygon: np.ndarray, 
                         ego_velocity: np.ndarray = None, npc_velocity: np.ndarray = None,
                         threshold: float = 10.0) -> Optional[Contact]:
    """
    Find primary contact using SAT normal vector projections.
    
    Strategy:
    1. Use SAT to detect collision and get MTV
    2. For each NPC edge, get its normal (from SAT)
    3. Project EGO vertices onto NPC edge normals
    4. Find vertex-edge pairs with minimal overlap
    5. Select best contact based on overlap distance
    """
    if ego_velocity is None:
        ego_velocity = np.array([0.0, 0.0])
    if npc_velocity is None:
        npc_velocity = np.array([0.0, 0.0])
    
    sat_result = separating_axis_theorem(ego_polygon, npc_polygon, ego_velocity, npc_velocity)
    
    if not sat_result.intersecting or sat_result.minimum_translation_vector is None:
        return None
    
    mtv = sat_result.minimum_translation_vector
    mtv_normalized = mtv / (np.linalg.norm(mtv) + 1e-10)
    
    # Determine collision side from relative positions
    ego_centroid = ego_polygon[:-1].mean(axis=0)
    npc_centroid = npc_polygon[:-1].mean(axis=0)
    y_diff = npc_centroid[1] - ego_centroid[1]
    
    collision_side = None
    if abs(y_diff) > 0.1:
        collision_side = "left" if y_diff < 0 else "right"
    elif abs(mtv_normalized[1]) > 0.3:
        collision_side = "left" if mtv_normalized[1] > 0 else "right"
    
    all_contacts = []
    
    # Check edge-edge contacts (front/rear only, for rear-end collisions)
    if not collision_side:
        front_rear_edges = [0, 2]
        for ego_eid in front_rear_edges:
            for npc_eid in front_rear_edges:
                # Get normals for both edges
                ego_normal = get_edge_normal(ego_polygon, ego_eid)
                npc_normal = get_edge_normal(npc_polygon, npc_eid)
                
                # Check if edges are parallel (dot product close to 1)
                if abs(np.dot(ego_normal, npc_normal)) < 0.95:
                    continue
                
                # Project both edges onto NPC normal
                ego_edge_verts = np.array([ego_polygon[ego_eid], ego_polygon[(ego_eid + 1) % 4]])
                npc_edge_verts = np.array([npc_polygon[npc_eid], npc_polygon[(npc_eid + 1) % 4]])
                
                min_ego, max_ego = project_polygon(ego_edge_verts, npc_normal)
                min_npc, max_npc = project_polygon(npc_edge_verts, npc_normal)
                
                overlap_dist = interval_distance(min_ego, max_ego, min_npc, max_npc)
                
                if overlap_dist <= threshold:
                    # Contact point is midpoint of overlap region
                    overlap_start = max(min_ego, min_npc)
                    overlap_end = min(max_ego, max_npc)
                    overlap_mid = (overlap_start + overlap_end) / 2
                    
                    # Find point on edge closest to overlap midpoint
                    edge_vec = npc_polygon[(npc_eid + 1) % 4] - npc_polygon[npc_eid]
                    edge_len_sq = np.dot(edge_vec, edge_vec)
                    if edge_len_sq > 1e-10:
                        to_mid = npc_polygon[npc_eid] - npc_centroid
                        t = np.clip(np.dot(to_mid, edge_vec) / edge_len_sq, 0.0, 1.0)
                        contact_pos = npc_polygon[npc_eid] + t * edge_vec
                    else:
                        contact_pos = npc_polygon[npc_eid]
                    
                    all_contacts.append((
                        Contact(
                            contact_type="edge-edge",
                            npc_edge_id=npc_eid,
                            distance=abs(overlap_dist),
                            position=contact_pos,
                            signed_distance=overlap_dist,
                            ego_edge_id=ego_eid
                        ),
                        overlap_dist - 1.0  # Priority boost for edge-edge
                    ))
    
    # Check vertex-edge contacts using SAT normal projections
    vertex_sides = {0: "right", 1: "left", 2: "left", 3: "right"}
    
    for ego_vid in range(4):
        ego_vertex = ego_polygon[ego_vid]
        vertex_side = vertex_sides[ego_vid]
        
        # Filter by collision side
        if collision_side and vertex_side != collision_side:
            continue
        
        for npc_eid in range(4):
            # Get NPC edge normal (this is what SAT uses)
            npc_normal = get_edge_normal(npc_polygon, npc_eid)
            
            # Project EGO vertex onto NPC edge normal
            vertex_proj = np.dot(ego_vertex, npc_normal)
            
            # Project NPC edge onto its own normal
            npc_edge_verts = np.array([npc_polygon[npc_eid], npc_polygon[(npc_eid + 1) % 4]])
            min_npc, max_npc = project_polygon(npc_edge_verts, npc_normal)
            
            # Distance from vertex projection to edge projection interval
            if vertex_proj < min_npc:
                overlap_dist = min_npc - vertex_proj
            elif vertex_proj > max_npc:
                overlap_dist = vertex_proj - max_npc
            else:
                overlap_dist = 0.0  # Vertex projection overlaps edge projection
            
            if overlap_dist <= threshold:
                # Find closest point on edge to vertex
                edge_start = npc_polygon[npc_eid]
                edge_end = npc_polygon[(npc_eid + 1) % 4]
                edge_vec = edge_end - edge_start
                point_vec = ego_vertex - edge_start
                
                edge_len_sq = np.dot(edge_vec, edge_vec)
                if edge_len_sq > 1e-10:
                    t = np.clip(np.dot(point_vec, edge_vec) / edge_len_sq, 0.0, 1.0)
                    contact_pos = edge_start + t * edge_vec
                else:
                    contact_pos = edge_start
                
                # Compute signed distance (negative = penetrating)
                to_vertex = ego_vertex - contact_pos
                signed_dist = np.dot(to_vertex, npc_normal)
                
                # Determine expected edge for side collisions
                expected_edge = None
                if collision_side == "left":
                    expected_edge = 3  # NPC right edge
                elif collision_side == "right":
                    expected_edge = 1  # NPC left edge
                
                # Score: prioritize expected edge, then by overlap distance
                score = overlap_dist
                if expected_edge == npc_eid:
                    score = overlap_dist - 5.0  # Huge boost for expected edge
                elif signed_dist < 0.01:  # Penetrating
                    score = overlap_dist - 0.5
                
                all_contacts.append((
                    Contact(
                        contact_type="vertex-edge",
                        npc_edge_id=npc_eid,
                        distance=overlap_dist,
                        position=contact_pos,
                        signed_distance=signed_dist,
                        ego_vertex_id=ego_vid
                    ),
                    score
                ))
    
    if not all_contacts:
        return None
    
    # Select contact with lowest score (best overlap)
    all_contacts.sort(key=lambda x: x[1])
    return all_contacts[0][0]


def classify_collision(contact: Contact) -> str:
    """Classify collision based on contact geometry."""
    EDGE_NAMES = {0: "rear", 1: "left", 2: "front", 3: "right"}
    VERTEX_NAMES = {0: "rear-right", 1: "rear-left", 2: "front-left", 3: "front-right"}
    
    npc_part = EDGE_NAMES[contact.npc_edge_id]
    
    if contact.contact_type == "edge-edge":
        ego_part = EDGE_NAMES[contact.ego_edge_id]
        if ego_part == "front" and npc_part == "rear":
            return "Rear-end collision (EGO hit NPC from behind)"
        elif ego_part == "rear" and npc_part == "front":
            return "Rear-ended collision (NPC hit EGO from behind)"
        elif ego_part == "front" and npc_part == "front":
            return "Head-on collision"
        else:
            return f"Edge-to-edge collision (EGO {ego_part}, NPC {npc_part})"
    else:
        ego_v = contact.ego_vertex_id
        ego_part = VERTEX_NAMES[ego_v]
        
        if ego_v in [1, 2]:
            ego_side = "left"
        elif ego_v in [0, 3]:
            ego_side = "right"
        else:
            ego_side = None
        
        if ego_side and npc_part in ["left", "right"]:
            return f"Side-swipe collision (EGO's {ego_side} side, NPC's {npc_part} side)"
        elif ego_v in [2, 3] and npc_part in ["left", "right"]:
            return f"T-bone collision (EGO front hit NPC {npc_part} side)"
        elif ego_v in [0, 1] and npc_part in ["left", "right"]:
            return f"T-bone collision (EGO rear hit NPC {npc_part} side)"
        else:
            return "Complex collision"


def format_contact(contact: Contact) -> str:
    """Format contact as human-readable string."""
    # More descriptive corner names: "top" = front, "bottom" = rear
    VERTEX_NAMES = {
        0: "bottom-right corner",  # rear-right
        1: "bottom-left corner",   # rear-left
        2: "top-left corner",      # front-left
        3: "top-right corner"      # front-right
    }
    EDGE_NAMES = {0: "rear edge", 1: "left edge", 2: "front edge", 3: "right edge"}
    
    npc_part = EDGE_NAMES[contact.npc_edge_id]
    
    if contact.contact_type == "edge-edge":
        ego_part = EDGE_NAMES[contact.ego_edge_id]
        return f"EGO's {ego_part} hit NPC's {npc_part}"
    else:
        ego_part = VERTEX_NAMES[contact.ego_vertex_id]
        return f"EGO's {ego_part} hit NPC's {npc_part}"
