"""
Skeleton-Based Topological Graph Builder for Ground Robot Navigation
Builds sparse, safe graphs for forest environments using medial axis (skeleton)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import distance_transform_edt
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.graph_structure import TopologicalGraph
from drone.semantic_segmentation import SemanticProcessor, TerrainType


class SkeletonGraphBuilder:
    """
    Builds topological graph using skeleton (medial axis) as backbone.
    Designed for forest environments with ground robot navigation (1m clearance required).
    
    Strategy:
    1. Extract skeleton from free space
    2. Sample nodes along skeleton with regular spacing
    3. Ensure all nodes have minimum clearance from obstacles
    4. Connect nodes sparsely (k=2) along skeleton paths
    5. Add minimal shortcut edges for efficiency
    """
    
    def __init__(self,
                 drone_id: str,
                 resolution: float = 0.5,
                 node_spacing: float = 8.0,
                 min_clearance: float = 1.5,
                 max_edge_distance: float = 4.0,
                 k_neighbors: int = 3,
                 world_bounds=None):
        """
        Initialize skeleton-based graph builder.

        Args:
            drone_id:          Unique identifier for this drone.
            resolution:        Map resolution in metres per pixel.
            node_spacing:      Minimum distance between sampled nodes (metres).
            min_clearance:     Minimum clearance from obstacles (metres).
            max_edge_distance: Maximum length of a direct edge (metres).
            k_neighbors:       Maximum neighbours to connect per node.

        Note — perception_radius / safe_build_ratio are intentionally absent.
        safe_build_radius is derived from the map dimensions at each call so
        it automatically adapts to the drone's current altitude.
        """
        self.drone_id = drone_id
        self.resolution = resolution
        self.node_spacing = node_spacing
        self.min_clearance = min_clearance
        self.max_edge_distance = max_edge_distance
        self.k_neighbors = k_neighbors
        self.world_bounds = world_bounds

        # Convert fixed spatial params to pixels
        # Leave room for intermediate nodes even with a small camera footprint.
        self.node_spacing_pixels = max(1., min(node_spacing, max_edge_distance / 2.) / resolution)
        self.min_clearance_pixels = min_clearance / resolution
        self.max_edge_pixels = max_edge_distance / resolution
        self._path_mask_source = None
        
        self.semantic_processor = SemanticProcessor(resolution=resolution)
        self._observation_sequence = 0

        # Timing statistics
        self.timing_stats = {
            'skeleton_extraction': 0.0,
            'node_sampling': 0.0,
            'edge_creation': 0.0
        }

    def _frame(self, observation_id):
        self._observation_sequence += 1
        return float(self._observation_sequence if observation_id is None else observation_id)
    
    def build_graph(self, semantic_map: np.ndarray,
                   drone_position: Tuple[float, float, float], observation_id=None) -> TopologicalGraph:
        """
        Build topological graph from semantic map using skeleton-based approach.
        
        Args:
            semantic_map: 2D array of terrain types from semantic segmentation
            drone_position: (x, y, z) position of drone in world frame
            
        Returns:
            TopologicalGraph instance
        """
        start_time = time.time()
        frame = self._frame(observation_id)
        
        # Process semantic data
        semantic_data = self.semantic_processor.process_segmentation(
            semantic_map, drone_position
        )
        
        # Create traversability mask
        traversable_mask = self._create_traversability_mask(semantic_map)
        
        # Step 1: Extract skeleton with distance transform
        skeleton_time = time.time()
        skeleton, distance_map = self._extract_skeleton_with_distance(traversable_mask)
        self.timing_stats['skeleton_extraction'] = time.time() - skeleton_time

        # Step 2: Sample nodes along skeleton with clearance checking.
        # safe_build_half_x/y are the rectangular safe zone derived from the
        # actual map dimensions (altitude-adaptive).  Using per-axis bounds
        # instead of a single circular radius uses the full camera footprint
        # — especially important for wide-aspect (landscape) cameras where
        # the x half-extent is much larger than the y half-extent.
        margin = self.min_clearance
        safe_build_half_x = max(0.0, (semantic_map.shape[0] * self.resolution) / 2.0 - margin)
        safe_build_half_y = max(0.0, (semantic_map.shape[1] * self.resolution) / 2.0 - margin)
        
        node_time = time.time()
        node_positions_pixel = self._sample_nodes_on_skeleton(
            skeleton, distance_map, semantic_map, drone_position,
            safe_build_half_x, safe_build_half_y,
            semantic_data['coverage_bounds']
        )
        self.timing_stats['node_sampling'] = time.time() - node_time
        
        # Create graph and add nodes
        graph = TopologicalGraph(drone_id=self.drone_id)
        node_id_mapping = {}
        
        for idx, (py, px) in enumerate(node_positions_pixel):
            # Convert to world coordinates
            world_pos = self.semantic_processor.pixel_to_world(
                (px, py),
                semantic_data['coverage_bounds']
            )
            
            # Get terrain type
            terrain_type = self.semantic_processor.get_terrain_type(
                semantic_map, px, py
            )
            
            # Create node
            node_id = f"{self.drone_id}_node_{idx}"
            node_id_mapping[(py, px)] = node_id
            
            # Add node with attributes
            graph.add_node(
                node_id,
                position=world_pos,
                attributes={
                    'observation_count': 0, 'legacy_observation_count': 0,
                    'terrain_type': terrain_type.name,
                    'traversability_cost': self.semantic_processor.get_traversability_cost(terrain_type),
                    'clearance': distance_map[py, px] * self.resolution  # Store clearance
                }
            )
            graph.observe_node(node_id, self.drone_id, frame)
        
        # Step 3: Connect nodes with sparse edges
        edge_time = time.time()
        self._add_sparse_edges(graph, node_positions_pixel, node_id_mapping, semantic_map)
        self.timing_stats['edge_creation'] = time.time() - edge_time
        
        # Update metadata
        graph.update_metadata(
            timestamp=time.time(),
            drone_id=self.drone_id,
            bounds=semantic_data['coverage_bounds'],
            node_count=len(graph),
            edge_count=graph.graph.number_of_edges()
        )
        
        # Validation: Check no nodes are on obstacles (debug)
        self._validate_graph(graph, semantic_map, semantic_data['coverage_bounds'])
        
        # Debug: Print statistics
        if len(graph) > 0:
            traversable_pixels = np.sum(traversable_mask)
            obstacle_pixels = np.sum(~traversable_mask)
            skeleton_pixels = np.sum(skeleton)
            print(f"  [Graph Builder] Traversable: {traversable_pixels} px, "
                  f"Obstacles: {obstacle_pixels} px, "
                  f"Skeleton: {skeleton_pixels} px, "
                  f"Nodes sampled: {len(graph)} "
                  f"(safe zone: ±{safe_build_half_x:.1f}m x ±{safe_build_half_y:.1f}m)")
        
        return graph
    
    def update_graph(self, existing_graph: TopologicalGraph,
                    semantic_map: np.ndarray,
                    drone_position: Tuple[float, float, float],
                    merge_distance_threshold: float = 3.0, observation_id=None):
        """
        Update existing graph with new observations.
        
        Args:
            existing_graph: Existing TopologicalGraph to update
            semantic_map: New semantic map from current observation
            drone_position: Current drone position
            merge_distance_threshold: Distance threshold to consider nodes as duplicates (meters)
        """
        frame = self._frame(observation_id)
        # Process semantic data
        semantic_data = self.semantic_processor.process_segmentation(
            semantic_map, drone_position
        )

        # Every already mapped, actually visible traversable node receives one
        # vote per local sensor frame, independent of skeleton resampling.
        for node_id, data in list(existing_graph.graph.nodes(data=True)):
            min_x,min_y,max_x,max_y=semantic_data['coverage_bounds']
            wx,wy=data['position'][:2]
            if not (min_x<=wx<max_x and min_y<=wy<max_y):
                continue
            pixel = self.semantic_processor.world_to_pixel(data['position'],
                semantic_data['coverage_bounds'], semantic_map.shape)
            if pixel is not None:
                x, y = pixel
                if semantic_map[y, x] in (TerrainType.TRAVERSABLE.value,
                        TerrainType.DIFFICULT.value, TerrainType.CLEARNING.value):
                    existing_graph.observe_node(node_id, self.drone_id, frame)
        
        # Create traversability mask
        traversable_mask = self._create_traversability_mask(semantic_map)
        
        # Extract skeleton
        skeleton_time = time.time()
        skeleton, distance_map = self._extract_skeleton_with_distance(traversable_mask)
        self.timing_stats['skeleton_extraction'] += time.time() - skeleton_time

        # safe bounds derived from current map dimensions (altitude-adaptive)
        margin = self.min_clearance
        safe_build_half_x = max(0.0, (semantic_map.shape[0] * self.resolution) / 2.0 - margin)
        safe_build_half_y = max(0.0, (semantic_map.shape[1] * self.resolution) / 2.0 - margin)

        node_time = time.time()
        new_node_positions_pixel = self._sample_nodes_on_skeleton(
            skeleton, distance_map, semantic_map, drone_position,
            safe_build_half_x, safe_build_half_y,
            semantic_data['coverage_bounds']
        )
        self.timing_stats['node_sampling'] += time.time() - node_time
        
        # Build spatial index if needed
        if not existing_graph._spatial_index_valid:
            existing_graph.build_spatial_index()
        
        # Track new nodes
        new_node_ids = []
        node_id_counter = existing_graph.graph.number_of_nodes()
        
        # Add new nodes (filter duplicates)
        for py, px in new_node_positions_pixel:
            # Convert to world coordinates
            world_pos = self.semantic_processor.pixel_to_world(
                (px, py),
                semantic_data['coverage_bounds']
            )
            
            # Check if too close to existing nodes
            if existing_graph.spatial_index is not None:
                nearby_nodes = existing_graph.find_nearby_nodes(world_pos, merge_distance_threshold)
                if len(nearby_nodes) > 0:
                    continue
            
            # Get terrain type
            terrain_type = self.semantic_processor.get_terrain_type(
                semantic_map, px, py
            )
            
            # Create new node
            node_id = f"{self.drone_id}_node_{node_id_counter}"
            node_id_counter += 1
            
            # Add node
            existing_graph.add_node(
                node_id,
                position=world_pos,
                attributes={
                    'observation_count': 0, 'legacy_observation_count': 0,
                    'terrain_type': terrain_type.name,
                    'traversability_cost': self.semantic_processor.get_traversability_cost(terrain_type),
                    'clearance': distance_map[py, px] * self.resolution
                }
            )
            existing_graph.observe_node(node_id, self.drone_id, frame)
            new_node_ids.append((node_id, world_pos))
        
        # Re-observation can reveal a connection between two OLD nodes too.
        existing_graph.build_spatial_index()
        edge_time = time.time()
        self._refresh_visible_edges(existing_graph, semantic_map, semantic_data)
        self.timing_stats['edge_creation'] += time.time() - edge_time
        
        # Update metadata
        existing_graph.update_metadata(
            timestamp=time.time(),
            node_count=existing_graph.graph.number_of_nodes(),
            edge_count=existing_graph.graph.number_of_edges()
        )
    
    def _create_traversability_mask(self, semantic_map: np.ndarray) -> np.ndarray:
        """
        Create binary traversability mask from semantic map.
        
        Args:
            semantic_map: 2D array of terrain types
            
        Returns:
            Binary mask where True = traversable
        """
        traversable_types = [
            TerrainType.TRAVERSABLE,
            TerrainType.DIFFICULT,
            TerrainType.CLEARNING
        ]
        
        mask = np.zeros_like(semantic_map, dtype=bool)
        for terrain_type in traversable_types:
            mask |= (semantic_map == terrain_type.value)
        
        return mask
    
    def _extract_skeleton_with_distance(self, traversable_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract skeleton (medial axis) with distance transform.
        The distance transform gives clearance from obstacles.
        
        Args:
            traversable_mask: Binary mask of traversable regions
            
        Returns:
            (skeleton, distance_map) tuple
        """
        # Compute distance transform (distance to nearest obstacle)
        # Unknown outside the current observation is not evidence of free space.
        distance_map = distance_transform_edt(np.pad(traversable_mask, 1))[1:-1, 1:-1]
        
        # Extract skeleton using medial axis
        # This gives us paths that maximize clearance from obstacles
        skeleton = skeletonize(traversable_mask)
        
        # Filter skeleton by minimum clearance FIRST (before cleaning)
        # This removes skeleton pixels too close to obstacles
        skeleton = skeleton & (distance_map >= self.min_clearance_pixels)
        
        # Clean up skeleton: remove isolated pixels and small branches MORE AGGRESSIVELY
        # Preserve endpoints: peeling every endpoint also erases useful corridors.
        
        return skeleton, distance_map
    
    def _clean_skeleton(self, skeleton: np.ndarray, min_branch_length: int = 10) -> np.ndarray:
        """
        Clean skeleton by removing short branches and isolated pixels.
        
        Args:
            skeleton: Binary skeleton image
            min_branch_length: Minimum branch length in pixels
            
        Returns:
            Cleaned skeleton
        """
        # Remove isolated pixels (pixels with <2 neighbors)
        cleaned = skeleton.copy()
        
        # Simple pruning: iteratively remove endpoints of short branches
        for _ in range(min_branch_length):
            # Find endpoints (pixels with exactly 1 neighbor)
            neighbors = ndimage.convolve(cleaned.astype(int), 
                                        np.ones((3, 3)), 
                                        mode='constant')
            endpoints = (cleaned) & (neighbors == 2)  # 1 neighbor + self = 2
            
            if np.sum(endpoints) == 0:
                break
            
            # Remove endpoints
            cleaned[endpoints] = False
        
        return cleaned
    
    def _sample_nodes_on_skeleton(self, skeleton: np.ndarray,
                                   distance_map: np.ndarray,
                                   semantic_map: np.ndarray,
                                   drone_position: Tuple[float, float, float],
                                   safe_build_half_x: float,
                                   safe_build_half_y: float,
                                   coverage_bounds: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """
        Sample nodes along skeleton with regular spacing using spatial filtering.
        Uses a greedy approach to maintain minimum spacing globally.
        Only samples nodes within the rectangular safe zone to avoid edge artifacts.

        Args:
            skeleton:            Binary skeleton image (already filtered by clearance)
            distance_map:        Distance transform (clearance from obstacles)
            semantic_map:        Semantic segmentation map
            drone_position:      (x, y, z) drone position in world frame
            safe_build_half_x:   Half-width of the safe zone in x (metres)
            safe_build_half_y:   Half-width of the safe zone in y (metres)
            coverage_bounds:     (min_x, min_y, max_x, max_y) from SemanticProcessor

        Returns:
            List of (y, x) pixel coordinates for nodes
        """
        # Get all skeleton pixels
        # A local skeleton alone does not cover open space or view boundaries.
        # Keep it as the preferred backbone, then fill observed safe space.
        safe = self._create_traversability_mask(semantic_map) & (
            distance_map >= self.min_clearance_pixels + .5)
        skeleton_pixels = np.argwhere(safe)

        if len(skeleton_pixels) == 0:
            return []

        min_x, min_y, max_x, max_y = coverage_bounds

        # Filter skeleton pixels by rectangular safe zone (per-axis bounds).
        # Camera axis mapping (pitch=+90° downward camera, drone facing +X):
        #   row (py) → world X (drone forward/back)
        #   col (px) → world Y (drone left/right)
        safe_skeleton_pixels = []
        for py, px in skeleton_pixels:
            world_x = min_x + (py + 0.5) * self.resolution   # row → X
            world_y = min_y + (px + 0.5) * self.resolution   # col → Y
            local = (np.asarray([py,px])+.5)*self.resolution
            if not self._observed_segment_clear(semantic_map,local,local):
                continue

            if self.world_bounds is not None:
                lo_x, lo_y, hi_x, hi_y = self.world_bounds
                if not (lo_x+self.min_clearance <= world_x <= hi_x-self.min_clearance and
                        lo_y+self.min_clearance <= world_y <= hi_y-self.min_clearance):
                    continue

            if (abs(world_x - drone_position[0]) <= safe_build_half_x and
                    abs(world_y - drone_position[1]) <= safe_build_half_y):
                safe_skeleton_pixels.append((py, px))

        if len(safe_skeleton_pixels) == 0:
            return []
        
        skeleton_pixels = np.array(safe_skeleton_pixels)
        
        # Sort by clearance (prioritize high-clearance nodes)
        clearances = [distance_map[py, px] for py, px in skeleton_pixels]
        backbone = [bool(skeleton[py, px]) for py, px in skeleton_pixels]
        sorted_indices = sorted(range(len(clearances)),
            key=lambda i: (not backbone[i], -clearances[i], tuple(skeleton_pixels[i])))
        skeleton_pixels = skeleton_pixels[sorted_indices]
        
        # Greedy sampling: add nodes that are far from all existing nodes
        nodes = []
        # At least several support points must fit across a small observation.
        spacing_pixels = min(self.node_spacing_pixels, min(semantic_map.shape)/4.)
        
        for py, px in skeleton_pixels:
            # Check terrain type (should be traversable)
            if 0 <= py < semantic_map.shape[0] and 0 <= px < semantic_map.shape[1]:
                terrain_type = TerrainType(semantic_map[py, px])
                if terrain_type in [TerrainType.OBSTACLE, TerrainType.WATER]:
                    continue
            else:
                continue
            
            # Check if far enough from ALL existing nodes
            if len(nodes) > 0:
                nodes_array = np.array(nodes)
                distances = np.sqrt((nodes_array[:, 0] - py)**2 + (nodes_array[:, 1] - px)**2)
                if np.min(distances) < spacing_pixels:
                    continue
            
            # Add node (clearance already verified by skeleton filtering)
            nodes.append((py, px))
        
        return nodes
    
    def _add_sparse_edges(self, graph: TopologicalGraph,
                          node_positions_pixel: List[Tuple[int, int]],
                          node_id_mapping: Dict[Tuple[int, int], str],
                          semantic_map: np.ndarray):
        """
        Add sparse edges connecting nodes.
        Uses k-nearest neighbors with traversability check.
        
        Args:
            graph: TopologicalGraph to add edges to
            node_positions_pixel: List of (y, x) pixel coordinates
            node_id_mapping: Mapping from pixel position to node ID
            semantic_map: Semantic segmentation map
        """
        bounds = graph.metadata.get('bounds')
        # build_graph has not published its metadata yet; derive it from the
        # sampled pixel/world correspondence without assuming a camera origin.
        if node_positions_pixel:
            py, px = node_positions_pixel[0]
            p = graph.get_node_position(node_id_mapping[(py, px)])
            min_x = p[0] - (py + .5) * self.resolution
            min_y = p[1] - (px + .5) * self.resolution
            bounds = (min_x, min_y, min_x + semantic_map.shape[0]*self.resolution,
                      min_y + semantic_map.shape[1]*self.resolution)
            self._refresh_visible_edges(graph, semantic_map,
                dict(coverage_bounds=bounds, map_shape=semantic_map.shape))
    
    def _add_edges_incremental(self, graph: TopologicalGraph,
                               new_nodes: List[Tuple[str, Tuple[float, float]]],
                               semantic_map: np.ndarray,
                               semantic_data: Dict):
        """
        Add edges for newly added nodes (incremental update).
        
        Args:
            graph: TopologicalGraph to add edges to
            new_nodes: List of (node_id, world_position) tuples
            semantic_map: Semantic segmentation map
            semantic_data: Processed semantic data
        """
        self._refresh_visible_edges(graph, semantic_map, semantic_data)
    
    def _is_path_traversable(self, semantic_map: np.ndarray,
                            start_pixel: Tuple[int, int],
                            end_pixel: Tuple[int, int]) -> bool:
        """
        Check if straight-line path between two pixels is traversable.
        Simplified for forest: just check if line hits obstacles.
        
        Args:
            semantic_map: Semantic segmentation map
            start_pixel: (x, y) starting pixel
            end_pixel: (x, y) ending pixel
            
        Returns:
            True if path is traversable
        """
        a = (np.asarray(start_pixel)[::-1] + .5) * self.resolution
        b = (np.asarray(end_pixel)[::-1] + .5) * self.resolution
        return self._observed_segment_clear(semantic_map, a, b)

    def _observed_segment_clear(self, semantic_map, a, b):
        """Exact segment versus inflated pixel rectangles; coordinates relative
        to footprint minimum, ordered world X/Y. Unknown pixels block edges.
        """
        a, b = np.asarray(a), np.asarray(b)
        if ((np.minimum(a,b) < self.min_clearance).any() or
                (np.maximum(a,b) > np.asarray(semantic_map.shape)*self.resolution-self.min_clearance).any()):
            return False
        if self._path_mask_source is not semantic_map:
            blocked = np.argwhere(~self._create_traversability_mask(semantic_map))
            self._blocked_lo = blocked*self.resolution-self.min_clearance
            self._blocked_hi = (blocked+1)*self.resolution+self.min_clearance
            self._path_mask_source = semantic_map
        if not len(self._blocked_lo):
            return True
        near = np.zeros(len(self._blocked_lo))
        far = np.ones(len(self._blocked_lo))
        possible = np.ones(len(near),bool)
        for axis in (0,1):
            delta = b[axis]-a[axis]
            if abs(delta) < 1e-12:
                possible &= (a[axis] >= self._blocked_lo[:,axis]) & (a[axis] <= self._blocked_hi[:,axis])
            else:
                first=(self._blocked_lo[:,axis]-a[axis])/delta
                second=(self._blocked_hi[:,axis]-a[axis])/delta
                near=np.maximum(near,np.minimum(first,second))
                far=np.minimum(far,np.maximum(first,second))
        return not bool(np.any(possible & (near <= far)))

    def _refresh_visible_edges(self, graph, semantic_map, semantic_data):
        """Connect visible components before spending degree on redundant edges.

        Every edge is supported by this actual observation, including clearance.
        No edge is inferred from Euclidean proximity alone.
        """
        import networkx as nx
        from scipy.spatial import cKDTree
        bounds = semantic_data['coverage_bounds']
        visible = []
        for node, data in graph.graph.nodes(data=True):
            x, y = data['position'][:2]
            if bounds[0] <= x < bounds[2] and bounds[1] <= y < bounds[3]:
                pixel = self.semantic_processor.world_to_pixel(data['position'], bounds, semantic_map.shape)
                visible.append((node, data['position'], pixel))
        if len(visible) < 2:
            return
        components = nx.utils.UnionFind()
        for component in nx.connected_components(graph.graph):
            components.union(*component)
        points = np.asarray([v[1] for v in visible])
        candidates = []
        for i, j in cKDTree(points).query_pairs(self.max_edge_distance):
            a, b = visible[i][0], visible[j][0]
            if graph.graph.has_edge(a, b):
                continue
            distance = float(np.linalg.norm(points[i]-points[j]))
            candidates.append((distance, i, j))
        for distance, i, j in sorted(candidates):
            a, b = visible[i][0], visible[j][0]
            bridge = components[a] != components[b]
            if not bridge and min(graph.graph.degree(a), graph.graph.degree(b)) >= self.k_neighbors:
                continue
            origin = np.asarray(bounds[:2])
            if self._observed_segment_clear(semantic_map, points[i]-origin, points[j]-origin):
                weight = self._compute_edge_weight(semantic_map, visible[i][2], visible[j][2], distance)
                graph.add_edge(a, b, weight=weight)
                components.union(a, b)
    
    def _compute_edge_weight(self, semantic_map: np.ndarray,
                           start_pixel: Tuple[int, int],
                           end_pixel: Tuple[int, int],
                           distance: float) -> float:
        """
        Compute edge weight considering distance and terrain difficulty.
        
        Args:
            semantic_map: Semantic segmentation map
            start_pixel: (x, y) starting pixel
            end_pixel: (x, y) ending pixel
            distance: Euclidean distance (meters)
            
        Returns:
            Edge weight (cost)
        """
        # Sample terrain along path
        x1, y1 = start_pixel
        x2, y2 = end_pixel
        
        num_samples = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2)) + 1
        terrain_costs = []
        
        for i in range(num_samples):
            t = i / max(num_samples - 1, 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= y < semantic_map.shape[0] and 0 <= x < semantic_map.shape[1]:
                terrain_type = TerrainType(semantic_map[y, x])
                cost = self.semantic_processor.get_traversability_cost(terrain_type)
                terrain_costs.append(cost)
        
        # Average terrain cost
        avg_terrain_cost = np.mean(terrain_costs) if terrain_costs else 1.0
        
        # Weight = distance * terrain_cost
        return distance * avg_terrain_cost
    
    def _validate_graph(self, graph: TopologicalGraph, semantic_map: np.ndarray,
                       coverage_bounds: Tuple[float, float, float, float] = None):
        """
        Validate that no nodes are placed on obstacles.
        Prints warning if any violations found.
        
        Args:
            graph: Graph to validate
            semantic_map: Semantic map to check against
            coverage_bounds: (min_x, min_y, max_x, max_y) bounds of the map
        """
        if coverage_bounds is None:
            # Can't validate without knowing the map bounds
            return
        
        min_x, min_y, max_x, max_y = coverage_bounds
        violations = []
        
        for node_id, node_data in graph.graph.nodes(data=True):
            world_pos = node_data['position']
            
            # Convert world position to pixel coordinates
            pixel_x = int((world_pos[0] - min_x) / self.resolution)
            pixel_y = int((world_pos[1] - min_y) / self.resolution)
            
            # Check bounds
            if 0 <= pixel_y < semantic_map.shape[0] and 0 <= pixel_x < semantic_map.shape[1]:
                terrain = TerrainType(semantic_map[pixel_y, pixel_x])
                if terrain in [TerrainType.OBSTACLE, TerrainType.WATER]:
                    violations.append((node_id, world_pos, terrain.name, (pixel_x, pixel_y)))
        
        if len(violations) > 0:
            print(f"⚠️  WARNING: {len(violations)} nodes on obstacles!")
            for node_id, pos, terrain, pix in violations[:3]:  # Show first 3
                print(f"    {node_id} at world({pos[0]:.1f}, {pos[1]:.1f}) pixel({pix[0]}, {pix[1]}) on {terrain}")
            if len(violations) > 3:
                print(f"    ... and {len(violations) - 3} more")

