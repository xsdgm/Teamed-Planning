"""
Topological Graph Structure for Multi-Robot Navigation
Combines NetworkX for graph operations with KD-tree for spatial queries
"""

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Optional, Any
import json
import time


class TopologicalGraph:
    """
    A topological graph for navigation with spatial indexing capabilities.
    
    Attributes:
        graph: NetworkX graph storing nodes and edges
        spatial_index: KDTree for fast spatial queries
        metadata: Dictionary containing graph metadata (timestamp, drone_id, bounds, etc.)
    """

    # How informative one drone observation is per terrain type.
    # Derived from the fraction of ground area visible through each vegetation class
    # with a nadir-facing camera.  One physical constant (TRAVERSABLE=0.7) anchors
    # the scale; all others follow relative occlusion ratios.
    _TERRAIN_RELIABILITY: Dict[str, float] = {
        'CLEARNING':   1.0,   # open clearing — fully visible
        'TRAVERSABLE': 0.7,   # normal ground — mostly visible
        'DIFFICULT':   0.3,   # rough / partially occluded terrain
        'UNKNOWN':     0.1,   # unclassified — very low reliability
        'OBSTACLE':    0.0,   # irrelevant for traversal confidence
        'WATER':       0.0,   # not traversable
    }
    
    def __init__(self, drone_id: Optional[str] = None):
        """
        Initialize an empty topological graph.
        
        Args:
            drone_id: Identifier for the drone that created this graph
        """
        self.graph = nx.Graph()
        self.spatial_index = None
        self._spatial_index_valid = False
        self.metadata = {
            'drone_id': drone_id,
            'timestamp': None,
            'bounds': None,  # (min_x, min_y, max_x, max_y)
            'node_count': 0,
            'edge_count': 0
        }
        # Track fusion history with other drones
        self.fusion_history = {}  # {other_drone_id: last_fusion_timestamp}
        
    def add_node(self, node_id: str, position: Tuple[float, float], 
                 attributes: Optional[Dict[str, Any]] = None):
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            position: (x, y) position of the node
            attributes: Additional attributes (terrain_type, traversability, etc.)
        """
        if attributes is None:
            attributes = {}
        
        # Add default fusion attributes if not provided
        if 'observation_count' not in attributes:
            attributes['observation_count'] = 1
        if 'confidence' not in attributes:
            attributes['confidence'] = 0.1  # Initial confidence
        if 'last_updated' not in attributes:
            attributes['last_updated'] = time.time()
        
        # Store position as node attribute
        node_attrs = {
            'position': position,
            **attributes
        }
        
        self.graph.add_node(node_id, **node_attrs)
        self._spatial_index_valid = False  # Invalidate spatial index
        self.metadata['node_count'] = self.graph.number_of_nodes()
        
    def add_edge(self, node_id1: str, node_id2: str, 
                 weight: Optional[float] = None, 
                 attributes: Optional[Dict[str, Any]] = None):
        """
        Add an edge between two nodes.
        
        Args:
            node_id1: First node identifier
            node_id2: Second node identifier
            weight: Edge weight (typically distance or traversal cost)
            attributes: Additional edge attributes
        """
        if attributes is None:
            attributes = {}
            
        # Calculate weight based on distance if not provided
        if weight is None and node_id1 in self.graph and node_id2 in self.graph:
            pos1 = np.array(self.graph.nodes[node_id1]['position'])
            pos2 = np.array(self.graph.nodes[node_id2]['position'])
            weight = np.linalg.norm(pos1 - pos2)
        
        edge_attrs = {'weight': weight, **attributes}
        self.graph.add_edge(node_id1, node_id2, **edge_attrs)
        self.metadata['edge_count'] = self.graph.number_of_edges()
        
    def build_spatial_index(self):
        """
        Build KD-tree spatial index from node positions.
        Should be called before spatial queries for efficiency.
        """
        if self.graph.number_of_nodes() == 0:
            self.spatial_index = None
            self._spatial_index_valid = False
            return
            
        positions = []
        node_ids = []
        
        for node_id, data in self.graph.nodes(data=True):
            positions.append(data['position'])
            node_ids.append(node_id)
        
        self.spatial_index = KDTree(np.array(positions))
        self._node_id_list = node_ids  # Keep mapping from index to node_id
        self._spatial_index_valid = True
        
    def find_nearby_nodes(self, position: Tuple[float, float], 
                         radius: float) -> List[Tuple[str, float]]:
        """
        Find all nodes within a given radius of a position.
        
        Args:
            position: (x, y) query position
            radius: Search radius
            
        Returns:
            List of (node_id, distance) tuples
        """
        if not self._spatial_index_valid:
            self.build_spatial_index()
        
        if self.spatial_index is None:
            return []
        
        indices = self.spatial_index.query_ball_point(position, radius)
        results = []
        
        for idx in indices:
            node_id = self._node_id_list[idx]
            node_pos = np.array(self.graph.nodes[node_id]['position'])
            distance = np.linalg.norm(np.array(position) - node_pos)
            results.append((node_id, distance))
        
        return sorted(results, key=lambda x: x[1])  # Sort by distance
    
    def find_nearest_node(self, position: Tuple[float, float]) -> Optional[Tuple[str, float]]:
        """
        Find the nearest node to a given position.
        
        Args:
            position: (x, y) query position
            
        Returns:
            (node_id, distance) tuple or None if graph is empty
        """
        if not self._spatial_index_valid:
            self.build_spatial_index()
        
        if self.spatial_index is None:
            return None
        
        distance, idx = self.spatial_index.query(position)
        node_id = self._node_id_list[idx]
        
        return (node_id, distance)
    
    def get_node_position(self, node_id: str) -> Optional[Tuple[float, float]]:
        """Get position of a node."""
        if node_id in self.graph:
            return self.graph.nodes[node_id]['position']
        return None
    
    def update_metadata(self, **kwargs):
        """Update metadata fields."""
        self.metadata.update(kwargs)
        
    def to_message(self) -> Dict:
        """
        Serialize graph to dictionary format for transmission.
        
        Returns:
            Dictionary containing graph data and metadata
        """
        # Use NetworkX's node-link format
        graph_data = nx.node_link_data(self.graph)
        
        return {
            'graph_data': graph_data,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_message(cls, message_data: Dict) -> 'TopologicalGraph':
        """
        Deserialize graph from message dictionary.
        
        Args:
            message_data: Dictionary containing graph data and metadata
            
        Returns:
            TopologicalGraph instance
        """
        graph = cls(drone_id=message_data['metadata'].get('drone_id'))
        graph.graph = nx.node_link_graph(message_data['graph_data'])
        graph.metadata = message_data['metadata']
        graph._spatial_index_valid = False
        
        return graph
    
    def to_json(self) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_message())
    
    @classmethod
    def from_json(cls, json_string: str) -> 'TopologicalGraph':
        """Deserialize graph from JSON string."""
        message_data = json.loads(json_string)
        return cls.from_message(message_data)
    
    def get_path(self, start_node: str, goal_node: str, 
                 weight: str = 'weight') -> Optional[List[str]]:
        """
        Find shortest path between two nodes using A* algorithm.
        
        Args:
            start_node: Starting node ID
            goal_node: Goal node ID
            weight: Edge attribute to use as weight
            
        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        if start_node not in self.graph or goal_node not in self.graph:
            return None
        try:
            path = nx.astar_path(self.graph, start_node, goal_node, weight=weight)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
            return None
    
    def get_path_length(self, start_node: str, goal_node: str,
                       weight: str = 'weight') -> Optional[float]:
        """
        Get the length of the shortest path between two nodes.
        
        Args:
            start_node: Starting node ID
            goal_node: Goal node ID  
            weight: Edge attribute to use as weight
            
        Returns:
            Path length or None if no path exists
        """
        if start_node not in self.graph or goal_node not in self.graph:
            return None
        try:
            length = nx.astar_path_length(self.graph, start_node, goal_node, weight=weight)
            return length
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
            return None
    
    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    def update_node_attributes(self, node_id: str, **attributes):
        """
        Update attributes of an existing node.
        
        Args:
            node_id: Node identifier
            **attributes: Attributes to update
        """
        if node_id in self.graph:
            for key, value in attributes.items():
                self.graph.nodes[node_id][key] = value
    
    def fuse_with_graph(self, other_graph: 'TopologicalGraph', 
                       merge_distance_threshold: float = 3.0,
                       current_time: Optional[float] = None) -> Dict[str, int]:
        """
        Fuse another graph into this graph with intelligent node merging.
        
        Args:
            other_graph: The graph to fuse into this one
            merge_distance_threshold: Distance threshold for merging nodes (meters)
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            Dictionary with fusion statistics
        """
        if current_time is None:
            current_time = time.time()
        
        stats = {
            'nodes_merged': 0,
            'nodes_added': 0,
            'edges_added': 0
        }
        
        # Incremental index for this fusion only. Rebuilding a KD-tree for
        # every averaged node makes large fleet graphs unnecessarily slow.
        cell_size = merge_distance_threshold if merge_distance_threshold > 0 else 1.
        def cell(position):
            return tuple(int(np.floor(value / cell_size)) for value in position[:2])
        buckets = {}
        for node_id, data in self.graph.nodes(data=True):
            buckets.setdefault(cell(data['position']), []).append(node_id)
        
        # Track mapping from other_graph node_ids to this graph node_ids
        node_mapping = {}
        
        # Process each node from other graph
        for other_node_id, other_node_data in other_graph.graph.nodes(data=True):
            other_pos = other_node_data['position']
            
            # Check if this position is close to an existing node
            cx, cy = cell(other_pos)
            nearby_nodes = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for node_id in buckets.get((cx + dx, cy + dy), []):
                        distance = float(np.linalg.norm(np.asarray(other_pos) - self.graph.nodes[node_id]['position']))
                        if distance <= merge_distance_threshold:
                            nearby_nodes.append((node_id, distance))
            nearby_nodes.sort(key=lambda item: item[1])
            
            if len(nearby_nodes) > 0:
                # Merge with the nearest existing node
                nearest_node_id, _ = nearby_nodes[0]
                node_mapping[other_node_id] = nearest_node_id
                
                # Update existing node with fusion logic
                existing_node = self.graph.nodes[nearest_node_id]
                
                # Update observation count (cap to avoid int32 overflow on
                # repeated inter-drone fusion — confidence saturates anyway)
                existing_count = existing_node.get('observation_count', 1)
                other_count = other_node_data.get('observation_count', 1)
                new_count = min(existing_count + other_count, 1000)
                
                # Update confidence using the formal model c(v) = 1-exp(-obs*r)
                r_val = self._terrain_reliability(
                    existing_node.get('terrain_type', 'UNKNOWN'))
                new_confidence = float(1.0 - np.exp(-new_count * r_val))
                
                # Update position with weighted average based on confidence
                existing_confidence = existing_node.get('confidence', 0.1)
                other_confidence = other_node_data.get('confidence', 0.1)
                total_confidence = existing_confidence + other_confidence
                
                existing_pos = np.array(existing_node['position'])
                other_pos_arr = np.array(other_pos)
                new_pos = (existing_pos * existing_confidence + 
                          other_pos_arr * other_confidence) / total_confidence
                old_cell, new_cell = cell(existing_pos), cell(new_pos)
                if old_cell != new_cell:
                    buckets[old_cell].remove(nearest_node_id)
                    buckets.setdefault(new_cell, []).append(nearest_node_id)
                
                # Update node attributes
                self.update_node_attributes(
                    nearest_node_id,
                    position=tuple(new_pos),
                    observation_count=new_count,
                    confidence=new_confidence,
                    last_updated=current_time
                )
                
                stats['nodes_merged'] += 1
                self._spatial_index_valid = False  # Need to rebuild after position update
                
            else:
                # Add as new node
                new_node_id = f"{self.metadata.get('drone_id', 'unknown')}_fused_{len(self.graph.nodes())}"
                node_mapping[other_node_id] = new_node_id
                
                # Copy all attributes
                node_attrs = dict(other_node_data)
                self.add_node(new_node_id, position=other_pos, attributes=node_attrs)
                buckets.setdefault(cell(other_pos), []).append(new_node_id)
                
                stats['nodes_added'] += 1
        
        # Rebuild spatial index after all node updates
        if stats['nodes_merged'] > 0 or stats['nodes_added'] > 0:
            self.build_spatial_index()
        
        # Process edges from other graph
        for node1, node2, edge_data in other_graph.graph.edges(data=True):
            # Map to this graph's node IDs
            mapped_node1 = node_mapping.get(node1)
            mapped_node2 = node_mapping.get(node2)
            
            if mapped_node1 and mapped_node2:
                # Add edge if it doesn't exist
                if not self.graph.has_edge(mapped_node1, mapped_node2):
                    self.add_edge(mapped_node1, mapped_node2, 
                                weight=edge_data.get('weight'),
                                attributes={k: v for k, v in edge_data.items() if k != 'weight'})
                    stats['edges_added'] += 1
        
        # Update metadata
        self.metadata['node_count'] = self.graph.number_of_nodes()
        self.metadata['edge_count'] = self.graph.number_of_edges()
        self.metadata['timestamp'] = current_time
        
        # Record fusion history
        other_drone_id = other_graph.metadata.get('drone_id')
        if other_drone_id:
            self.fusion_history[other_drone_id] = current_time
        
        return stats
    
    def should_fuse_with_drone(self, other_drone_id: str, 
                               current_time: float,
                               fusion_interval_threshold: float = 30.0) -> bool:
        """
        Check if enough time has passed since last fusion with a specific drone.
        
        Args:
            other_drone_id: ID of the other drone
            current_time: Current timestamp
            fusion_interval_threshold: Minimum seconds between fusions (default 30s)
            
        Returns:
            True if should fuse, False otherwise
        """
        last_fusion_time = self.fusion_history.get(other_drone_id, float('-inf'))
        time_since_fusion = current_time - last_fusion_time
        return time_since_fusion >= fusion_interval_threshold
    
    def to_ros_msg(self):
        """
        Convert to teamed_planning/TopologicalGraphMsg ROS message.
        Import is deferred so this file can be used without ROS.
        """
        from teamed_planning.msg import TopologicalGraphMsg, GraphNode, GraphEdge
        import rospy

        msg = TopologicalGraphMsg()
        msg.header.stamp = rospy.Time.now()
        msg.drone_id = self.metadata.get('drone_id', '')
        msg.is_full_graph = True

        bounds = self.metadata.get('bounds')
        if bounds and len(bounds) == 4:
            msg.bounds_min_x = bounds[0]
            msg.bounds_min_y = bounds[1]
            msg.bounds_max_x = bounds[2]
            msg.bounds_max_y = bounds[3]

        for node_id, data in self.graph.nodes(data=True):
            n = GraphNode()
            n.node_id = str(node_id)
            pos = data.get('position', (0.0, 0.0))
            n.x = float(pos[0])
            n.y = float(pos[1])
            n.observation_count = min(int(data.get('observation_count', 1)), 2_000_000_000)
            n.confidence = float(data.get('confidence', 0.1))
            n.last_updated = float(data.get('last_updated', 0.0))
            msg.nodes.append(n)

        for u, v, data in self.graph.edges(data=True):
            e = GraphEdge()
            e.source = str(u)
            e.target = str(v)
            e.weight = float(data.get('weight', 0.0))
            msg.edges.append(e)

        return msg

    @classmethod
    def from_ros_msg(cls, msg) -> 'TopologicalGraph':
        """Create TopologicalGraph from a teamed_planning/TopologicalGraphMsg."""
        graph = cls(drone_id=msg.drone_id)

        for n in msg.nodes:
            graph.add_node(n.node_id, position=(n.x, n.y), attributes={
                'observation_count': n.observation_count,
                'confidence': n.confidence,
                'last_updated': n.last_updated,
            })

        for e in msg.edges:
            if e.source in graph.graph and e.target in graph.graph:
                graph.add_edge(e.source, e.target, weight=e.weight)

        graph.metadata['bounds'] = (
            msg.bounds_min_x, msg.bounds_min_y,
            msg.bounds_max_x, msg.bounds_max_y,
        )
        return graph

    # ── Uncertainty / confidence helpers ─────────────────────────────────────

    def _terrain_reliability(self, terrain_type_str: str) -> float:
        """r(v): how informative one aerial observation is for this terrain type."""
        return self._TERRAIN_RELIABILITY.get(str(terrain_type_str), 0.1)

    def node_confidence(self, node_id: str) -> float:
        """c(v) = 1 - exp(-obs(v) * r(v)).

        Returns 0.0 if the node does not exist.
        """
        if node_id not in self.graph:
            return 0.0
        data = self.graph.nodes[node_id]
        obs = float(data.get('observation_count', 1))
        r   = self._terrain_reliability(data.get('terrain_type', 'UNKNOWN'))
        return float(1.0 - np.exp(-obs * r))

    _D_BAR_FLOOR: float = 5.0

    def mean_edge_length(self) -> float:
        """d_bar: median edge length of the graph (self-calibrating scale parameter).

        Uses the median rather than the mean so that many short edges added
        by dense drone exploration cannot drag the threshold down and cause
        the gap-edge rule to reject legitimate segments.

        A floor of ``_D_BAR_FLOOR`` (5 m) prevents collapse in very dense
        or small-scale graphs.  Falls back to 8.0 m when the graph has no
        edges at all.
        """
        lengths = [
            d['weight']
            for _, _, d in self.graph.edges(data=True)
            if d.get('weight') and d['weight'] > 0
        ]
        if not lengths:
            return 8.0
        return max(float(np.median(lengths)), self._D_BAR_FLOOR)

    def _find_corridor_nodes(self,
                              p1: np.ndarray,
                              p2: np.ndarray,
                              half_width: float) -> List[str]:
        """Return node IDs whose positions lie within half_width of segment p1→p2."""
        seg = p2 - p1
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            return []
        seg_dir = seg / seg_len

        # Broad spatial query centred on the segment midpoint.
        centre = (p1 + p2) / 2.0
        search_r = seg_len / 2.0 + half_width
        candidates = self.find_nearby_nodes(tuple(centre), search_r)

        if not candidates:
            return []
        node_ids = [node_id for node_id, _ in candidates]
        positions = np.asarray([self.graph.nodes[n]['position'] for n in node_ids])
        rel = positions - p1
        along = np.sum(rel * seg_dir, axis=1)
        perp = np.linalg.norm(rel - along[:, None] * seg_dir, axis=1)
        # Projection roundoff can put the destination node a few ulps beyond
        # seg_len. Endpoints belong to the corridor; losing the only observed
        # node here would incorrectly turn a known short segment into a gap.
        tolerance = 1e-9 * max(1.0, seg_len, half_width)
        inside = ((along >= -tolerance) & (along <= seg_len + tolerance)
                  & (perp <= half_width + tolerance))
        return [node_ids[i] for i in np.flatnonzero(inside)]

    def edge_confidence_from_positions(self,
                                        p1: Tuple[float, float],
                                        p2: Tuple[float, float]) -> float:
        """c(e) for the corridor between world positions p1 and p2.

        Formula:
            gap edge  (length > 2 * d_bar)  →  c = 0
            otherwise  c = 1 - exp( -(|N(e)| / expected_nodes) * mean_r )

        where expected_nodes = edge_length / d_bar and mean_r is the mean
        accumulated terrain evidence (observation_count * terrain reliability)
        of nodes found in the corridor.
        """
        return self._edge_confidence(p1, p2, self.mean_edge_length())

    def _edge_confidence(self, p1, p2, d_bar: float) -> float:
        """Evaluate corridor confidence with a precomputed graph scale."""
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)
        length = float(np.linalg.norm(p2_arr - p1_arr))

        if length > 2.0 * d_bar:
            # A long edge explicitly built from observed traversability is
            # not an artificial gap. Sparse sampling can make it longer
            # than twice the median without making it unobserved.
            first, second = self.find_nearest_node(p1), self.find_nearest_node(p2)
            observed_edge = (first is not None and second is not None
                             and first[1] < 1e-6 and second[1] < 1e-6
                             and self.graph.has_edge(first[0], second[0]))
            if not observed_edge:
                return 0.0

        corridor = self._find_corridor_nodes(p1_arr, p2_arr, d_bar / 2.0)
        if not corridor:
            return 0.0

        expected = max(1.0, length / d_bar)
        coverage = len(corridor) / expected
        mean_r   = float(np.mean([
            self._terrain_reliability(
                self.graph.nodes[nid].get('terrain_type', 'UNKNOWN'))
            * self.graph.nodes[nid].get('observation_count', 1)
            for nid in corridor
        ]))
        return float(1.0 - np.exp(-coverage * mean_r))

    def _uncertainty_weight_function(self, epsilon: float):
        """Recompute evidence weights for each search using checked geometry."""
        d_bar = self.mean_edge_length()
        geometry = self._corridor_geometry(d_bar)
        nodes = geometry['nodes']
        evidence = np.asarray([self._terrain_reliability(self.graph.nodes[node].get('terrain_type', 'UNKNOWN'))
                               * self.graph.nodes[node].get('observation_count', 1) for node in nodes])
        accumulated = geometry['incidence'] @ evidence
        confidence = (1. - np.exp(-accumulated / geometry['expected'])) * geometry['valid']
        values = geometry['lengths'] / np.maximum(confidence, epsilon)
        weights = {frozenset((first, second)): value
                   for (first, second, _), value in zip(geometry['edges'], values)}

        def weight(u, v, data):
            # This is an undirected graph, so both traversal directions share
            # one confidence calculation. The cache expires with this search.
            key = frozenset((u, v))
            return weights[key]

        return weight

    def path_confidence(self, waypoints: List[Tuple[float, float]]) -> float:
        """C(π) = min c(e) over all consecutive waypoint pairs (weakest-link).

        Returns 0.0 for paths shorter than two waypoints.
        """
        if len(waypoints) < 2:
            return 0.0
        confidences = [
            self.edge_confidence_from_positions(waypoints[i], waypoints[i + 1])
            for i in range(len(waypoints) - 1)
        ]
        return float(min(confidences))

    # ── Path cost & observation prediction (for C-ADMM dispatcher) ──────────

    def path_cost(self,
                  start_pos: Tuple[float, float],
                  goal_pos: Tuple[float, float],
                  epsilon: float = 0.05) -> float:
        """Uncertainty-weighted shortest-path cost from start to goal.

        Each edge cost = distance / max(c(e), epsilon), so low-confidence
        edges are expensive and gap edges are nearly impassable.

        Returns float('inf') when no graph path exists.
        """
        if self.graph.number_of_nodes() == 0:
            return float('inf')
        start = self.find_nearest_node(start_pos)
        goal = self.find_nearest_node(goal_pos)
        if start is None or goal is None:
            return float('inf')
        start_id, start_dist = start
        goal_id, goal_dist = goal

        if start_id == goal_id:
            return start_dist + goal_dist

        try:
            cost = nx.astar_path_length(
                self.graph, start_id, goal_id,
                weight=self._uncertainty_weight_function(epsilon))
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            return float('inf')

        off_graph_penalty = 1.0 / epsilon
        cost += start_dist * off_graph_penalty
        cost += goal_dist * off_graph_penalty
        return float(cost)

    def _corridor_geometry(self, scale):
        """Reuse geometry only when every position, edge and weight matches.

        Observation counts and terrain are always read afresh by searches.
        The signature also catches direct NetworkX edits made by callers.
        """
        from scipy.sparse import csr_matrix
        nodes = list(self.graph.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        positions = np.asarray([self.get_node_position(node) for node in nodes])
        edges = list(self.graph.edges(data=True))
        signature = (tuple((node, tuple(self.get_node_position(node))) for node in nodes),
                     tuple((first, second, data.get('weight', 0.)) for first, second, data in edges), scale)
        cached = getattr(self, '_corridor_geometry_cache', None)
        if cached is not None and cached[0] == signature:
            return cached[1]
        rows, columns, expected, lengths, endpoints, valid = [], [], [], [], [], []
        for edge_index, (first, second, data) in enumerate(edges):
            a, b = positions[index[first]], positions[index[second]]
            length = float(np.linalg.norm(b - a))
            corridor = self._find_corridor_nodes(a, b, scale / 2.)
            rows.extend([edge_index] * len(corridor))
            columns.extend(index[node] for node in corridor)
            expected.append(max(1., length / scale))
            weight = data.get('weight', 0.)
            lengths.append(weight if weight > 0 else length)
            endpoints.append((index[first], index[second]))
            permitted = True
            if length > 2. * scale:
                near_a, near_b = self.find_nearest_node(a), self.find_nearest_node(b)
                permitted = self.graph.has_edge(near_a[0], near_b[0])
            valid.append(permitted)
        incidence = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(edges), len(nodes)))
        geometry = dict(nodes=nodes, positions=positions, edges=edges, incidence=incidence,
                        expected=np.asarray(expected), lengths=np.asarray(lengths),
                        endpoints=np.asarray(endpoints, dtype=int).reshape(-1, 2), valid=np.asarray(valid))
        self._corridor_geometry_cache = (signature, geometry)
        return geometry

    def predicted_path_costs(self, centers, radius, start, goal, epsilon=0.05):
        """Batch count-only observations with exact uncertainty search weights."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        if not len(centers):
            return np.zeros(0)
        nodes = list(self.graph.nodes())
        if not nodes:
            return np.zeros(len(centers))
        index = {node: i for i, node in enumerate(nodes)}
        start_node, start_distance = self.find_nearest_node(start)
        goal_node, goal_distance = self.find_nearest_node(goal)
        start_index, goal_index = index[start_node], index[goal_node]
        if start_index == goal_index:
            return np.full(len(centers), start_distance + goal_distance)
        geometry = self._corridor_geometry(self.mean_edge_length())
        positions = geometry['positions']
        evidence = np.asarray([self._terrain_reliability(self.graph.nodes[node].get('terrain_type', 'UNKNOWN'))
                               for node in nodes])
        counts = np.asarray([self.graph.nodes[node].get('observation_count', 1) for node in nodes])
        centers = np.asarray(centers, dtype=float)
        observed = np.linalg.norm(positions[None, :, :] - centers[:, None, :], axis=2) <= radius
        accumulated = geometry['incidence'] @ ((counts[None, :] + observed) * evidence).T
        confidences = (1. - np.exp(-accumulated / geometry['expected'][:, None])) * geometry['valid'][:, None]
        weights = geometry['lengths'][:, None] / np.maximum(confidences, epsilon)
        ends = geometry['endpoints']
        adjacency_rows = np.concatenate((ends[:, 0], ends[:, 1]))
        adjacency_columns = np.concatenate((ends[:, 1], ends[:, 0]))
        remaining = np.linalg.norm(positions - np.asarray(goal), axis=1)
        costs = []
        for candidate in range(len(centers)):
            adjacency = csr_matrix((np.tile(weights[:, candidate], 2),
                                    (adjacency_rows, adjacency_columns)), shape=(len(nodes), len(nodes)))
            distances = dijkstra(adjacency, directed=False, indices=start_index)
            if np.isfinite(distances[goal_index]):
                cost = distances[goal_index] + (start_distance + goal_distance) / epsilon
            else:
                frontier_index = int(np.argmin(distances + remaining / epsilon))
                prefix = (start_distance if frontier_index == start_index else
                          distances[frontier_index] + start_distance / epsilon)
                cost = prefix + remaining[frontier_index] / epsilon
            costs.append(float(cost))
        return np.asarray(costs)

    def predict_observation(self,
                            center: Tuple[float, float],
                            fov_radius: float) -> 'TopologicalGraph':
        """Return a lightweight copy of the graph with boosted observation
        counts for nodes within *fov_radius* of *center*.

        Used by the dispatcher to estimate ``G'(w)`` without modifying the
        real graph.
        """
        import copy
        predicted = TopologicalGraph(drone_id=self.metadata.get('drone_id'))
        predicted.graph = copy.deepcopy(self.graph)
        predicted.metadata = dict(self.metadata)
        predicted.fusion_history = dict(self.fusion_history)

        nearby = self.find_nearby_nodes(center, fov_radius)
        for node_id, _dist in nearby:
            data = predicted.graph.nodes[node_id]
            data['observation_count'] = data.get('observation_count', 1) + 1
        return predicted

    def find_best_frontier(self,
                           start_pos: Tuple[float, float],
                           goal_pos: Tuple[float, float],
                           epsilon: float = 0.05,
                           ) -> Optional[Tuple[str, Tuple[float, float]]]:
        """Find the graph node that minimises total estimated cost to the goal.

        For each node *n* reachable from *start_pos* inside the same connected
        component, the score is::

            path_cost(start, n) + (1/epsilon) * euclidean(n, goal)

        The first term is the uncertainty-weighted cost through explored terrain.
        The second term is an optimistic estimate for the remaining unexplored
        segment (same penalty rate as off-graph traversal in ``path_cost``).

        Returns ``(node_id, node_position)`` of the best frontier node,
        or *None* when the graph is empty.
        """
        if self.graph.number_of_nodes() == 0:
            return None

        start = self.find_nearest_node(start_pos)
        if start is None:
            return None
        start_id, start_dist = start

        goal_arr = np.array(goal_pos)
        off_graph_penalty = 1.0 / epsilon

        # Single-source Dijkstra from the start node over the full graph.
        try:
            costs = nx.single_source_dijkstra_path_length(
                self.graph, start_id,
                weight=self._uncertainty_weight_function(epsilon))
        except nx.NetworkXError:
            return None

        best_node: Optional[str] = None
        best_total = float('inf')

        for nid, graph_cost in costs.items():
            pos = np.array(self.graph.nodes[nid]['position'])
            remaining = float(np.linalg.norm(pos - goal_arr))
            total = (graph_cost
                     + start_dist * off_graph_penalty
                     + remaining * off_graph_penalty)
            if total < best_total:
                best_total = total
                best_node = nid

        if best_node is None:
            return None
        return (best_node, tuple(self.graph.nodes[best_node]['position']))

    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"TopologicalGraph(nodes={self.graph.number_of_nodes()}, "
                f"edges={self.graph.number_of_edges()}, "
                f"drone_id={self.metadata.get('drone_id')})")
