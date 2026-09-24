"""
Message Definitions for Multi-Robot Communication
Defines standardized message formats for graph transmission and robot communication
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import json


class MessageType(Enum):
    """Types of messages in the system."""
    GRAPH_BROADCAST = "graph_broadcast"
    POSITION_UPDATE = "position_update"
    GOAL_REQUEST = "goal_request"
    ACK = "acknowledgment"


@dataclass
class Message:
    """Base message class."""
    sender_id: str
    message_type: MessageType
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        return data
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class GraphBroadcastMessage(Message):
    """
    Message containing a topological graph from a drone.
    
    Attributes:
        sender_id: ID of the drone sending the graph
        graph_data: Serialized graph data (from TopologicalGraph.to_message())
        coverage_bounds: (min_x, min_y, max_x, max_y) bounds of the mapped area
        node_count: Number of nodes in the graph
        edge_count: Number of edges in the graph
    """
    graph_data: Dict = field(default_factory=dict)
    coverage_bounds: Optional[tuple] = None
    node_count: int = 0
    edge_count: int = 0
    
    def __init__(self, sender_id: str, graph_data: Dict, 
                 coverage_bounds: Optional[tuple] = None,
                 node_count: int = 0, edge_count: int = 0):
        super().__init__(sender_id, MessageType.GRAPH_BROADCAST)
        self.graph_data = graph_data
        self.coverage_bounds = coverage_bounds
        self.node_count = node_count
        self.edge_count = edge_count
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GraphBroadcastMessage':
        """Create message from dictionary."""
        return cls(
            sender_id=data['sender_id'],
            graph_data=data['graph_data'],
            coverage_bounds=data.get('coverage_bounds'),
            node_count=data.get('node_count', 0),
            edge_count=data.get('edge_count', 0)
        )


@dataclass
class PositionUpdateMessage(Message):
    """
    Message containing position update from a robot.
    
    Attributes:
        sender_id: ID of the robot
        position: (x, y) or (x, y, z) position
        heading: Orientation in radians (optional)
    """
    position: tuple = field(default_factory=tuple)
    heading: Optional[float] = None
    
    def __init__(self, sender_id: str, position: tuple, 
                 heading: Optional[float] = None):
        super().__init__(sender_id, MessageType.POSITION_UPDATE)
        self.position = position
        self.heading = heading
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PositionUpdateMessage':
        """Create message from dictionary."""
        return cls(
            sender_id=data['sender_id'],
            position=tuple(data['position']),
            heading=data.get('heading')
        )


@dataclass
class AckMessage(Message):
    """
    Acknowledgment message.
    
    Attributes:
        sender_id: ID of the robot sending acknowledgment
        ack_message_id: ID/timestamp of the message being acknowledged
        status: Status of the acknowledgment (success, partial, failed)
    """
    ack_message_id: float = 0.0
    status: str = "success"
    
    def __init__(self, sender_id: str, ack_message_id: float, 
                 status: str = "success"):
        super().__init__(sender_id, MessageType.ACK)
        self.ack_message_id = ack_message_id
        self.status = status
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AckMessage':
        """Create message from dictionary."""
        return cls(
            sender_id=data['sender_id'],
            ack_message_id=data['ack_message_id'],
            status=data.get('status', 'success')
        )


class MessageParser:
    """Utility class for parsing messages."""
    
    @staticmethod
    def parse(json_string: str) -> Optional[Message]:
        """
        Parse JSON string into appropriate message type.
        
        Args:
            json_string: JSON string containing message data
            
        Returns:
            Message object or None if parsing fails
        """
        try:
            data = json.loads(json_string)
            msg_type = data.get('message_type')
            
            if msg_type == MessageType.GRAPH_BROADCAST.value:
                return GraphBroadcastMessage.from_dict(data)
            elif msg_type == MessageType.POSITION_UPDATE.value:
                return PositionUpdateMessage.from_dict(data)
            elif msg_type == MessageType.ACK.value:
                return AckMessage.from_dict(data)
            else:
                return None
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing message: {e}")
            return None
    
    @staticmethod
    def create_graph_message(sender_id: str, graph_data: Dict,
                           coverage_bounds: Optional[tuple] = None,
                           node_count: int = 0, 
                           edge_count: int = 0) -> GraphBroadcastMessage:
        """
        Convenience method to create a graph broadcast message.
        
        Args:
            sender_id: Drone ID
            graph_data: Serialized graph from TopologicalGraph.to_message()
            coverage_bounds: Coverage area bounds
            node_count: Number of nodes
            edge_count: Number of edges
            
        Returns:
            GraphBroadcastMessage instance
        """
        return GraphBroadcastMessage(
            sender_id=sender_id,
            graph_data=graph_data,
            coverage_bounds=coverage_bounds,
            node_count=node_count,
            edge_count=edge_count
        )


class SimpleMessageBroker:
    """
    Simple message broker for simulator environment.
    This will be replaced with ROS topics in the real system.
    """
    
    def __init__(self):
        """Initialize message broker."""
        self.message_queues: Dict[str, List[Dict]] = {}  # {receiver_id: [messages]}
        self.broadcast_queue: List[Dict] = []  # Messages to all robots
        
    def publish(self, message: Message, recipients: Optional[List[str]] = None):
        """
        Publish a message to specific recipients or broadcast to all.
        
        Args:
            message: Message object to send
            recipients: List of recipient IDs, or None for broadcast
        """
        message_dict = message.to_dict()
        
        if recipients is None:
            # Broadcast to all
            self.broadcast_queue.append(message_dict)
        else:
            # Send to specific recipients
            for recipient_id in recipients:
                if recipient_id not in self.message_queues:
                    self.message_queues[recipient_id] = []
                self.message_queues[recipient_id].append(message_dict)
    
    def receive(self, receiver_id: str) -> List[Message]:
        """
        Receive all messages for a specific robot.
        
        Args:
            receiver_id: ID of the receiving robot
            
        Returns:
            List of Message objects
        """
        messages = []
        
        # Get direct messages
        if receiver_id in self.message_queues:
            for msg_dict in self.message_queues[receiver_id]:
                msg = MessageParser.parse(json.dumps(msg_dict))
                if msg:
                    messages.append(msg)
            self.message_queues[receiver_id] = []  # Clear queue
        
        # Get broadcast messages
        for msg_dict in self.broadcast_queue:
            # Don't send message back to sender
            if msg_dict['sender_id'] != receiver_id:
                msg = MessageParser.parse(json.dumps(msg_dict))
                if msg:
                    messages.append(msg)
        
        return messages
    
    def clear_broadcasts(self):
        """Clear broadcast queue (call this after all robots have received)."""
        self.broadcast_queue = []
    
    def get_pending_count(self, receiver_id: str) -> int:
        """Get number of pending messages for a receiver."""
        count = len(self.message_queues.get(receiver_id, []))
        count += len(self.broadcast_queue)  # Broadcasts available to all
        return count
