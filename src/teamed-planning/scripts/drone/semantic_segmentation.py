"""
Drone Perception Module
Processes semantic segmentation data from top-down camera view.

Classes:
  TerrainType       — terrain category enum
  SemanticProcessor — converts occupancy maps to semantic maps (planning use)
  SegFormerONNX     — runs SegFormer-B0 (ADE20K) ONNX model on RGB images
"""

import os
import math
import numpy as np
from typing import Dict, Tuple, Optional
from enum import IntEnum


class TerrainType(IntEnum):
    """Terrain types from semantic segmentation."""
    UNKNOWN = 0
    TRAVERSABLE = 1
    OBSTACLE = 2
    DIFFICULT = 3  # Difficult terrain (dense vegetation, rough ground)
    WATER = 4
    CLEARNING = 5  # Open areas, easier to traverse


class SemanticProcessor:
    """
    Processes semantic segmentation data into traversability information.
    
    In the first version, this receives ground truth semantic data from simulator.
    Later versions can integrate actual segmentation networks.
    """
    
    def __init__(self, resolution: float = 0.5):
        """
        Initialize semantic processor.
        
        Args:
            resolution: Resolution of the semantic map in meters per pixel
        """
        self.resolution = resolution
        self.terrain_costs = {
            TerrainType.UNKNOWN: 1.0,
            TerrainType.TRAVERSABLE: 1.0,
            TerrainType.OBSTACLE: float('inf'),
            TerrainType.DIFFICULT: 2.5,
            TerrainType.WATER: float('inf'),
            TerrainType.CLEARNING: 0.5
        }
        
    def process_occupancy_map(self, occupancy_map: np.ndarray) -> np.ndarray:
        """
        Transform binary occupancy map to semantic map.
        
        This method adapts raw sensor data (occupancy grid) to semantic representation.
        In simulation: transforms binary 0/1 map.
        In real-world: would transform depth/RGB data via segmentation network.
        
        Args:
            occupancy_map: Binary occupancy grid (0=free, 1=occupied)
            
        Returns:
            semantic_map: Map with TerrainType values (TRAVERSABLE or OBSTACLE)
        """
        semantic_map = np.where(
            occupancy_map == 1,
            TerrainType.OBSTACLE.value,
            TerrainType.TRAVERSABLE.value
        )
        return semantic_map
    
    def process_segmentation(self, semantic_map: np.ndarray, 
                           drone_position: Tuple[float, float, float]) -> Dict:
        """
        Process raw semantic segmentation map.
        
        Args:
            semantic_map: 2D array of terrain types (HxW)
            drone_position: (x, y, z) position of drone in world frame
            
        Returns:
            Dictionary containing processed semantic information
        """
        height, width = semantic_map.shape

        # Camera axis → world axis mapping (downward camera, pitch=+90°, yaw=0):
        #   image rows  (shape[0]=height, image top→drone fwd) → world X axis
        #   image cols  (shape[1]=width,  image left→drone left) → world Y axis
        # So the X range spans image height, and Y range spans image width.
        half_x = (height * self.resolution) / 2.0   # rows → X
        half_y = (width  * self.resolution) / 2.0   # cols → Y

        min_x = drone_position[0] - half_x
        max_x = drone_position[0] + half_x
        min_y = drone_position[1] - half_y
        max_y = drone_position[1] + half_y

        coverage_bounds = (min_x, min_y, max_x, max_y)
        
        return {
            'semantic_map': semantic_map,
            'resolution': self.resolution,
            'coverage_bounds': coverage_bounds,
            'drone_position': drone_position,
            'map_shape': semantic_map.shape
        }
    
    def get_terrain_type(self, semantic_map: np.ndarray, 
                        pixel_x: int, pixel_y: int) -> TerrainType:
        """
        Get terrain type at a specific pixel location.
        
        Args:
            semantic_map: Semantic segmentation map
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            
        Returns:
            TerrainType at that location
        """
        if (0 <= pixel_y < semantic_map.shape[0] and 
            0 <= pixel_x < semantic_map.shape[1]):
            return TerrainType(semantic_map[pixel_y, pixel_x])
        return TerrainType.UNKNOWN
    
    def get_traversability_cost(self, terrain_type: TerrainType) -> float:
        """
        Get traversability cost for a terrain type.
        
        Args:
            terrain_type: Type of terrain
            
        Returns:
            Cost value (1.0 = normal, higher = harder, inf = impossible)
        """
        return self.terrain_costs.get(terrain_type, 1.0)
    
    def world_to_pixel(self, world_pos: Tuple[float, float],
                      coverage_bounds: Tuple[float, float, float, float],
                      map_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Convert world coordinates to pixel coordinates.

        Axis mapping (downward camera, pitch=+90°, drone faces +X world):
            world X → image row  (shape[0] = height)
            world Y → image col  (shape[1] = width)
        Returns (col, row) so callers using semantic_map[row, col] work correctly.

        Args:
            world_pos: (x, y) position in world frame
            coverage_bounds: (min_x, min_y, max_x, max_y) bounds
            map_shape: (height, width) of the map

        Returns:
            (col, row) = (pixel_x, pixel_y) or None if out of bounds
        """
        min_x, min_y, max_x, max_y = coverage_bounds
        height, width = map_shape

        row = int((world_pos[0] - min_x) / self.resolution)   # world X → row
        col = int((world_pos[1] - min_y) / self.resolution)   # world Y → col

        if 0 <= row < height and 0 <= col < width:
            return (col, row)   # (pixel_x, pixel_y) convention for callers
        return None

    def pixel_to_world(self, pixel_pos: Tuple[int, int],
                      coverage_bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Convert pixel coordinates to world coordinates.

        Axis mapping (downward camera, pitch=+90°, drone faces +X world):
            col (pixel_x) → world Y
            row (pixel_y) → world X

        Args:
            pixel_pos: (col, row) = (pixel_x, pixel_y) coordinates
            coverage_bounds: (min_x, min_y, max_x, max_y) bounds

        Returns:
            (x, y) position in world frame
        """
        min_x, min_y, max_x, max_y = coverage_bounds
        col, row = pixel_pos   # pixel_x=col, pixel_y=row

        world_x = min_x + (row + 0.5) * self.resolution   # row → X (+0.5 for center)
        world_y = min_y + (col + 0.5) * self.resolution   # col → Y

        return (world_x, world_y)
    
    def compute_traversability_map(self, semantic_map: np.ndarray) -> np.ndarray:
        """
        Convert semantic map to traversability cost map.
        
        Args:
            semantic_map: 2D array of terrain types
            
        Returns:
            2D array of traversability costs
        """
        traversability_map = np.zeros_like(semantic_map, dtype=float)
        
        for terrain_type in TerrainType:
            mask = semantic_map == terrain_type.value
            traversability_map[mask] = self.get_traversability_cost(terrain_type)

        return traversability_map


# ══════════════════════════════════════════════════════════════════════════════
# SegFormer-B0 ONNX inference
# ══════════════════════════════════════════════════════════════════════════════

class SegFormerONNX:
    """
    Runs SegFormer-B0 (finetuned on ADE20K) via ONNX runtime and converts
    the 150-class segmentation output to a binary occupancy map
    (0 = traversable, 1 = obstacle) suitable for SkeletonGraphBuilder.

    Model:   nvidia/segformer-b0-finetuned-ade-512-512
    Input:   [1, 3, 512, 512]  float32, ImageNet-normalised
    Output:  [1, 150, 128, 128] logits  (upsampled internally)

    ADE20K traversability mapping (forest environment):
      TRAVERSABLE  — grass(3), dirt(9), earth(13), sand(46), path(52), road(53)
      OBSTACLE     — tree(4), rock(17), water(21), hill(26), fence(29), wall(30)
      UNKNOWN      — everything else → treated as traversable (conservative)
    """

    # ADE20K class IDs (0-indexed, matching nvidia/segformer-b0-finetuned-ade-512-512)
    # Strategy: unknown = OBSTACLE (conservative for forest navigation)
    _TRAVERSABLE_IDS = {
        3,   # floor/flooring   (rare outdoors but harmless)
        6,   # road/route
        9,   # grass
        13,  # earth/ground
        29,  # field            (open terrain)
        46,  # sand
        52,  # path
        91,  # dirt track
        94,  # land
    }
    _OBSTACLE_IDS = {
        4,   # tree
        16,  # mountain
        17,  # plant/flora/vegetation
        21,  # water
        26,  # sea
        32,  # fence
        34,  # rock
        60,  # river
        68,  # hill
        72,  # palm
        128, # lake
    }

    # ImageNet normalisation constants (RGB order)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Model input / output sizes
    _INPUT_SIZE  = 512   # px
    _LOGIT_SIZE  = 128   # px  (model output before upsampling)

    def __init__(self, model_path: str, output_resolution: float = 0.5):
        """
        Args:
            model_path:        Path to model.onnx
            output_resolution: Desired occupancy-map resolution in m/px.
                               Used by callers to set SkeletonGraphBuilder.resolution.
        """
        import onnxruntime as ort  # imported lazily so the module loads without GPU
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"SegFormer ONNX model not found: {model_path}\n"
                "Run scripts/drone/models/download_models.sh first.")

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self.output_resolution = output_resolution

    # ── Public API ────────────────────────────────────────────────────────────

    def segment(self,
                image_rgb: np.ndarray,
                altitude: float,
                hfov_rad: float = 2.0944) -> Tuple[np.ndarray, float]:
        """
        Run segmentation on a downward-facing camera image.

        Args:
            image_rgb:  H×W×3 uint8 RGB image from the drone's camera.
            altitude:   Drone altitude above ground in metres (from MAVROS z).
            hfov_rad:   Horizontal field-of-view in radians (default 120° = 2.0944).

        Returns:
            occupancy_map:  2D numpy array uint8, shape (map_h, map_w).
                            0 = free / traversable, 1 = obstacle.
            pixel_size_m:   Ground size of each map pixel in metres.
                            Pass this as `resolution` to SkeletonGraphBuilder.
        """
        import cv2

        img_h, img_w = image_rgb.shape[:2]

        # ── 1. Pre-process: pad to square to preserve aspect ratio ───────
        # Squishing W×H → 512×512 distorts features (trees look wrong).
        # Instead, zero-pad the shorter axis so the image becomes square,
        # then resize the square to INPUT_SIZE.  The padded border is black
        # (0,0,0) which will be classified as obstacle — safe default.
        if img_h != img_w:
            pad_size = max(img_h, img_w)
            padded = np.zeros((pad_size, pad_size, 3), dtype=np.uint8)
            # Centre the image in the padded canvas
            y_off = (pad_size - img_h) // 2
            x_off = (pad_size - img_w) // 2
            padded[y_off:y_off + img_h, x_off:x_off + img_w] = image_rgb
        else:
            padded = image_rgb
            y_off = x_off = 0

        img = cv2.resize(padded, (self._INPUT_SIZE, self._INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD                    # HWC, normalised
        inp = img.transpose(2, 0, 1)[np.newaxis, ...]           # NCHW

        # ── 2. ONNX inference ─────────────────────────────────────────────
        logits = self._session.run([self._output_name], {self._input_name: inp})[0]
        # logits: [1, 150, 128, 128]

        # ── 3. Argmax → class map (128×128) ──────────────────────────────
        class_map = logits[0].argmax(axis=0).astype(np.uint8)   # (128, 128)

        # ── 4. Map ADE20K IDs to occupancy ───────────────────────────────
        # Conservative forest strategy: unknown class = OBSTACLE.
        # Only pixels explicitly in _TRAVERSABLE_IDS are marked free (0).
        occupancy = np.ones_like(class_map, dtype=np.uint8)     # default: obstacle
        for cls_id in self._TRAVERSABLE_IDS:
            occupancy[class_map == cls_id] = 0

        # ── 5. Crop padded border out of class map ────────────────────────
        # The padding added black borders that get classified as obstacle.
        # Crop them back out so the occupancy map covers only the real image.
        pad_size = max(img_h, img_w)
        # Fraction of the padded canvas occupied by the real image
        crop_y0 = int(round(y_off * self._LOGIT_SIZE / pad_size))
        crop_x0 = int(round(x_off * self._LOGIT_SIZE / pad_size))
        crop_y1 = int(round((y_off + img_h) * self._LOGIT_SIZE / pad_size))
        crop_x1 = int(round((x_off + img_w) * self._LOGIT_SIZE / pad_size))
        crop_y1 = max(crop_y1, crop_y0 + 1)
        crop_x1 = max(crop_x1, crop_x0 + 1)
        occupancy = occupancy[crop_y0:crop_y1, crop_x0:crop_x1]

        # ── 6. Resize to desired output resolution ────────────────────────
        # Ground coverage of the full image at this altitude (pinhole model):
        #   ground_width_m  = 2 * altitude * tan(hfov / 2)
        #   ground_height_m = 2 * altitude * tan(vfov / 2)
        #   vfov = 2 * arctan(tan(hfov/2) * img_h / img_w)
        # Equivalently: ground_height_m = ground_width_m * (img_h / img_w)
        ground_width_m  = 2.0 * altitude * math.tan(hfov_rad / 2.0)
        ground_height_m = ground_width_m * (img_h / img_w)

        map_w = max(1, int(round(ground_width_m  / self.output_resolution)))
        map_h = max(1, int(round(ground_height_m / self.output_resolution)))

        # cv2.resize dsize = (width, height)
        occupancy_resized = cv2.resize(occupancy, (map_w, map_h),
                                       interpolation=cv2.INTER_NEAREST)

        pixel_size_m = ground_width_m / map_w
        return occupancy_resized, pixel_size_m
