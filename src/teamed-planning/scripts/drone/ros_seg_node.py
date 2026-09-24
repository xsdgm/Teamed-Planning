#!/usr/bin/env python3
"""
Segmentation ROS Node — one instance per drone.

Subscribes to the drone's downward RGB camera and runs SegFormer-B0 (ONNX)
semantic segmentation at its natural inference rate (~5 Hz).  Publishes the
result for drone_node to accumulate into the world-frame evidence grid.

Publishes:
  seg_image  (sensor_msgs/Image)       — coloured debug vis (green=free, red=obstacle)
  seg_occ    (nav_msgs/OccupancyGrid)  — raw per-frame binary occupancy in world frame

Encoding of seg_occ (custom, understood by drone_node._seg_occ_cb):
  grid.info.height = occ_h  (rows, world-X cells)
  grid.info.width  = occ_w  (cols, world-Y cells)
  grid.data[row*occ_w + col] = occ[row][col] * 100  (row-major, rows→X)
  grid.info.origin.position.{x,y} = (min_x, min_y) of footprint in world frame
Note: height/width are swapped vs standard OccupancyGrid convention intentionally
so that drone_node can recover the exact occ array without needing a custom message.
"""

import rospy
import numpy as np
import math
import os
import sys

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from drone.semantic_segmentation import SegFormerONNX


class SegROSNode:
    def __init__(self):
        rospy.init_node('seg_node', anonymous=False)

        # ── Parameters ──────────────────────────────────────────────────
        self._drone_id    = rospy.get_param('~drone_id',    'drone_0')
        self._vehicle_id  = rospy.get_param('~vehicle_id',  self._drone_id)
        self._camera_hfov = rospy.get_param('~camera_hfov', 2.0944)   # 120 deg default
        self._resolution  = rospy.get_param('~resolution',  0.5)
        self._spawn_offset = (
            float(rospy.get_param('~spawn_x', 0.0)),
            float(rospy.get_param('~spawn_y', 0.0)),
        )
        _model_path = rospy.get_param('~seg_model_path', '')

        # ── Load SegFormer ───────────────────────────────────────────────
        if _model_path and os.path.isfile(_model_path):
            try:
                self._segformer = SegFormerONNX(_model_path)
                rospy.loginfo(f"[{self._drone_id}/seg] SegFormer loaded: {_model_path}")
            except Exception as e:
                rospy.logwarn(f"[{self._drone_id}/seg] SegFormer load failed: {e} — zeros fallback")
                self._segformer = None
        else:
            if _model_path:
                rospy.logwarn(f"[{self._drone_id}/seg] Model not found: {_model_path} — zeros fallback")
            self._segformer = None

        # ── State ───────────────────────────────────────────────────────
        self._current_image: Image = None
        self._current_position    = None   # MAVROS local frame (x, y, z)
        self._current_orientation = None   # quaternion (x, y, z, w)
        self._last_seg_stamp      = None   # (secs, nsecs) of last processed frame
        self._cv_bridge = CvBridge()

        # ── Publishers ──────────────────────────────────────────────────
        self._seg_image_pub = rospy.Publisher('seg_image', Image,        queue_size=1)
        self._seg_occ_pub   = rospy.Publisher('seg_occ',   OccupancyGrid, queue_size=1)

        # ── Subscribers ─────────────────────────────────────────────────
        # Camera is at an absolute topic (different Gazebo namespace per drone).
        rospy.Subscriber(
            f'/{self._vehicle_id}/camera/image_raw',
            Image, self._image_cb, queue_size=1, buff_size=2**24)
        # Odom is remapped in the launch file to the correct MAVROS topic.
        rospy.Subscriber('odom', Odometry, self._odom_cb)

        # Timer: fires at 10 Hz; inference naturally limits effective rate to ~5 Hz.
        rospy.Timer(rospy.Duration(0.1), self._seg_loop)

        rospy.loginfo(f"[{self._drone_id}/seg] Node ready "
                      f"(model={'loaded' if self._segformer else 'MISSING'})")

    # ════════════════════════ Callbacks ══════════════════════════════

    def _image_cb(self, msg: Image):
        self._current_image = msg

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._current_position = (p.x, p.y, p.z)
        o = msg.pose.pose.orientation
        self._current_orientation = (o.x, o.y, o.z, o.w)

    # ════════════════════════ Helpers ════════════════════════════════

    @staticmethod
    def _quat_to_yaw(q) -> float:
        x, y, z, w = q
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # ════════════════════════ Segmentation loop ══════════════════════

    def _seg_loop(self, _event):
        if self._current_image is None or self._current_position is None:
            return

        # Skip identical frame (avoids double-processing the same image)
        stamp = self._current_image.header.stamp
        cur_stamp = (stamp.secs, stamp.nsecs)
        if cur_stamp == self._last_seg_stamp:
            return
        self._last_seg_stamp = cur_stamp

        img_msg = self._current_image   # snapshot to avoid mid-process update
        pos     = self._current_position

        # Convert MAVROS-local position to shared map frame
        pos_map = (
            pos[0] + self._spawn_offset[0],
            pos[1] + self._spawn_offset[1],
            pos[2],
        )
        altitude = max(pos[2], 1.0)

        try:
            import cv2
            raw = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, -1)
            if img_msg.encoding in ('bgr8', 'bgra8'):
                rgb = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB)
            else:
                rgb = np.ascontiguousarray(raw[..., :3])

            if self._segformer is not None:
                occ, _ = self._segformer.segment(rgb, altitude, self._camera_hfov)

                # Yaw correction: un-rotate so the occupancy map is axis-aligned
                # with the world frame (rows→+X, cols→+Y regardless of heading).
                if self._current_orientation is not None:
                    yaw = self._quat_to_yaw(self._current_orientation)
                    if abs(yaw) > 0.02:
                        occ_h, occ_w = occ.shape
                        cx, cy = occ_w / 2.0, occ_h / 2.0
                        deg = math.degrees(-yaw)
                        M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
                        occ = cv2.warpAffine(
                            occ.astype(np.float32), M, (occ_w, occ_h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=1.0,   # corners outside original image → obstacle
                        ).astype(np.uint8)

                # Debug: coloured segmentation image
                try:
                    vis = np.zeros((*occ.shape, 3), dtype=np.uint8)
                    vis[occ == 0] = (0,   200,  80)   # green — traversable
                    vis[occ == 1] = (220,  40,  40)   # red   — obstacle
                    dbg = self._cv_bridge.cv2_to_imgmsg(vis, encoding='rgb8')
                    dbg.header.stamp = rospy.Time.now()
                    dbg.header.frame_id = self._drone_id
                    self._seg_image_pub.publish(dbg)
                except Exception:
                    pass

            else:
                # No model: zero occupancy sized from altitude + HFOV + image aspect
                aspect = img_msg.height / max(img_msg.width, 1)
                ground_w = 2.0 * altitude * math.tan(self._camera_hfov / 2.0)
                occ_w = max(4, int(round(ground_w / self._resolution)))
                occ_h = max(4, int(round(ground_w * aspect / self._resolution)))
                occ = np.zeros((occ_h, occ_w), dtype=np.uint8)
                # Pass raw camera image to RViz so the topic is not empty
                self._seg_image_pub.publish(img_msg)

            self._publish_seg_occ(occ, pos_map)

        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[{self._drone_id}/seg] Error: {e}")

    def _publish_seg_occ(self, occ: np.ndarray, pos_map: tuple):
        """Encode and publish the raw per-frame occupancy for drone_node.

        Encoding (rows→world-X, cols→world-Y, direct row-major):
          height = occ_h,  width = occ_w
          data[row * occ_w + col] = occ[row][col] * 100
          origin = (min_x, min_y)
        """
        occ_h, occ_w = occ.shape
        res    = self._resolution
        half_x = occ_h * res / 2.0   # rows → world X span
        half_y = occ_w * res / 2.0   # cols → world Y span

        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = 'map'
        grid.info.resolution = res
        grid.info.height = occ_h    # rows → X cells (swapped vs OccupancyGrid convention)
        grid.info.width  = occ_w    # cols → Y cells
        grid.info.origin.position.x = pos_map[0] - half_x   # min_x
        grid.info.origin.position.y = pos_map[1] - half_y   # min_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        # Direct row-major storage — NO flipud/T (drone_node knows the encoding)
        grid.data = (occ.flatten().astype(np.int8) * 100).tolist()
        self._seg_occ_pub.publish(grid)


if __name__ == '__main__':
    try:
        SegROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
