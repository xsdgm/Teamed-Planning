"""Add ROS camera and CPU LiDAR to the pinned upstream Iris model."""
from pathlib import Path
import re

path = Path('/opt/PX4-Autopilot/Tools/sitl_gazebo/models/iris/iris.sdf.jinja')
text = path.read_text()
# Insert sensors into the existing base_link without changing flight dynamics.
match = re.search(r"<link\s+name=['\"]base_link['\"]>", text)
if match is None:
    raise RuntimeError('Pinned Iris model is missing base_link')
start = match.start()
end = text.index('</link>', start)
sensors = '''
      <sensor name="teamed_lidar" type="ray">
        <pose>0 0 0.2 0 0 0</pose>
        <update_rate>10</update_rate>
        <ray>
          <scan><horizontal><samples>360</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
          <vertical><samples>16</samples><resolution>1</resolution><min_angle>-1.4</min_angle><max_angle>0.1</max_angle></vertical></scan>
          <range><min>0.2</min><max>40</max><resolution>0.02</resolution></range>
        </ray>
        <plugin name="teamed_lidar_ros" filename="libgazebo_ros_velodyne_laser.so">
          <robotNamespace>/iris_{{ (mavlink_id | int) - 1 }}</robotNamespace>
          <topicName>velodyne_points</topicName><frameName>iris_{{ (mavlink_id | int) - 1 }}/base_link</frameName>
          <min_range>0.2</min_range><max_range>40</max_range><gaussianNoise>0.01</gaussianNoise>
        </plugin>
      </sensor>
      <sensor name="teamed_down_camera" type="camera">
        <pose>0 0 -0.1 0 1.5707963 0</pose><update_rate>5</update_rate>
        <camera><horizontal_fov>2.0944</horizontal_fov><image><width>512</width><height>512</height><format>R8G8B8</format></image><clip><near>0.1</near><far>100</far></clip></camera>
        <plugin name="teamed_camera_ros" filename="libgazebo_ros_camera.so">
          <robotNamespace>/iris_{{ (mavlink_id | int) - 1 }}</robotNamespace><cameraName>camera</cameraName>
          <imageTopicName>image_raw</imageTopicName><cameraInfoTopicName>camera_info</cameraInfoTopicName>
          <frameName>iris_{{ (mavlink_id | int) - 1 }}/camera_link</frameName>
        </plugin>
      </sensor>
'''
path.write_text(text[:end] + sensors + text[end:])
