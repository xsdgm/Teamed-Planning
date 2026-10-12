#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET

source = "/root/PX4_Firmware/launch/multi_vehicle_with_ground.launch"
tree = ET.parse(source)
root = tree.getroot()
count = int(os.environ.get("NUM_DRONES", "3"))
for group in list(root.findall("group")):
    namespace = group.get("ns", "")
    if namespace.startswith("iris_") and int(namespace.split("_")[-1]) >= count:
        root.remove(group)
tree.write("/tmp/teamed-gazebo.launch", encoding="utf-8", xml_declaration=True)
