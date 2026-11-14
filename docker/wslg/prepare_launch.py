#!/usr/bin/env python3
from copy import deepcopy
import os
from pathlib import Path
import xml.etree.ElementTree as ET

source = "/root/PX4_Firmware/launch/multi_vehicle_with_ground.launch"
tree = ET.parse(source)
root = tree.getroot()
count = int(os.environ.get("NUM_DRONES", "3"))
for group in list(root.findall("group")):
    namespace = group.get("ns", "")
    if namespace.startswith("iris_") and int(namespace.split("_")[-1]) >= count:
        root.remove(group)

if os.environ.get("PLANNING_MODE") == "v9-vtemp":
    # Translate intact tree models to extend the original forest along X.
    # Keep their physical scale and provide the trained fleet's 18 m spacing.
    px4 = Path(source).parents[1]
    world = ET.parse(px4 / "Tools/sitl_gazebo/worlds/forest.world")
    scene = world.getroot().find("world")
    trees = [model for model in scene.findall("include")
             if model.findtext("uri") in ("model://pine_tree", "model://oak_tree")]
    spawn_positions = [(0., 0.), (9., -18.), (9., 0.), (9., 18.)]
    for tree_model in trees:
        x, y, *_ = map(float, tree_model.findtext("pose").split())
        if any((x - sx)**2 + (y - sy)**2 < 6.**2 for sx, sy in spawn_positions):
            scene.remove(tree_model)
        for offset in (110, 220, 330, 440, 550, 660):
            if x + offset > 675.:
                continue
            copy = deepcopy(tree_model)
            copy.find("name").text += "_{}".format(offset)
            pose = copy.find("pose").text.split()
            pose[0] = str(x + offset)
            copy.find("pose").text = " ".join(pose)
            scene.append(copy)
    plane = scene.find("model[@name='ground_plane']")
    ET.SubElement(plane, "pose").text = "300 0 0 0 0 0"
    for size in plane.findall(".//plane/size"):
        size.text = "750 150"
    world.write("/tmp/teamed-v9-forest.world", encoding="utf-8", xml_declaration=True)
    root.find("arg[@name='world']").set("default", "/tmp/teamed-v9-forest.world")
    for index, (x, y) in enumerate(spawn_positions[1:]):
        vehicle = root.find("group[@ns='iris_{}']/include".format(index))
        vehicle.find("arg[@name='x']").set("value", str(x))
        vehicle.find("arg[@name='y']").set("value", str(y))
tree.write("/tmp/teamed-gazebo.launch", encoding="utf-8", xml_declaration=True)
