#!/usr/bin/env python3
from pathlib import Path
import re
import shutil

src = Path(r"C:\Users\26087\Desktop\urdf\URDFzhuangpei.SLDASM\config\rs01_motor_limits.yaml")
dst = Path(r"E:\URDFzhuangpei.SLDASM\config\rs01_motor_limits.yaml")
shutil.copy2(src, dst)
print("copied", dst, "size", dst.stat().st_size)

urdf = Path(r"E:\URDFzhuangpei.SLDASM\urdf\URDFzhuangpei.SLDASM.urdf")
text = urdf.read_text(encoding="utf-8")


def fix_hip(m: re.Match) -> str:
    block = m.group(0)
    return re.sub(r'<axis xyz="[^"]+" />', '<axis xyz="1 0 0" />', block, count=1)


new_text, n = re.subn(
    r'<joint name="(?:FR|FL|RR|RL)_hip_joint" type="revolute">.*?</joint>',
    fix_hip,
    text,
    flags=re.S,
)
print("hip joints fixed:", n)
urdf.write_text(new_text, encoding="utf-8")

for leg in ("FR", "FL", "RR", "RL"):
    m = re.search(
        rf'<joint name="{leg}_hip_joint" type="revolute">(.*?)</joint>',
        new_text,
        re.S,
    )
    axis = re.search(r'<axis xyz="([^"]+)"', m.group(1)).group(1)
    lim = re.search(r"<limit ([^/]+)/>", m.group(1)).group(1)
    print(leg, "axis", axis, lim)

# Update package.xml deps for gazebo control (keep package name)
pkg = Path(r"E:\URDFzhuangpei.SLDASM\package.xml")
pkg.write_text(
    """<?xml version="1.0"?>
<package format="2">
  <name>URDFzhuangpei.SLDASM</name>
  <version>1.0.0</version>
  <description>
    <p>URDF, collision meshes, and launch files for the custom quadruped robot.</p>
  </description>
  <author>TODO</author>
  <maintainer email="TODO@email.com" />
  <license>BSD</license>
  <buildtool_depend>catkin</buildtool_depend>
  <exec_depend>controller_manager</exec_depend>
  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>gazebo_ros_control</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>roslaunch</exec_depend>
  <exec_depend>rviz</exec_depend>
  <export>
    <architecture_independent />
  </export>
</package>
""",
    encoding="utf-8",
)
print("updated package.xml")
