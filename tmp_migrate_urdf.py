#!/usr/bin/env python3
"""Migrate collision / mass / joint limits from refined OLD URDF into NEW CAD-zero URDF.

Preserve from NEW:
  - all joint origin xyz/rpy (CAD zero pose)
  - all joint axes (SolidWorks export)
  - visual meshes and package://URDFzhuangpei.SLDASM paths

Take from OLD (with frame transforms where needed):
  - Trunk calibrated inertial + primitive collisions
  - Hip primitive collisions (frames match)
  - Thigh/calf calibrated masses via scaling NEW CAD inertia
  - Thigh/calf primitive collisions rotated OLD->NEW about Y
  - Foot spheres + fixed joints (foot origin rotated into NEW calf frame)
  - Joint limit magnitudes (sign-flipped when NEW axis opposes OLD axis)
  - imu box collision, foot_rubber material, transmissions, gazebo plugin
"""
from __future__ import annotations

import copy
import math
import re
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

OLD_URDF = Path(r"C:\Users\26087\Desktop\urdf\URDFzhuangpei.SLDASM\urdf\URDFzhuangpei.SLDASM.urdf")
NEW_URDF = Path(r"E:\URDFzhuangpei.SLDASM\urdf\URDFzhuangpei.SLDASM.urdf")
BACKUP = NEW_URDF.with_suffix(".urdf.bak_before_migrate")
OUT = NEW_URDF

PKG = "URDFzhuangpei.SLDASM"

# Calibrated target masses from OLD refinement
TARGET_MASS = {
    "Trunk": 5.2,
    "thigh": 0.8,
    "calf": 0.14,
    "foot": 0.01,
}

# OLD limits relative to kneeling zero (same physical zero as NEW CAD assembly)
LIMITS_OLD = {
    "hip": dict(lower=-1.0472, upper=1.0472, effort=17.0, velocity=32.9867),
    "thigh": dict(lower=-2.76228694714, upper=2.29921305286, effort=17.0, velocity=32.9867),
    "calf": dict(lower=0.0, upper=1.91986217719, effort=17.0, velocity=32.9867),
}

# OLD thigh/calf axes were always +Y; hip was +X
OLD_AXIS = {
    "hip": np.array([1.0, 0.0, 0.0]),
    "thigh": np.array([0.0, 1.0, 0.0]),
    "calf": np.array([0.0, 1.0, 0.0]),
}

LEGS = ("FR", "FL", "RR", "RL")


def local(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_child(elem: ET.Element, name: str):
    for c in elem:
        if local(c.tag) == name:
            return c
    return None


def find_children(elem: ET.Element, name: str):
    return [c for c in elem if local(c.tag) == name]


def parse_xyz(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()], dtype=float)


def fmt_num(v: float) -> str:
    if abs(v) < 1e-15:
        return "0"
    s = f"{v:.12g}"
    return s


def fmt_xyz(v: np.ndarray) -> str:
    return " ".join(fmt_num(float(x)) for x in v)


def ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_y_align(src: np.ndarray, dst: np.ndarray) -> float:
    """Return theta such that Ry(theta) @ src ≈ dst (about Y; ignore y)."""
    a = math.atan2(src[0], src[2])
    b = math.atan2(dst[0], dst[2])
    return b - a


def rotate_rpy_add_pitch(rpy: np.ndarray, d_pitch: float) -> np.ndarray:
    # collision origins in this model only use pitch (or roll=pi/2 for cylinders)
    out = rpy.copy()
    out[1] += d_pitch
    return out


def scale_inertia(ixx, ixy, ixz, iyy, iyz, izz, ratio: float):
    return tuple(v * ratio for v in (ixx, ixy, ixz, iyy, iyz, izz))


def joint_role(name: str) -> str | None:
    if name.endswith("_hip_joint") and not name.endswith("_hip_joint_"):
        if name in {f"{leg}_hip_joint" for leg in LEGS}:
            return "hip"
    if name.endswith("_thigh_joint"):
        return "thigh"
    if name.endswith("_calf_joint"):
        return "calf"
    return None


def limit_for_axis(role: str, new_axis: np.ndarray) -> dict:
    base = LIMITS_OLD[role]
    old_ax = OLD_AXIS[role]
    # If axes are anti-parallel, flip limit signs (swap & negate)
    if abs(np.dot(new_axis, old_ax) + 1.0) < 1e-6:
        return dict(
            lower=-base["upper"],
            upper=-base["lower"],
            effort=base["effort"],
            velocity=base["velocity"],
        )
    # Parallel or unrelated (e.g. hip X vs Y): keep OLD numeric range
    return dict(base)


def get_link_map(root: ET.Element) -> dict[str, ET.Element]:
    return {e.attrib["name"]: e for e in root if local(e.tag) == "link"}


def get_joint_map(root: ET.Element) -> dict[str, ET.Element]:
    return {e.attrib["name"]: e for e in root if local(e.tag) == "joint"}


def read_inertial(link: ET.Element):
    inn = find_child(link, "inertial")
    origin = find_child(inn, "origin")
    mass = find_child(inn, "mass")
    inertia = find_child(inn, "inertia")
    com = parse_xyz(origin.attrib["xyz"])
    m = float(mass.attrib["value"])
    I = {k: float(inertia.attrib[k]) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    return com, m, I


def replace_inertial(link: ET.Element, com: np.ndarray, mass: float, I: dict):
    inn = find_child(link, "inertial")
    origin = find_child(inn, "origin")
    origin.attrib["xyz"] = fmt_xyz(com)
    origin.attrib["rpy"] = "0 0 0"
    find_child(inn, "mass").attrib["value"] = fmt_num(mass)
    inertia = find_child(inn, "inertia")
    for k, v in I.items():
        inertia.attrib[k] = fmt_num(v)


def clear_collisions(link: ET.Element):
    for c in list(link):
        if local(c.tag) == "collision":
            link.remove(c)


def append_collision_elems(link: ET.Element, collision_elems: list[ET.Element]):
    # Insert collisions after visual if present, else after inertial
    visual = find_child(link, "visual")
    if visual is not None:
        idx = list(link).index(visual) + 1
    else:
        inn = find_child(link, "inertial")
        idx = list(link).index(inn) + 1 if inn is not None else len(list(link))
    for i, col in enumerate(collision_elems):
        link.insert(idx + i, col)


def transform_collision(col: ET.Element, theta: float) -> ET.Element:
    """Rotate collision origin about Y by theta (OLD -> NEW)."""
    new_col = copy.deepcopy(col)
    R = ry(theta)
    origin = find_child(new_col, "origin")
    xyz = parse_xyz(origin.attrib.get("xyz", "0 0 0"))
    rpy = parse_xyz(origin.attrib.get("rpy", "0 0 0"))
    origin.attrib["xyz"] = fmt_xyz(R @ xyz)
    origin.attrib["rpy"] = fmt_xyz(rotate_rpy_add_pitch(rpy, theta))
    return new_col


def indent(elem: ET.Element, level: int = 0):
    """Rough pretty-print compatible with existing 2-space style."""
    pad = "\n" + "  " * level
    children = list(elem)
    if children:
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for i, child in enumerate(children):
            indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad + "  " if i < len(children) - 1 else pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def make_foot_link(name: str) -> ET.Element:
    link = ET.Element("link", {"name": name})
    inn = ET.SubElement(link, "inertial")
    ET.SubElement(inn, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inn, "mass", {"value": "0.01"})
    ET.SubElement(
        inn,
        "inertia",
        {
            "ixx": "9.0E-07",
            "ixy": "0",
            "ixz": "0",
            "iyy": "9.0E-07",
            "iyz": "0",
            "izz": "9.0E-07",
        },
    )
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    g = ET.SubElement(vis, "geometry")
    ET.SubElement(g, "sphere", {"radius": "0.015"})
    ET.SubElement(vis, "material", {"name": "foot_rubber"})
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    cg = ET.SubElement(col, "geometry")
    ET.SubElement(cg, "sphere", {"radius": "0.016"})
    return link


def make_foot_joint(leg: str, origin_xyz: np.ndarray) -> ET.Element:
    j = ET.Element("joint", {"name": f"{leg}_foot_fixed", "type": "fixed"})
    ET.SubElement(j, "origin", {"xyz": fmt_xyz(origin_xyz), "rpy": "0 0 0"})
    ET.SubElement(j, "parent", {"link": f"{leg}_calf_joint"})
    ET.SubElement(j, "child", {"link": f"{leg}_foot"})
    return j


def make_transmission(joint_name: str) -> ET.Element:
    prefix = joint_name.replace("_joint", "")
    t = ET.Element("transmission", {"name": f"{prefix}_transmission"})
    ET.SubElement(t, "type").text = "transmission_interface/SimpleTransmission"
    j = ET.SubElement(t, "joint", {"name": joint_name})
    ET.SubElement(j, "hardwareInterface").text = "hardware_interface/EffortJointInterface"
    a = ET.SubElement(t, "actuator", {"name": f"{prefix}_motor"})
    ET.SubElement(a, "mechanicalReduction").text = "1"
    return t


def main():
    shutil.copy2(NEW_URDF, BACKUP)
    print(f"Backup -> {BACKUP}")

    old_root = ET.parse(OLD_URDF).getroot()
    new_root = ET.parse(NEW_URDF).getroot()

    old_links = get_link_map(old_root)
    new_links = get_link_map(new_root)
    old_joints = get_joint_map(old_root)
    new_joints = get_joint_map(new_root)

    # --- header comment + foot material ---
    # ElementTree drops comments on write; we'll post-process the file text.
    # Insert foot_rubber material as first child if missing
    has_mat = any(
        local(c.tag) == "material" and c.attrib.get("name") == "foot_rubber"
        for c in new_root
    )
    if not has_mat:
        mat = ET.Element("material", {"name": "foot_rubber"})
        ET.SubElement(mat, "color", {"rgba": "0.08 0.08 0.08 1"})
        new_root.insert(0, mat)

    # --- Trunk: calibrated inertial + primitive collisions ---
    old_trunk = old_links["Trunk"]
    new_trunk = new_links["Trunk"]
    com, m, I = read_inertial(old_trunk)
    replace_inertial(new_trunk, com, m, I)
    clear_collisions(new_trunk)
    append_collision_elems(new_trunk, [copy.deepcopy(c) for c in find_children(old_trunk, "collision")])
    print("Trunk: calibrated mass/inertia + 3 box collisions")

    # --- Hip: keep NEW inertial (≈ identical), copy collisions ---
    for leg in LEGS:
        name = f"{leg}_hip_joint"
        clear_collisions(new_links[name])
        append_collision_elems(
            new_links[name],
            [copy.deepcopy(c) for c in find_children(old_links[name], "collision")],
        )
    print("Hips: primitive collisions copied")

    # --- imu: box collision ---
    clear_collisions(new_links["imu_Link"])
    append_collision_elems(
        new_links["imu_Link"],
        [copy.deepcopy(c) for c in find_children(old_links["imu_Link"], "collision")],
    )
    print("imu_Link: box collision")

    # --- Thigh / Calf: scale mass on NEW COM/inertia; rotate collisions ---
    foot_origins = {}
    for leg in LEGS:
        for role, target_mass in (("thigh", TARGET_MASS["thigh"]), ("calf", TARGET_MASS["calf"])):
            name = f"{leg}_{role}_joint"
            old_com, old_m, old_I = read_inertial(old_links[name])
            new_com, new_m, new_I = read_inertial(new_links[name])
            ratio = target_mass / new_m
            scaled = {k: v * ratio for k, v in new_I.items()}
            replace_inertial(new_links[name], new_com, target_mass, scaled)

            # Rotation taking OLD local vectors into NEW local frame:
            # Ry(theta) @ old_com_dir ≈ new_com_dir
            theta = rot_y_align(old_com, new_com)
            # Verify
            aligned = ry(theta) @ old_com
            err = np.linalg.norm(aligned - new_com)
            print(
                f"  {name}: mass {new_m:.6f}->{target_mass}, "
                f"Ry({theta:.6f}) align err={err:.6e}"
            )

            clear_collisions(new_links[name])
            transformed = [
                transform_collision(c, theta) for c in find_children(old_links[name], "collision")
            ]
            append_collision_elems(new_links[name], transformed)

            if role == "calf":
                # OLD foot fixed origin in OLD calf frame
                old_foot_j = old_joints[f"{leg}_foot_fixed"]
                old_foot_xyz = parse_xyz(find_child(old_foot_j, "origin").attrib["xyz"])
                foot_origins[leg] = ry(theta) @ old_foot_xyz
                print(f"  {leg}_foot origin NEW: {fmt_xyz(foot_origins[leg])}")

    # --- Joint limits (preserve NEW origin & axis) ---
    for jname, j in new_joints.items():
        if j.attrib.get("type") != "revolute":
            continue
        role = joint_role(jname)
        if role is None:
            continue
        axis_e = find_child(j, "axis")
        axis = parse_xyz(axis_e.attrib["xyz"])
        lim = limit_for_axis(role, axis)
        limit_e = find_child(j, "limit")
        limit_e.attrib["lower"] = fmt_num(lim["lower"])
        limit_e.attrib["upper"] = fmt_num(lim["upper"])
        limit_e.attrib["effort"] = fmt_num(lim["effort"])
        limit_e.attrib["velocity"] = fmt_num(lim["velocity"])
        print(
            f"limit {jname}: [{lim['lower']:.6g}, {lim['upper']:.6g}] "
            f"axis={fmt_xyz(axis)} (dot OLD={np.dot(axis, OLD_AXIS[role]):.3f})"
        )

    # --- Add feet after each calf joint block ---
    # Remove existing feet if any
    for leg in LEGS:
        for n in (f"{leg}_foot", f"{leg}_foot_fixed"):
            for child in list(new_root):
                if child.attrib.get("name") == n:
                    new_root.remove(child)

    # Insert foot link+joint immediately after each calf joint element
    children = list(new_root)
    for leg in LEGS:
        calf_joint_name = f"{leg}_calf_joint"
        # find index of calf joint
        idx = None
        for i, ch in enumerate(children):
            if local(ch.tag) == "joint" and ch.attrib.get("name") == calf_joint_name:
                idx = i
                break
        if idx is None:
            raise RuntimeError(f"missing {calf_joint_name}")
        foot_link = make_foot_link(f"{leg}_foot")
        foot_joint = make_foot_joint(leg, foot_origins[leg])
        # insert into tree after calf joint
        calf_joint_elem = children[idx]
        pos = list(new_root).index(calf_joint_elem)
        new_root.insert(pos + 1, foot_link)
        new_root.insert(pos + 2, foot_joint)
        children = list(new_root)
        print(f"added {leg}_foot")

    # --- transmissions + gazebo ---
    # Remove old ones if re-running
    for ch in list(new_root):
        if local(ch.tag) in ("transmission", "gazebo"):
            new_root.remove(ch)

    actuated = []
    for leg in LEGS:
        for role in ("hip", "thigh", "calf"):
            actuated.append(f"{leg}_{role}_joint")
    for jn in actuated:
        new_root.append(make_transmission(jn))

    gazebo = ET.Element("gazebo")
    plugin = ET.SubElement(
        gazebo,
        "plugin",
        {"name": "gazebo_ros_control", "filename": "libgazebo_ros_control.so"},
    )
    ET.SubElement(plugin, "robotNamespace").text = "/"
    new_root.append(gazebo)

    indent(new_root)
    xml_body = ET.tostring(new_root, encoding="unicode")
    # ET may emit ns junk; ensure declaration
    header = '''<?xml version="1.0" encoding="utf-8"?>
<!-- This URDF was automatically created by SolidWorks to URDF Exporter, then
     migrated with calibrated mass, primitive collisions, RS01 joint limits,
     and foot contact spheres from the refined dog_urdf model.
     CAD joint origins (zero pose) and visual meshes are preserved from the
     new SolidWorks export. Thigh/calf collision primitives and foot offsets
     were rotated about Y to match the new link frames. Joint axes follow the
     new export; limit signs are flipped when an axis is anti-parallel to the
     refined model. Effort=17 N.m (RS01 peak), velocity=32.9867 rad/s. -->
'''
    # Fix self-closing style preferences lightly
    text = header + xml_body + "\n"
    # Ensure package paths stayed correct (should already)
    text = text.replace("package://dog_urdf/", f"package://{PKG}/")

    OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT}")

    # sanity totals
    total = 0.0
    for name, link in get_link_map(ET.fromstring(text)).items():
        _, m, _ = read_inertial(link)
        total += m
        print(f"  mass {name}: {m}")
    print(f"Total mass: {total:.6f} kg")


if __name__ == "__main__":
    main()
