import os
import bpy
import numpy as np
from bpy_extras.object_utils import *
from bpy_extras.view3d_utils import *
from mathutils import *
from scipy.spatial import cKDTree

LIST = []
Array_Acupoints = []
collection_names = ["acu", "acu_1"]

def sort(obj_main):

    global LIST
    global Array_Acupoints

    world_matrix = obj_main.matrix_world
    # mesh collection
    array_mesh = obj_main.data.polygons
    # mesh origin collection
    array_mesh_coordinates = np.array([world_matrix @ mesh.center for mesh in array_mesh])

    # Array_Acupoints = []
    for collection_name in collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            for subcollection in collection.children:
                for acupoint in subcollection.all_objects:
                    if acupoint.type == 'MESH':
                        Array_Acupoints.append(acupoint)

    # acupoints collection
    array_acupoints_coordinates = np.array([acupoint.location for acupoint in Array_Acupoints])
    # KD-Tree of clooection
    tree = cKDTree(array_mesh_coordinates)
    # KD-Tree
    distances, indices = tree.query(array_acupoints_coordinates, k=1)
    for i, idx in enumerate(indices):
        nearest_mesh_normal_vector = array_mesh[idx].normal
        LIST.append(array_mesh[idx])
        # print(f"name: {Array_Acupoints[i].name} | normal： {nearest_mesh_normal_vector}")

def get_normal_in_camera_viewport(camera, obj, frame_num):

    # M_cam
    camera_world_matrix = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    frame_camera_normal = [camera_world_matrix.x, camera_world_matrix.y, camera_world_matrix.z]

    # if frame_num == 37:
    # print(f"{frame_num} -> {frame_camera_normal}")

    if obj in Array_Acupoints and len(LIST) >= Array_Acupoints.index(obj):
        normal_vector = LIST[Array_Acupoints.index(obj)].normal
        frame_obj_to_camera_normal = [normal_vector.x, normal_vector.y, normal_vector.z]
        # print(f"closest normal：{frame_obj_to_camera_normal}")
    else:
        frame_obj_to_camera_normal = [0, 0, 0]
        print("not in list")

    return frame_camera_normal, frame_obj_to_camera_normal


def get_bounding_box_2d_in_camera_viewport(camera, obj, frame_num):

    # depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()

    frame = bpy.context.scene.frame_current
    camera_eval = camera.evaluated_get(depsgraph)
    para = 2048

    obj_eval = obj.evaluated_get(depsgraph)
    obj_to_cam = camera_eval.matrix_world.inverted() @ obj_eval.matrix_world
    bbox_corners = [obj_to_cam @ Vector(corner) for corner in obj_eval.bound_box]

    normalized_coords = [(corner / corner.z)[:2] for corner in bbox_corners]

    x_min = min([coord[0] for coord in normalized_coords])
    x_max = max([coord[0] for coord in normalized_coords])
    y_min = min([coord[1] for coord in normalized_coords])
    y_max = max([coord[1] for coord in normalized_coords])
    x_min_percent = (x_min + 1) / 2
    x_max_percent = (x_max + 1) / 2
    y_min_percent = (y_min + 1) / 2
    y_max_percent = (y_max + 1) / 2

    distance = abs(1000 * (camera.matrix_world.translation - obj.matrix_world.translation).dot(camera.matrix_world.to_4x4().inverted().to_3x3().col[2]))

    return x_max_percent, x_min_percent, y_max_percent, y_min_percent, distance

for obj_main in bpy.context.scene.objects:
    # if MESH
    if obj_main.type == "MESH" and ("mesh" in obj_main.name.lower() or "Mesh" in obj_main.name.lower()):
        # print(f"{obj_main.name}: {obj_main.type}")
        sort(obj_main)

for obj_temp in bpy.context.scene.objects:
    # get camera
    if obj_temp.type == "CAMERA":
        camera = obj_temp

# main loop

for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)

    # Initialize output string for this frame
    output_str = f"frame_{frame}:\n"

    # Iterate over all objects in scene
    for obj in Array_Acupoints:

        screen_coords = get_bounding_box_2d_in_camera_viewport(camera, obj, frame)
        _2d_coords = get_normal_in_camera_viewport(camera, obj, frame)

        dot_product = np.dot(_2d_coords[0], _2d_coords[1])
        magnitude_A = np.linalg.norm(_2d_coords[0])
        magnitude_B = np.linalg.norm(_2d_coords[1])
        _C = dot_product / (magnitude_A * magnitude_B)

        if screen_coords is not None:
            # Append name and coordinates to output string
            output_str += f"{obj.name},{screen_coords[0]},{screen_coords[1]},{screen_coords[2]},{screen_coords[3]},{screen_coords[4]},{_C}\n"

    # Save output string to file
    with open(f"{bpy.path.abspath('//')}/{os.path.basename(os.path.dirname(bpy.data.filepath))}_1_output.txt","a") as f:
        f.write(output_str)