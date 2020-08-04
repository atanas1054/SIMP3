from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import json
import os, glob

p1 = (18.361338, 62.657757)
p2 = (49.6069, 74.7921)
p3 = (21.173973, 55.728195)
p4 = (52.361614, 67.71394)
p5 = (24.51247, 48.710445)
p6 = (55.36713, 60.451206)
p7 = (27.050087, 43.62588)
p8 = (57.22464, 55.228348)
left_sidewalk_polygon = Polygon([p1, p2, p4, p3])
street_polygon = Polygon([p3, p4, p6, p5])
right_sidewalk_polygon = Polygon([p5, p6, p8, p7])

# left_sidewalk_polygon = Polygon([(18.361338, 62.657757), (49.6069, 74.7921), (52.361614, 67.71394), (21.173973, 55.728195)])
# street_polygon = Polygon([(21.173973, 55.728195), (52.361614, 67.71394), (55.36713, 60.451206), (24.51247, 48.710445)])
# right_sidewalk_polygon = Polygon([(24.51247, 48.710445), (55.36713, 60.451206), (57.22464, 55.228348), (27.050087, 43.62588)])

def walkable_area(x, y, left_sidewalk=False, right_sidewalk=False, street=False):
    """Checks if a given point is in a given walkable area.
    
    Arguments:
        x {float} -- x coordinate of point
        y {flot} -- y-coordinate of point
    
    Keyword Arguments:
        left_sidewalk {bool} -- should be {True} if function should check if the point is in this area (default: {False})
        right_sidewalk {bool} -- should be {True} if function should check if the point is in this area (default: {False})
        street {bool} -- should be {True} if function should check if the point is in this area (default: {False})
    
    Returns:
        bool -- {True} if point is in defined walkable are, {False} otherwise
    """
 
    assert left_sidewalk or right_sidewalk or street, "At least one of the arguments 'left_sidewalk', 'right_sidewalk' or 'street' should be True."

    point = Point(x, y)
    
    ret = False
    if left_sidewalk:
        ret = ret or left_sidewalk_polygon.contains(point)
    if street:
        ret = ret or street_polygon.contains(point)
    if right_sidewalk:
        ret = ret or right_sidewalk_polygon.contains(point)
        
    return ret
    


def orientation_quaternion(pos_ped, target):
    vec1 = np.array([1, 0])
    vec2 = target - pos_ped
    v1_u = vec1 / np.linalg.norm(vec1)
    v2_u = vec2 / np.linalg.norm(vec2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    # set direction of angle wrt the jmonkey/opends coordinate system
    cross = np.cross(v1_u, v2_u)
    angle = angle * (- np.sign(np.dot(vec1,cross)))
    
    # euler_to_quaternion
    roll = angle[0]
    yaw = 0
    pitch = 0
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return {'x' : qx, 'y' : qy, 'z' : qz, 'w' : qw}

def orientation_angle(pos_ped, target):
    quaternion = orientation_quaternion(pos_ped, target)
    qx = quaternion['x']
    qy = quaternion['y']
    qz = quaternion['z']
    qw = quaternion['w']
    return '{"x":' + str(qx) + ',"y":' + str(qy) + ',"z":' + str(qz) + ',"w":' + str(qw) + '}'


def max_next_velocity(curr_velocity):
    # Kramer provides the following formula, which defines how long (t) it takes to reach a desired velocity (v) starting with v=0
    # t =  -0.0898 * v^2 + 0.8344 * v
    # We determine that in the next time step, the pedestrian may adopt the speed that he would reach within 0.5 second at most
    # thus, we solve the formula for v:
    # v_1 = 2/449 (sqrt(1087849-561250*t) + 1043)
    # v_2 = 2/449 (1043 - sqrt(1087849-561250*t)) <==
    t_to_curr = -0.0898 * curr_velocity**2 + 0.8344 * curr_velocity
    radicand = 1087849-561250*(t_to_curr+0.5)
    if radicand < 0:
        return 4
    v = 2/449 * (1043 - np.sqrt(radicand))
    return v


def resize_vector(vector, length=1):
    """Return a resized vector of length 'length'. If no length is specified, then the unit vector is returned.
    
    Arguments:
        vector {list-like} -- the vector which length should be resized
    
    Keyword Arguments:
        length {int} -- length to which the vector should be resized (default: {1})
    
    Returns:
        list-like -- the resized vector
    """
    return vector / np.linalg.norm(vector) * length


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'.
    
    Arguments:
        v1 {vector-like} -- first vector
        v2 {vector-like} -- second vector
    
    Returns:
        float -- angle in radians betwen given vectors
    """
    v1_u = resize_vector(v1)
    v2_u = resize_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def parse_action_file(path):
    with open(path, 'r') as myfile:
        action_file = myfile.read().replace('\n', '')
        action_file = action_file.replace('\t', '')
        action_file = action_file.replace(' ', '')
        action_file = action_file.replace('\":', '\": ')
        action_file = action_file.replace(',', ', ')

        str1 = 'ActionSequence:'
        str2 = 'SetPose:'
        i = action_file.find(str1)
        j = action_file.find(str2)
        action_seq_str = action_file[i + len(str1):]
        set_pose_str = action_file[j + len(str2):i]

        action_seq_data = json.loads(action_seq_str)
        set_pose_data = json.loads(set_pose_str)

        return set_pose_data, action_seq_data


def test_walkable_areas():
    root_dir = '/Users/nora/Documents/Uni/mastersthesis/emidas-code/dataset/scenes'

    for root, _, files in os.walk(root_dir):
        if root[len(root_dir):].count(os.sep) == 3:
            scene, refl = root[len(root_dir) + 1:].split(sep=os.sep)[1:3]

            # determine which pedestrian is crossing
            if 'p1_crossing.txt' in files:
                p1_crossing = True
                p2_crossing = False
            elif 'p2_crossing.txt' in files:
                p1_crossing = False
                p2_crossing = True
            else:
                p1_crossing = False
                p2_crossing = False

            # Extract all waypoints
            waypoints = {'p1' : [], 'p2' : []}
            for action_file in glob.glob(root + '/p[1-2]*.txt'):
                p = 'p1' if 'p1' in action_file[len(root):] else 'p2'
                set_pose, action_seq = parse_action_file(action_file)
                waypoints[p].append((set_pose['position']['x']/100, set_pose['position']['z']/100))
                for action in action_seq:
                    x = action['controlPoints'][1]['x']/100
                    z = action['controlPoints'][1]['z']/100
                    waypoints[p].append((x, z))

            # Check if they are on the sidewalks
            for num, (x, z) in enumerate(waypoints['p1']):
                if not walkable_area(x, z, left_sidewalk=True, right_sidewalk=True) and not (p1_crossing and num == len(waypoints['p1']) - 1):
                    print(scene, refl, num + 1)
            for num, (x, z) in enumerate(waypoints['p2']):
                if not walkable_area(x, z, left_sidewalk=True, right_sidewalk=True) and not (p2_crossing and num == len(waypoints['p2']) - 1):
                    print(scene, refl, num + 1)