from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import nearest_points
import numpy as np
import json
import os, glob


# definition of polygons for function walkable_area
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


def nearest_walkable_area(x, y):
    """Returns the closest walkable area to the given coordinates.

    Args:
        x (float): x coordinate
        y (float): y coordinate

    Returns:
        str, float: closest walkable area and distance of point to it
    """
    point = Point(x, y)
    distance = float('inf')
    closest_area = ''

    for poly, area_name in [(left_sidewalk_polygon, 'left'), (street_polygon, 'street'), (right_sidewalk_polygon, 'right')]:
        p1, _ = nearest_points(poly, point)
        d = point.distance(p1)
        if d < distance:
            distance = d
            closest_area = area_name
    
    assert closest_area != ''
    return closest_area, distance


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
    """Parse the action file (p1*.txt/p2*.txt) used to create a scene in OpenDS.

    Args:
        path (str): absolute path of action file

    Returns:
        dict, dict: two dictionaries containing the two building blocks of the file
    """
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