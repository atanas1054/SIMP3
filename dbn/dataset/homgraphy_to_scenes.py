import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import re
import matplotlib.pyplot as plt
import util


def rotate(origin, point, angle):
    """Rotate a point counterclockwise by a given angle around a given origin. The angle should be given in radians.
    
    Arguments:
        origin {tuple-like} -- point around which a given point should be rotated
        point {tuple-like} -- point that sould be rotated
        angle {float} -- angle in radians
    
    Returns:
        float, float -- coordinate of resulting point
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def find_new_origin(old_origin, scene):

    if scene == '0101':
        return 48.4, 68.75

    if scene == '0102':
        return 30.948837, 60.975452
    
    if scene == '0105':
        return 48.4, 68.75

    if scene == '0106':
        return 35.598907, 63.747414

    if scene == '0107':
        return 48.186043, 68.75136

    if scene == '0108':
        return 48.186043, 68.75136

    if scene == '0201':
        return 26.589005, 59.83778

    if scene == '0203':
        return 37.083706, 63.562252

    if scene == '03a02':
        return 48.4, 68.75

    if scene == '03a03':
        return 48.4, 68.75

    if scene == '03a04':
        return 48.4, 68.75

    if scene == '03a06':
        return 48.4, 68.75

    if scene == '03a07':
        return 48.4, 68.75

    if scene == '03a08':
        return 48.4, 68.75
    
    if scene == '03b05':
        return 48.4, 68.75
    
    if scene == '03b06':
        return 48.4, 68.75

    if scene == '03b07':
        return 49.99914, 69.261986

    if scene == '03b12':
        return 49.99914, 69.261986

    if scene == '0402':
        return 37.083706, 63.562252

    if scene == '0403':
        return 37.083706, 63.562252

    if scene == '0404':
        return 37.083706, 63.562252

    if scene == '0503':
        return 45.42387, 66.73454

    if scene == '0505':
        return 42.838642, 65.70871

    if scene == '0506':
        return 42.10944, 65.60742

    if scene == '0508':
        return 40.36687, 64.9503

    if scene == '0603':
        return 45.42387, 66.73454

    if scene == '0801':
        return 48.4, 68.75

    if scene == '0802':
        return 31.265488, 61.009747

    #TODO:
    
    assert False, "Failure in find_new_origin, old_origin=" + str(old_origin)

def perpendicular_vector(p, q):
    """Returns a vector that is perpendicular to the line defined by two points.
    
    Arguments:
        p {list-like} -- one of the points defining the line
        q {list-like} -- the other point defining the line
    
    Returns:
        list-like -- a vector that is perpendicular to the line defined by p and q 
    """
    p_x, p_y = p
    q_x, q_y = q

    # calculate vector PQ in 3D, where the second dimension is 0 (as in the jmonkey and OpenDS coordinate system)
    pq = np.subtract([q_x, 0, q_y], [p_x, 0, p_y])

    # the cross product of two vectors is perpendicular to both vectors (unless both vectors are parallel -- but this is avoided here)
    perp = np.cross(pq, [0, 1, 0])
    return [perp[0], perp[2]]


def correct_right_side(point):
    # Vector modeling the center of the new street
    p = np.array([53.694565, 64.41612])
    q = np.array([6.853607, 46.365456])

    vector = perpendicular_vector(p3, p4)
    vector = util.resize_vector(vector, length=5)

    p = p + vector
    q = q + vector

    # point of intersection https://math.stackexchange.com/a/2325299
    pq = q - p
    p_to_point = point - p
    p_to_perp = (np.dot(pq, p_to_point) / np.dot(pq, pq)) * pq
    perp = p + p_to_perp
    
    return point - perp


def move_to_new_street(origin, waypoints_left, waypoints_right, angle, scene):
    origin = np.array(origin)
    waypoints_left = np.array(waypoints_left)
    waypoints_right = np.array(waypoints_right)
    new_waypoints_left = []
    new_waypoints_right = []

    # Rotate points
    for i in range(len(waypoints_left)):
        x, y = rotate(origin, waypoints_left[i], -angle)
        new_waypoints_left.append([x, y])
    new_waypoints_left = np.array(new_waypoints_left)

    for i in range(len(waypoints_right)):
        x, y = rotate(origin, waypoints_right[i], -angle)
        new_waypoints_right.append([x, y])
    new_waypoints_right = np.array(new_waypoints_right)

    # Shift points
    new_origin = find_new_origin(origin, scene)
    shift = origin - new_origin
    new_waypoints_left = new_waypoints_left - shift
    new_waypoints_right = new_waypoints_right - shift

    # correct road width for right sidewalk
    shift = correct_right_side(new_waypoints_right[0])
    new_waypoints_right = new_waypoints_right - shift

    # for coordinate in new_waypoints_left:
    #     print(walkable_area(coordinate[0], coordinate[1], left_sidewalk=True))
    # for coordinate in new_waypoints_right:
    #     print(walkable_area(coordinate[0], coordinate[1], right_sidewalk=True))

    # new_waypoints_left = new_waypoints_left * 100
    # new_waypoints_right = new_waypoints_right * 100

    # for coordinate in new_waypoints_left:
    #     print('{"x":%f,"y":0.0,"z":%f}' % (coordinate[0], coordinate[1]))
    # print()
    # for coordinate in new_waypoints_right:
    #     print('{"x":%f,"y":0.0,"z":%f}' % (coordinate[0], coordinate[1]))

    new_waypoints_p1, new_waypoints_p2 = side_to_ped(new_waypoints_left, new_waypoints_right, scene)

    x = [p[0] for p in list(new_waypoints_p1) + list(new_waypoints_p2)] + [1.5437756, 4.756474, 49.6069, 21.59172, 27.050087, 57.22464, 21.59172, 24.343214, 55.07038, 52.618435]
    y = [-p[1] for p in list(new_waypoints_p1) + list(new_waypoints_p2)] + [-56.754288, -50.054478, -74.7921, -55.520863, -43.62588, -55.228348, -55.520863, -48.8782, -60.77621, -67.47873]
    c = ['r'] * len(new_waypoints_p1) + ['b'] * len(new_waypoints_p2) + ['g'] * 10

    plt.scatter(x, y, color=c)
    plt.show()

    return new_waypoints_p1, new_waypoints_p2

def ped_to_side(p1_waypoints, p2_waypoints, scene):
    # links überquert
    if scene == '0101' or scene == '0102' or scene == '0105' or scene == '0107' or scene == '0108' or scene == '0201' or scene == '03a02' or scene == '03a03' or scene == '03a08' or scene == '03b05' or scene == '03b06' or scene == '03b07' or scene == '03b12' or scene == '0402' or scene == '0404' or scene == '0506':
        last = p1_waypoints.pop()
        p2_waypoints.insert(0, last)
        return p1_waypoints, p2_waypoints
    
    # rechts überquert
    if scene == '0106' or scene == '0203' or scene == '03a04' or scene == '03a06' or scene == '03a07' or scene == '0403' or scene == '0503' or scene == '0505' or scene == '0508':
        last = p2_waypoints.pop()
        p1_waypoints.insert(0, last)
        return p1_waypoints, p2_waypoints

    # niemand überquert
    if scene == '0603' or scene == '0801' or scene == '0802':
        return p1_waypoints, p2_waypoints

    #TODO:
    
    assert False, 'Scene not implemented in "ped_to_side".'


def side_to_ped(left_waypoints, right_waypoints, scene):
    if scene == '0603' or scene == '0801' or scene == '0802':
        return left_waypoints, right_waypoints

    left_waypoints = list(left_waypoints)
    right_waypoints = list(right_waypoints)

    if scene == '0101' or scene == '0102' or scene == '0105' or scene == '0107' or scene == '0108' or scene == '0201' or scene == '03a02' or scene == '03a03' or scene == '03a08' or scene == '03b05' or scene == '03b06' or scene == '03b07' or scene == '03b12' or scene == '0402' or scene == '0404' or scene == '0506':
        first = right_waypoints.pop(0)
        left_waypoints.append(first)
        return np.array(left_waypoints), np.array(right_waypoints)
    
    if scene == '0106' or scene == '0203' or scene == '03a04' or scene == '03a06' or scene == '03a07' or scene == '0403' or scene == '0503' or scene == '0505' or scene == '0508':
        first = left_waypoints.pop(0)
        right_waypoints.append(first)
        return np.array(left_waypoints), np.array(right_waypoints)

    #TODO:

    assert False, 'Scene not implemented in "side_to_ped".'


def orientation_at_start(p, direction):
    assert direction == 'building' or direction == 'car'

    if direction == 'building':
        print('Orientation of ' + p + ': {"x":0.9831359000778592,"y":0.0,"z":0.0,"w":0.18287646644141375}')
    if direction == 'car':
        print('Orientation of ' + p + ': {"x":-0.18287646644141378,"y":0.0,"z":0.0,"w":0.9831359000778592}')


def print_info(p_crossing, p_not_crossing, waypoints_crossing, waypoints_not_crossing, p_crossing_direction, p_not_crossing_direction, nobody_crosses=False):
    print('++++++++++++++++++++++++++++++')
    if not nobody_crosses:
        print('+++++++ ' + p_crossing + ' is crossing +++++++')
    else:
        print('+++++++++++++ ' + p_crossing + ' +++++++++++++')
    print('++++++++++++++++++++++++++++++')
    print()
    orientation_at_start(p_crossing, p_crossing_direction)
    print(waypoints_crossing)
    print()
    print('++++++++++++++++++++++++++++++')
    if not nobody_crosses:
        print('+++++ ' + p_not_crossing + ' is not crossing +++++')
    else:
        print('+++++++++++++ ' + p_not_crossing + ' +++++++++++++')
    print('++++++++++++++++++++++++++++++')
    print()
    orientation_at_start(p_not_crossing,p_not_crossing_direction)
    print(waypoints_not_crossing)
    if not nobody_crosses:
        print('Orientation at end: ' + util.orientation_angle(waypoints_not_crossing[len(waypoints_not_crossing)-1], waypoints_crossing[len(waypoints_crossing)-1]))





# Vector modeling the "old" street
p1 = [5, -1] #TODO:
p2 = [15, -1] #TODO:
vector_origin = np.subtract(p1, p2)

# Vector modeling the center of the new street
p3 = [53.694565, 64.41612]
p4 = [6.853607, 46.365456]
vector_target = np.subtract(p3, p4)

# Angle
angle = util.angle_between(vector_origin, vector_target)

scene = '0404' #TODO:
old_points_p1 = [[18.004396,-2.595322],[15.512027,-2.695125],[14.820369,-2.727293],[14.295451,-2.578527],[11.697393,1.030156]] #TODO:
old_points_p2 = [[11.373847,2.17854]] #TODO:
origin_p1 = old_points_p1[0]

old_points_left, old_points_right = ped_to_side(old_points_p1, old_points_p2, scene)


new_points_p1, new_points_p2 = move_to_new_street(origin_p1, old_points_left, old_points_right, angle, scene)


# p_crossing, p_not_crossing, waypoints_crossing, waypoints_not_crossing, p_crossing_direction, p_not_crossing_direction
if scene == '0101' or scene == '0105' or scene == '0107' or scene == '0108' or scene == '0506':
    print_info('p1', 'p2', new_points_p1, new_points_p2, 'building', 'building')

elif scene == '0503' or scene == '0505' or scene == '0508':
    print_info('p2', 'p1', new_points_p2, new_points_p1, 'building', 'building')

elif scene == '0106' or scene == '0203' or scene == '0403':
    print_info('p2', 'p1', new_points_p2, new_points_p1, 'car', 'car')

elif scene == '0102' or scene == '0201' or scene == '0402' or scene == '0404':
    print_info('p1', 'p2', new_points_p1, new_points_p2, 'car', 'car')

elif scene == '03a02' or scene == '03a03' or scene == '03a08' or scene == '03b05' or scene == '03b06' or scene == '03b07' or scene == '03b12':
    print_info('p1', 'p2', new_points_p1, new_points_p2, 'building', 'car')

elif scene == '03a04' or scene == '03a06' or scene == '03a07':
    print_info('p2', 'p1', new_points_p2, new_points_p1, 'car', 'building')

elif scene == '0603' or scene == '0801':
    print_info('p1', 'p2', new_points_p1, new_points_p2, 'building', 'car', nobody_crosses=True)

elif scene == '0802':
    print_info('p1', 'p2', new_points_p1, new_points_p2, 'car', 'car', nobody_crosses=True)


# TODO:


print(util.orientation_angle(np.array([48.8871, 56.136246]), np.array([46.62807686, 56.33456195])))