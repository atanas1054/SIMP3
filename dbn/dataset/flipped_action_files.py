import numpy as np
import json
import util
import os
import glob
import re

# Vertical reflection line
p1 = np.array([6.853607, 46.365456])
p2 = np.array([53.694565, 64.41612])
vert_refl_line = p2 - p1
vert_origin = p1

# Horizontal reflection line
p3 = np.array([18.361338, 62.657757])
p4 = np.array([49.6069, 74.7921])
p5 = np.array([27.050087, 43.62588])
p6 = np.array([57.22464, 55.228348])
p3p4 = p4 - p3
p5p6 = p6 - p5
p1 = p3 + 0.5 * p3p4
p2 = p5 + 0.5 * p5p6
hor_refl_line = p2 - p1
hor_origin = p1

def mirror_point(x, y, vertical=False, horizontal=False):
    assert vertical or horizontal, "Either 'vertical' or ' horizontal' should be True."

    p = np.array([x, y])
    p = p / 100

    for axis, refl_line, origin in [(vertical, vert_refl_line, vert_origin), (horizontal, hor_refl_line, hor_origin)]:
        if axis:
            # find point of intersection https://math.stackexchange.com/a/2325299
            origin_to_p = p - origin
            origin_to_perp = (np.dot(refl_line, origin_to_p) / np.dot(refl_line, refl_line)) * refl_line
            perp = origin + origin_to_perp
            p = p + 2 * (perp - p)

    return np.around(p * 100, 4)


def mirror_file(set_pose, action_seq, vertical=False, horizontal=False):
    waypoints = [(set_pose['position']['x'], set_pose['position']['z'])]
    look_targets = []

    for action in action_seq:
        x = action['controlPoints'][1]['x']
        z = action['controlPoints'][1]['z']
        waypoints.append((x, z))
        if 'lookAtTarget' in action:
            x = action['lookAtTarget']['x']
            z = action['lookAtTarget']['z']
            look_targets.append((x, z))
        else:
            look_targets.append(None)

    mirrored_waypoints = []
    for x, y in waypoints:
        x, y = mirror_point(x, y, vertical, horizontal)
        mirrored_waypoints.append((x, y))
    
    mirrored_look_targets = []
    for look in look_targets:
        if look is None:
            mirrored_look_targets.append(None)
        else:
            x, y = look
            x, y = mirror_point(x, y, vertical, horizontal)
            mirrored_look_targets.append((x,y))
    
    x, z = mirrored_waypoints[0]
    set_pose['position']['x'] = x
    set_pose['position']['z'] = z
    quaternion_start = util.orientation_quaternion(pos_ped=np.array(mirrored_waypoints[0]), target=np.array(mirrored_waypoints[1]))
    set_pose['orientation'] = quaternion_start
    

    assert len(action_seq) + 1 == len(mirrored_waypoints)
    assert len(action_seq) == len(mirrored_look_targets)

    for idx in range(len(action_seq)):
        x1, z1 = mirrored_waypoints[idx]
        x2, z2 = mirrored_waypoints[idx+1]
        action_seq[idx]['startPosition']['x'] = x1
        action_seq[idx]['startPosition']['z'] = z1
        action_seq[idx]['controlPoints'][0]['x'] = x1
        action_seq[idx]['controlPoints'][0]['z'] = z1
        action_seq[idx]['controlPoints'][1]['x'] = x2
        action_seq[idx]['controlPoints'][1]['z'] = z2

        if mirrored_look_targets[idx] is not None:
            x, z = mirrored_look_targets[idx]
            action_seq[idx]['lookAtTarget']['x'] = x
            action_seq[idx]['lookAtTarget']['z'] = z
            if 'spineTarget' in action_seq[idx]:
                action_seq[idx]['spineTarget']['x'] = x
                action_seq[idx]['spineTarget']['z'] = z
        else:
            assert not 'lookAtTarget' in action_seq[idx]
    


def final_orientation(action_seq_p1, action_seq_p2):
    p1 = 'orientation' in action_seq_p1[len(action_seq_p1)-1]
    p2 = 'orientation' in action_seq_p2[len(action_seq_p2)-1]

    assert not (p1 and p2)

    if not p1 and not p2:
        return action_seq_p1, action_seq_p2

    if p1:
        pos_x = action_seq_p1[len(action_seq_p1)-1]['controlPoints'][1]['x']
        pos_z = action_seq_p1[len(action_seq_p1)-1]['controlPoints'][1]['z']
        target_x = action_seq_p2[len(action_seq_p2)-1]['controlPoints'][1]['x']
        target_z = action_seq_p2[len(action_seq_p2)-1]['controlPoints'][1]['z']

    if p2:
        pos_x = action_seq_p2[len(action_seq_p2)-1]['controlPoints'][1]['x']
        pos_z = action_seq_p2[len(action_seq_p2)-1]['controlPoints'][1]['z']
        target_x = action_seq_p1[len(action_seq_p1)-1]['controlPoints'][1]['x']
        target_z = action_seq_p1[len(action_seq_p1)-1]['controlPoints'][1]['z']
    
    quaternion = util.orientation_quaternion(np.array([pos_x, pos_z]), np.array([target_x, target_z]))

    if p1:
        action_seq_p1[len(action_seq_p1)-1]['orientation'] = quaternion

    if p2:
        action_seq_p2[len(action_seq_p2)-1]['orientation'] = quaternion


def create_txt(set_pose, action_seq):
    action_file = 'SetPose:\n' +  json.dumps(set_pose, indent=4) + '\n\nActionSequence:\n' + json.dumps(action_seq, indent=4)

    action_file = re.sub(r'\{\n *\"x\": (\d+.\d*),\n *\"y\": (\d+.\d*),\n *\"z\": (\d+.\d*)\n *\}', '{"x":' + r'\1' + ',"y":' + r'\2' + ',"z":' r'\3' + '}', action_file)
    action_file = action_file.replace('"controlPoints": [', '"controlPoints":\n        [')
    action_file = re.sub(r'\{\n *\"x\": (-?\d+.\d*),\n *\"y\": (-?\d+.\d*),\n *\"z\": (-?\d+.\d*),\n *\"w\": (-?\d+.\d*)\n *\}', '{"x":' + r'\1' + ',"y":' + r'\2' + ',"z":' r'\3' + ',"w":' r'\4' + '}', action_file)

    # action_file = re.sub(r'\: \n *\"x\"', '"x"', action_file)
    # action_file = re.sub(r'\n *\{\"x\"', '{"x"', action_file)
    # action_file = re.sub(r'\n *\"y\"', '"y"', action_file)
    # action_file = re.sub(r'\n *\"z\"', '"z"', action_file)
    # action_file = re.sub(r'\"z\": (\d+.\d*)\n *', '"z": ' + r'\1', action_file)
    return action_file


def mirror_scene(folder_path, template_dir, file_names):
    if file_names[0].startswith('p1'):
        file_p1 = file_names[0]
        file_p2 = file_names[1]
    else:
        file_p1 = file_names[1]
        file_p2 = file_names[0]
    p1_suffix = file_p1[2:-4]
    p2_suffix = file_p2[2:-4]

    set_pose_p1, action_seq_p1 = util.parse_action_file(template_dir + '/' + file_p1)
    set_pose_p2, action_seq_p2 = util.parse_action_file(template_dir + '/' + file_p2)

    # Vertical flip => pedestrians switch role
    mirror_file(set_pose_p1, action_seq_p1, vertical=True)
    mirror_file(set_pose_p2, action_seq_p2, vertical=True)
    final_orientation(action_seq_p1, action_seq_p2)
    
    if not os.path.exists(folder_path + '/vertical-reflection'):
        os.mkdir(folder_path + '/vertical-reflection')
    else:
        old_files = glob.glob(folder_path + '/vertical-reflection/*.txt')
        for f in old_files:
            os.remove(f)
    text_file = open(folder_path + '/vertical-reflection/p1' + p2_suffix + '.txt', 'w')
    text_file.write(create_txt(set_pose_p2, action_seq_p2))
    text_file.close()
    
    text_file = open(folder_path + '/vertical-reflection/p2' + p1_suffix + '.txt', 'w')
    text_file.write(create_txt(set_pose_p1, action_seq_p1))
    text_file.close()

    # Recreate original one
    # set_pose_p1, action_seq_p1 = util.parse_action_file(template_dir + '/' + file_p1) <= reflect the reflection so don't load the original data again
    # set_pose_p2, action_seq_p2 = util.parse_action_file(template_dir + '/' + file_p2)
    mirror_file(set_pose_p1, action_seq_p1, vertical=True)
    mirror_file(set_pose_p2, action_seq_p2, vertical=True)
    final_orientation(action_seq_p1, action_seq_p2)
    
    if not os.path.exists(folder_path + '/no-reflection'):
        os.mkdir(folder_path + '/no-reflection')
    else:
        old_files = glob.glob(folder_path + '/no-reflection/*.txt')
        for f in old_files:
            os.remove(f)
    text_file = open(folder_path + '/no-reflection/' + file_p1, 'w')
    text_file.write(create_txt(set_pose_p1, action_seq_p1))
    text_file.close()
    
    text_file = open(folder_path + '/no-reflection/' + file_p2, 'w')
    text_file.write(create_txt(set_pose_p2, action_seq_p2))
    text_file.close()

    # Horizontal flip => changed walking direction
    set_pose_p1, action_seq_p1 = util.parse_action_file(template_dir + '/' + file_p1)
    set_pose_p2, action_seq_p2 = util.parse_action_file(template_dir + '/' + file_p2)
    mirror_file(set_pose_p1, action_seq_p1, horizontal=True)
    mirror_file(set_pose_p2, action_seq_p2, horizontal=True)
    final_orientation(action_seq_p1, action_seq_p2)
    
    if not os.path.exists(folder_path + '/horizontal-reflection'):
        os.mkdir(folder_path + '/horizontal-reflection')
    else:
        old_files = glob.glob(folder_path + '/horizontal-reflection/*.txt')
        for f in old_files:
            os.remove(f)
    text_file = open(folder_path + '/horizontal-reflection/p1' + p1_suffix + '.txt', 'w')
    text_file.write(create_txt(set_pose_p1, action_seq_p1))
    text_file.close()
    
    text_file = open(folder_path + '/horizontal-reflection/p2' + p2_suffix + '.txt', 'w')
    text_file.write(create_txt(set_pose_p2, action_seq_p2))
    text_file.close()

    # Vertical + horizontal flip => pedestrians switch role + changed walking direction
    set_pose_p1, action_seq_p1 = util.parse_action_file(template_dir + '/' + file_p1)
    set_pose_p2, action_seq_p2 = util.parse_action_file(template_dir + '/' + file_p2)
    mirror_file(set_pose_p1, action_seq_p1, vertical=True, horizontal=True)
    mirror_file(set_pose_p2, action_seq_p2, vertical=True, horizontal=True)
    final_orientation(action_seq_p1, action_seq_p2)
    
    if not os.path.exists(folder_path + '/vertical-horizontal-reflection'):
        os.mkdir(folder_path + '/vertical-horizontal-reflection')
    else:
        old_files = glob.glob(folder_path + '/vertical-horizontal-reflection/*.txt')
        for f in old_files:
            os.remove(f)
    text_file = open(folder_path + '/vertical-horizontal-reflection/p1' + p2_suffix + '.txt', 'w')
    text_file.write(create_txt(set_pose_p2, action_seq_p2))
    text_file.close()
    
    text_file = open(folder_path + '/vertical-horizontal-reflection/p2' + p1_suffix + '.txt', 'w')
    text_file.write(create_txt(set_pose_p1, action_seq_p1))
    text_file.close()



root_dir = '/Users/nora/Documents/Uni/mastersthesis/emidas-code/dataset/scenes'
templates_root_dir = '/Users/nora/Documents/Uni/mastersthesis/emidas-code/dataset/scenes-templates'

for root, dirs, files in os.walk(templates_root_dir):
    if root[len(root_dir):].count(os.sep) == 2:
        # if root[-4:] == '0102':
        target_dir = root_dir + root[len(templates_root_dir):]
        files = [f for f in files if f.endswith('.txt')]
        assert len(files) == 2
        mirror_scene(target_dir, root, files)




    

