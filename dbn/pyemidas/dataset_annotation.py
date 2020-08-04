import pandas as pd
import numpy as np
from itertools import chain
from datetime import datetime
import os, glob, logging, configparser
from pathlib import Path
import warnings

import util


# create logger
myformat = '%(levelname)s:%(message)s'
logfile_path = '../dataset/logging-total.log'
logging.basicConfig(filename=logfile_path, level=logging.DEBUG, format=myformat, filemode='a')
logfile_error_path = '../dataset/logging-error.log'
log_higher = logging.StreamHandler(stream=open(logfile_error_path, 'a'))
log_higher.setLevel(logging.WARNING)
formatter = logging.Formatter(myformat)
log_higher.setFormatter(formatter)
logging.getLogger('').addHandler(log_higher)

# ground truth data
gt_dir = '../study/0-total'
df_imas = pd.read_csv(gt_dir + '/ground_truth_imas_v3.csv', index_col='scene')
df_wti = pd.read_csv(gt_dir + '/ground_truth_wti_v3.csv', index_col='scene')
df_imas_08 = pd.read_csv(gt_dir + '/sc_08_ground_truth_imas.csv', dtype={'scene' : 'object'}).set_index('scene')
df_wti_08 = pd.read_csv(gt_dir + '/sc_08_ground_truth_wti.csv', dtype={'scene' : 'object'}).set_index('scene')
gt_dfs = {'imas' : df_imas, 'wti' : df_wti, 'imas_08' : df_imas_08, 'wti_08' : df_wti_08}
df_mapping = pd.read_csv('../dataset/ground_truth_mapping.csv', index_col='scene')

# mapping gesture names (name of gestures in OpenDS -> shorter names)
renamed_gestures = {'waveSlowSmallShoulderLevelShort' : 'wave_slow_shoulder',
                   'waveFastSmallShoulderLevelShort' : 'wave_fast_shoulder',
                   'waveSlowSmallHighHandShort' : 'wave_small_head',
                   'waveFastWideHighHandShort' : 'wave_wide_head',
                   'waveHandRaiseHigh' : 'arm_raise_high',
                   'waveHandRaise' : 'arm_raise_low',
                   'waveComeFast' : 'come_here',
                   'waveWait' : 'i_am_coming'}


def approaching_value(row, ped):
    """Determins the approaching feature. A pedestrian is approching the other one if s/he walks, looks at other, has a orientation angle of less than 90 degrees towards the other and the distance between both decreases.

    Arguments:
        row {Series} -- row of data frame
        ped {str} -- either '1' or '2', identifies the pedestrian

    Returns:
        str -- either 'y' (yes), 'm' (maybe), 'n' (no)
    """
    if row['motion_' + ped] == 'walking' and row['viewing_angle_' + ped] == 0 and row['body_facing_angle_' + ped] < 90:
        if 0.1 < row['reduced_distance']:
            return 'y'
        if -0.1 < row['reduced_distance'] <= 0.1:
            return 'm'
    return 'n'


def walking_area(row):
    """Determins the area in which the pedestrian is.

    Arguments:
        row {Series} -- row of the data frame

    Returns:
        str -- 'right'/'left'/'street'
    """
    x = row['X-location']
    y = row['Z-location']
    if util.walkable_area(x, y, right_sidewalk=True):
        return 'right'
    if util.walkable_area(x, y, left_sidewalk=True):
        return 'left'
    if util.walkable_area(x, y, street=True):
        return 'street'

    area, distance = util.nearest_walkable_area(x, y)
    logging.error(scene_id + ':The pedestrian is at position (' + str(x) + ', ' + str(y) + '), which away from the next walkable area by ' + str(distance) + ' m.)')
    return area


def viewing_area(row, ped):
    """Determins the discritization of the viewing area feature.

    Arguments:
        row {Series} -- row of the data frame
        ped {str} -- either '1' or '2', identifies the pedestrian

    Returns:
        str -- 'central'/'mid-peripheral'/'outside'
    """
    if row['viewing_angle_' + ped] == 0:
        return 'central'
    if row['viewing_angle_' + ped] < 60:
        return 'mid_peripheral'
    return 'outside'


def body_facing(row, ped):
    """Determins the discritization of the body facing feature.

    Arguments:
        row {Series} -- row of the data frame
        ped {str} -- either '1' or '2', identifies the pedestrian

    Returns:
        str -- 'strongly_facing'/'slightly_facing'/'turned_away'
    """
    if row['body_facing_angle_' + ped] < 10:
        return 'strongly_facing'
    if row['body_facing_angle_' + ped] < 90:
        return 'slightly_facing'
    return 'turned_away'


def match_rows(df_p1, df_p2):
    """May choose to remove time steps of pedestrians if there is no matching time step in the annotations of the other pedestrian.

    Arguments:
        df_p1 {DataFrame} -- annotations of pedestrian 1
        df_p2 {DataFrame} -- annotations of pedestrian 2

    Returns:
        list, list -- list containing the line indices that should be removed
    """
    del_p1 = []
    del_p2 = []
    p1 = df_p1[['time_1']].copy()
    p2 = df_p2[['time_2']].copy()
    next_i = 0
    while True:
        for i in range(next_i, min(len(p1.index), len(p2.index))):
            p1_val = p1.loc[i, 'time_1']
            p2_val = p2.loc[i, 'time_2']
            diff = abs(p1_val - p2_val)
            if diff > 90:
                # smaller time stamp is removed since it can not be matched
                if p1_val < p2_val:
                    del_p1.append(i + len(del_p1))
                    p1 = p1.drop([i]).reset_index(drop=True)
                else:
                    del_p2.append(i + len(del_p2))
                    p2 = p2.drop([i]).reset_index(drop=True)
                next_i = i
                break
        else:
            break
    if len(del_p1) > 0:
        logging.info(scene_id + ':Index ' + str(del_p1) + ' will be removed for p1.')
    if len(del_p2) > 0:
        logging.info(scene_id + ':Index ' + str(del_p2) + ' will be removed for p2.')
    return del_p1, del_p2


def remove_first_broken_lines(df):
    """Removes beginning time stamps if pedestrian is at position (0, 0).

    Arguments:
        df {DataFrame} -- annotations of pedestrian

    Returns:
        DataFrame -- the corrected data frame
    """
    return df.drop(df[(round(df['X-location'], 0) == 0) & (round(df['Z-location'], 0) == 0)].index).reset_index(drop=True)


def consecutive_index_list(index):
    """Given a list of indices, the function returns two elements that define the consecutive part of the indices list.

    Args:
        index (Pandas Index): the indices

    Returns:
        int, int: first and final indices of continuous list of indices that is a subset of `index`
    """
    index_list = index.tolist()
    if not index_list:
        return None, None
    last_idx = index_list[-1]
    list_len = len(index_list)
    for pos, idx in enumerate(index_list):
        if last_idx - idx + 1 == list_len - pos:
            first_idx = idx
            break
    return first_idx, last_idx


def scene_annotation(df_p1, df_p2, folder_path):
    """Creates a single CSV file containing the ground truth and feature values of the scene.

    Arguments:
        df_p1 {Dataframe} -- features of left pedestrian
        df_p2 {DataFrame} -- features of right pedestrian
        folder_path {str} -- current working directory

    Returns:
        DataFrame -- combined featurs ready for the DBN
    """
    # remove file that can exist if it's not the first time the annotation files are created
    file_to_remove = folder_path / 'defective_dbn_annotations.csv'
    file_to_remove.unlink(missing_ok=True)

    for df_ped, i in [(df_p1, '1'), (df_p2, '2')]:
        # derive motion (walking/standing)
        df_ped['next_X'] = df_ped['X-location'][1:].reset_index(drop=True)
        df_ped['next_Z'] = df_ped['Z-location'][1:].reset_index(drop=True)
        df_ped['motion_' + i] = df_ped.apply(lambda row: 'standing' if row['X-location'] == row['next_X'] and row['Z-location'] == row['next_Z'] else 'walking', axis=1)
        # correct last motion entry
        df_ped.loc[len(df_ped['motion_' + i]) - 1, 'motion_' + i] = df_ped['motion_' + i][len(df_ped['motion_' + i]) - 2]

        # in which walkable area
        df_ped['walkable_area_' + i] = df_ped.apply(walking_area, axis=1)

        # rename gestures 
        df_ped.replace({'upperBodyGesture' : renamed_gestures}, inplace=True)
        # stop gesture earlier if necessary
        start, end = consecutive_index_list(df_ped[(df_ped['motion_' + i] == 'standing') & (df_ped['upperBodyGesture'] != 'None')].index)
        if start is not None and end is not None:
            df_ped.iloc[start:end+1, list(df_ped.columns).index('upperBodyGesture')] = 'None'

    # rename columns to be able to concat the df's
    df_p1 = df_p1.rename({'Milliseconds Since Start' : 'time_1', 'upperBodyGesture' : 'gesture_1', 'lookAtTarget-X' : 'look_x_1', 'lookAtTarget-Z' : 'look_z_1', 'actionSequenceNumber' : 'action_seq_num_1', 'X-location' : 'X-location_1', 'Z-location' : 'Z-location_1', 'next_X' : 'next_X_1', 'next_Z' : 'next_Z_1'}, axis='columns')
    df_p2 = df_p2.rename({'Milliseconds Since Start' : 'time_2', 'upperBodyGesture' : 'gesture_2', 'lookAtTarget-X' : 'look_x_2', 'lookAtTarget-Z' : 'look_z_2', 'actionSequenceNumber' : 'action_seq_num_2', 'X-location' : 'X-location_2', 'Z-location' : 'Z-location_2', 'next_X' : 'next_X_2', 'next_Z' : 'next_Z_2'}, axis='columns')

    # cut after crossing ped reaches other side of the street (=> stands)
    for df_ped, p, crossing in [(df_p1, '1', p1_crossing), (df_p2, '2', p2_crossing)]:
        if crossing:
            start, end = consecutive_index_list(df_ped[df_ped['motion_' + p] == 'standing'].index)
            if end + 1 == len(df_ped.index):
                df_ped.drop([i for i in range(start, end+1)], inplace=True)
            else:
                # the pedestrian walks until the end, check if at least he reached the street
                if not list(df_ped[df_ped['walkable_area_' + p] == 'street'].index):
                    logging.error(scene_id + ':p' + p + ' never reaches the street. This shouldn\'t happen.')
                    return 


    # repair different time stamps
    del_p1, del_p2 = match_rows(df_p1, df_p2)
    df_p1 = df_p1.drop(del_p1).reset_index(drop=True)
    df_p2 = df_p2.drop(del_p2).reset_index(drop=True)
    
    # concat the df's
    df = pd.concat([df_p1, df_p2], axis=1 , sort=True, join='inner')

    # distance between pedestrians
    df['distance'] = np.linalg.norm(df[['X-location_1', 'Z-location_1']].values - df[['X-location_2', 'Z-location_2']].values, axis=1)
    global max_distance
    scene_max_distance = int(df.max(axis=0)['distance'])
    if max_distance < scene_max_distance:
        max_distance = scene_max_distance

    # viewing angle + body facing angle
    for p in ['1','2']:
        q = '2' if p == '1' else '1'
        for row in range(len(df.index)):
            x1 = df.loc[row, 'X-location_' + p]
            y1 = df.loc[row, 'Z-location_' + p]
            # get body-orientation vector
            for next_row in range(row + 1, len(df.index)):
                x2 = df.loc[next_row, 'X-location_' + p]
                y2 = df.loc[next_row, 'Z-location_' + p]
                if not (x1 == x2 and y1 == y2):
                    body_vec = np.array([x2 - x1, y2 - y1])
                    break
            else:
                for prev_row in range(row - 1, -1, -1):
                    x2 = df.loc[prev_row, 'X-location_' + p]
                    y2 = df.loc[prev_row, 'Z-location_' + p]
                    if not (x1 == x2 and y1 == y2):
                        body_vec = np.array([x1 - x2, y1 - y2])
                        break
                else:
                    logging.error(scene_id + ':p' + p + ' seems to stay in the same position all the time, annotation file won\'t be created. This shouldn\'t happen.')
                    return
            # calculate body facing angle
            ped_conn_vec = np.array([df.loc[row, 'X-location_' + q] - x1, df.loc[row, 'Z-location_' + q] - y1])
            if ped_conn_vec[0] == 0 and ped_conn_vec[1] == 0:
                logging.error(scene_id + ':Pedestrian connection vector is the zero vector at row ' + str(row) + '. This shouldn\'t happen.')
            angle = np.degrees(util.angle_between(ped_conn_vec, body_vec))
            df.loc[row, 'body_facing_angle_' + p] = angle
            # calculate viewing angle
            if not pd.isnull(df.loc[row, 'look_x_' + p]):
                df.loc[row, 'viewing_angle_' + p] = 0
            else:
                df.loc[row, 'viewing_angle_' + p] = angle

    # fix body facing angle due to wrong final orientation
    for p, not_crossing in [('1', not p1_crossing), ('2', not p2_crossing)]:
        if not_crossing:
            if scene[:2] == '01' or scene[:2] == '03' or scene[:2] == '04' or scene[:2] == '05':
                # all indices during the last action sequence where the pedestrian stands:
                max_action_seq = df.max(axis=0)['action_seq_num_' + p]
                start, end = consecutive_index_list(df[(df['action_seq_num_' + p] == max_action_seq) & (df['motion_' + p] == 'standing')].index)
                if start is not None and end is not None:
                    df.iloc[start:end+1, list(df.columns).index('body_facing_angle_' + p)] = 0

    # approaching column
    df['reduced_distance'] = df['distance'] - df['distance'][1:].reset_index(drop=True)
    df['approaching_1'] = df.apply(approaching_value, args=['1'], axis=1)
    df['approaching_2'] = df.apply(approaching_value, args=['2'], axis=1)

    # IMAS + WTI ground truth
    if 'vertical' in refl:
        pedestrians = [('1', '2'), ('2', '1')]
    else:
        pedestrians = [('1', '1'), ('2', '2')]
    if scene[:2] != '08':
        for p, q in pedestrians: # p: Dateiname, q: gt Spalte
            max_action_seq = int(df.max(axis=0)['action_seq_num_' + p])
            gt_imas_dict = {}
            gt_wti_dict = {}
            for action_seq in range(1, max_action_seq + 1):
                gt_code = df_mapping.loc[scene, 'p' + q + '_' + str(action_seq)]
                gt_imas_dict[action_seq] = gt_dfs['imas'].loc[scene, gt_code]
                gt_wti_dict[action_seq] = gt_dfs['wti'].loc[scene, gt_code]
            df['imas_' + p] = df.apply(lambda row: int(gt_imas_dict[row['action_seq_num_' + p]]), axis=1)
            df['wti_' + p] = df.apply(lambda row: int(gt_wti_dict[row['action_seq_num_' + p]]), axis=1)
    else:
        for p, q in pedestrians: # p: Dateiname, q: gt Spalte
            val1 = gt_dfs['imas_08'].loc[scene, 'p' + q + '_1']
            val2 = gt_dfs['imas_08'].loc[scene, 'p' + q + '_2']
            gt_imas_dict = {1 : val1, 2 : val1, 3 : val2, 4 : val2}
            val1 = gt_dfs['wti_08'].loc[scene, 'p' + q + '_1']
            val2 = gt_dfs['wti_08'].loc[scene, 'p' + q + '_2']
            gt_wti_dict = {1 : val1, 2 : val1, 3 : val2, 4 : val2}
            df['imas_' + p] = df.apply(lambda row: gt_imas_dict[row['action_seq_num_' + p]], axis=1)
            df['wti_' + p] = df.apply(lambda row: gt_wti_dict[row['action_seq_num_' + p]], axis=1)


    # locations csv
    df_pos = df[['time_1', 'X-location_1', 'Z-location_1', 'time_2', 'X-location_2', 'Z-location_2']]
    df_pos.to_csv(folder_path / 'pedestrian_positions.csv')

    # polish csv
    col_order = []
    for i in ['1','2']:
        df['viewing_area_' + i] = df.apply(viewing_area, args=[i], axis=1)
        df['body_facing_' + i] = df.apply(body_facing, args=[i], axis=1)
        # df = df.astype({'imas_' + i : 'uint8'})
        df = df.drop(columns=['time_' + i, 'X-location_' + i, 'Z-location_' + i, 'next_X_' + i, 'next_Z_' + i, 'look_x_' + i, 'look_z_' + i, 'viewing_angle_' + i, 'action_seq_num_' + i, 'body_facing_angle_' + i, 'motion_' + i, 'walkable_area_' + i])
        col_order += ['imas_' + i, 'wti_' + i, 'viewing_area_' + i, 'body_facing_' + i, 'gesture_' + i, 'approaching_' + i]
    df = df.drop(columns=['reduced_distance'])
    col_order += ['distance']
    df = df[col_order] # new column order

    df.to_csv(folder_path / 'dbn_annotations.csv')

    return df



def clear_before_annotations(dir_path):
    """Checks before creating the annotations file, if the scene is valid.

    Args:
        dir_path (str): path of scene

    Returns:
        pd.DataFrame, pd.DataFrame: a dataframe for each pedestrian, or `None, None` if the scene is not valid
    """
    skip_scene = False

    # load two csv's
    columns = ['Milliseconds Since Start', 'X-location', 'Z-location', 'actionSequenceNumber', 'upperBodyGesture', 'lookAtTarget-X', 'lookAtTarget-Z'] 
    df_p1 = pd.read_csv(Path(dir_path) / 'pedestrian 1 features.csv', usecols=columns)
    df_p2 = pd.read_csv(Path(dir_path) / 'pedestrian 2 features.csv', usecols=columns)

    # determine which pedestrian is crossing
    global p1_crossing, p2_crossing
    files_in_dir = os.listdir(dir_path) #dir_path.glob('**/*')
    if 'p1_crossing.txt' in files_in_dir:
        p1_crossing = True
        p2_crossing = False
    elif 'p2_crossing.txt' in files_in_dir:
        p1_crossing = False
        p2_crossing = True
    elif scene[0:2] == '06' or scene[0:2] == '08':
        p1_crossing = False
        p2_crossing = False
    else:
        logging.error(scene_id + ':Naming convention of action files is broken, annotation file won\'t be created.')
        df_p1, df_p2 = None, None 
        return df_p1, df_p2

    # check how many lines with position (0,0) exist
    old_len = [len(df_p1.index), len(df_p2.index)]
    df_p1 = remove_first_broken_lines(df_p1)
    df_p2 = remove_first_broken_lines(df_p2)
    len_diff = [old_len[0] - len(df_p1.index), old_len[1] - len(df_p2.index)]
    for i in range(len(len_diff)):
        if len_diff[i] == old_len[i]:
            logging.error(scene_id + ':p' + str(i+1) + ' remains at (0,0,0).')
            df_p1, df_p2 = None, None 
            return df_p1, df_p2
        elif len_diff[i] > 0:
            logging.info(scene_id + ':The pedestrian p' + str(i+1) + ' is at position (0,0,0) during ' + str(len_diff[i]) + ' steps.')

    # Extract all waypoints
    waypoints = {'p1' : [], 'p2' : []}
    for action_file in glob.glob(dir_path + '/p[1-2]*.txt'):
        p = 'p1' if 'p1' in action_file[len(dir_path):] else 'p2'
        set_pose, action_seq = util.parse_action_file(action_file)
        waypoints[p].append((set_pose['position']['x']/100, set_pose['position']['z']/100))
        for action in action_seq:
            x = action['controlPoints'][1]['x']/100
            z = action['controlPoints'][1]['z']/100
            waypoints[p].append((x, z))

    # Check if they are on the sidewalks
    error_msg = ''
    for num, (x, z) in enumerate(waypoints['p1']):
        if not util.walkable_area(x, z, left_sidewalk=True, right_sidewalk=True) and not (p1_crossing and num == len(waypoints['p1']) - 1):
            if not error_msg:
                skip_scene = True
                error_msg = scene_id + ':Waypoint of p1 outside walkable area: ' + str(num+1)
            else:
                error_msg += ', ' + str(num+1)
    if error_msg:
        logging.error(error_msg)
        error_msg = ''
    for num, (x, z) in enumerate(waypoints['p2']):
        if not util.walkable_area(x, z, left_sidewalk=True, right_sidewalk=True) and not (p2_crossing and num == len(waypoints['p2']) - 1):
            if scene_id.startswith('0603_vh') and num+1 == 5: # not too bad error
                break
            if not error_msg:
                skip_scene = True
                error_msg = scene_id + ':Waypoint of p2 outside walkable area: ' + str(num+1)
            else:
                error_msg += ', ' + str(num+1)
    if error_msg:
        logging.error(error_msg)

    # check if pedestrian stops walking at some point in time
    for df, p in [(df_p1, 'p1'), (df_p2, 'p2')]:
        prev_x = df.loc[0, 'X-location']
        prev_z = df.loc[0, 'Z-location']
        if abs(prev_x - waypoints[p][0][0]) > 2 or abs(prev_z - waypoints[p][0][1]) > 2:
            skip_scene = True
            logging.error(scene_id + ':' + p + ' doesn\'t start at right position.')
            break
        count = 0
        for row in range(1, len(df.index)):
            curr_x = df.loc[row, 'X-location']
            curr_z = df.loc[row, 'Z-location']
            if abs(curr_x - waypoints[p][-1][0]) < 1 and abs(curr_z - waypoints[p][-1][1]) < 1: # and not scene_id.startswith('04'):
                break
            if prev_x == curr_x and prev_z == curr_z:
                count += 1
            else:
                count = 0
            if count > 20:
                skip_scene = True
                logging.error(scene_id + ':' + p + ' doesn\'t walk.')
                break
            prev_x = curr_x
            prev_z = curr_z

    if skip_scene:
        df_p1, df_p2 = None, None 
    return df_p1, df_p2


def clear_after_annotations(df, curr_dir):
    """Checks whether the scene is valid after the annotations file creation to check the synchronocity of the interaction.

    Args:
        df (pd.DataFrame): combined variable values of both pedestrians
        curr_dir (str): scene directory
    """
    if scene[0:2] == '08':
        # no interaction at all
        return
    
    elif scene[0:2] == '06':
        # check if the interaction (more or less overlaps)
        p1_gaze = list(df[df['viewing_area_1'] == 'central'].index)
        p2_gaze = list(df[df['viewing_area_2'] == 'central'].index)
        p1_gesture = list(df[df['gesture_1'] != 'None'].index)
        p2_gesture = list(df[df['gesture_2'] != 'None'].index)

        first_gaze = 'p1' if p1_gaze[0] < p2_gaze[0] else 'p2'
        first_gesture = 'p1' if p1_gesture[0] < p2_gesture[0] else 'p2'

        if first_gaze == first_gesture:
            if (first_gaze == 'p1' and p2_gaze[0] - p1_gesture[-1] > 20) or (first_gaze == 'p2' and p1_gaze[0] - p2_gesture[-1] > 20):
                # 2 Sekunden zwischen Gestenende und Blickanfang ist zu lang
                logging.error(scene_id + ':Interaction is not synchronous enough.')
            else:
                return
        else:
            # when interaction is interleaved, then it overlaps for sure
            return
        
    else:
        if scene[0:2] == '01' or scene[0:2] == '02' or scene[0:3] == '03a' or scene[0:2] == '01' or scene == '0402' or scene == '0403' or scene == '0505' or scene == '0506' or scene == '0508':
            p = '1' if p1_crossing else '2' # p starts the interaction
            q = '2' if p1_crossing else '1' # q doesn't start the interaction

        elif scene[0:3] == '03b' or scene == '0404' or scene == '0503':
            p = '2' if p1_crossing else '1' # p starts the interaction
            q = '1' if p1_crossing else '2' # q doesn't start the interaction

        else:
            assert False, 'all scenes should be covered until yet'
        
        try:
            defective_scene = False
            # Hard requirement: Blick 1 vor Blick 2
            gaze1 = list(df[df['viewing_area_' + p] == 'central'].index)[0]
            gaze2 = list(df[df['viewing_area_' + q] == 'central'].index)[0]
            if gaze1 - gaze2 > 5: # formerly gaze1 > gaze2
                logging.error(scene_id + ':The pedestrian that should start the interaction looks second.')
                defective_scene = True

            # Hard requirement: Geste 2 nach Geste 1 und vor Ende
            gesture1_list = list(df[df['gesture_' + p] != 'None'].index)
            gesture1 = gesture1_list[0]
            gesture1_last = gesture1_list[-1]
            gesture2 = list(df[df['gesture_' + q] != 'None'].index)[0]
            if gesture1 - gesture2 > 5 and scene != '0402': # formerly gesture1 > gesture2
                logging.error(scene_id + ':The pedestrian that should start the interaction waves second.')
                defective_scene = True
        
            # Soft requirement: Gaze 2 during Gaze 1 or Gesture 1
            if gaze2 - gesture1_last > 5: # formerly gaze2 > gesture1_last
                logging.error(scene_id + ':The pedestrian that doesn\'t start the interaction actually doesn\'t see the gesture of the other.')
                #defective_scene = defective_scene => don't move file since only soft requirement

            if not defective_scene:
                return

        except IndexError:
            logging.error(scene_id + ':Some pedestrian is either not looking at the other (in time) or is not making a gesture (in time).')
        except:
            logging.error(scene_id + ':Unknown error in "clear_after_annotations".')
            print(scene_id)
        
        Path(curr_dir / 'dbn_annotations.csv').rename(Path(curr_dir / 'defective_dbn_annotations.csv'))



global scene, refl, scene_id
global p1_crossing, p2_crossing
global max_distance
max_distance = 0

def run_all(root_dir):
    """Creates the annotations file for each scene.

    Args:
        root_dir (str): path of directory containing all scenes

    Returns:
        num: the maximal distance between two pedestrians in the overall dataset
    """
    # fresh log files for fresh run
    with open(logfile_path, 'w'), open(logfile_error_path, 'w'):
        pass

    global scene, refl, scene_id
    refl_dict = {'no-reflection' : 'nr', 'horizontal-reflection' : 'hr', 'vertical-reflection' : 'vr', 'vertical-horizontal-reflection' : 'vh'}
    num_scenes = 0

    for root, _, files in os.walk(root_dir):
        if 'pedestrian 1 features.csv' in files and 'pedestrian 2 features.csv' in files:
            num_scenes += 1
            if num_scenes % 1000 == 0:
                print(num_scenes)
            if 'dbn_annotations.csv' in files:
                warnings.warn('The file "dbn_annotations.csv" is already contained in "' + str(root) + '" and is not recreated. This behavior applies to all directories.')
                continue
            relative_path = root[len(root_dir) + len(os.sep):]
            if relative_path.count(os.sep) == 0:
                scene, refl, num, _ = relative_path.split('_')
                scene_id = scene + '_' + refl_dict[refl] + '_' + num
            else:
                print('Entcountered an old directory organization structure. This will be ignored.')
                print(root)
                continue
            df_p1, df_p2 = clear_before_annotations(root)
            if df_p1 is not None: 
                df = scene_annotation(df_p1, df_p2, Path(root))
                if df is not None:
                    clear_after_annotations(df, Path(root))

    logging.error('Total number of scenes : ' + str(num_scenes))
    return max_distance