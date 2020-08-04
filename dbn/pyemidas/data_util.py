import csv, logging, os, random
import pandas as pd
import config as cfg
from pathlib import Path
from scipy.stats import pearsonr
from shutil import copyfile
import numpy as np

# the keys contain the variables name used in the 
renaming_cols = {'imas_1' : 'p1_imas', 'wti_1' : 'p1_wti', 'viewing_area_1' : 'p1_head', 'body_facing_1' : 'p1_body', 'gesture_1' : 'p1_gesture', 'approaching_1' : 'p1_approach', 'imas_2' : 'p2_imas', 'wti_2' : 'p2_wti', 'viewing_area_2' : 'p2_head', 'body_facing_2' : 'p2_body', 'gesture_2' : 'p2_gesture', 'approaching_2' : 'p2_approach', 'distance' : 'dist'}

def append_index(i):
    """Creates functions that appends slice number to input.

    Args:
        i (int): the slice number

    Returns:
        str -> str: lambda function
    """
    if i == 0:
        return lambda id : id
    return lambda column_name : column_name + '_' + str(i)


def is_train_annotations_file(path_to_file):
    """Checks whether the file is the annotations file of a training data scene. This fully relies on naming convention.

    Args:
        path_to_file (str): absolute path of file

    Returns:
        bool: True if file is the annotations file of a training data scene, False otherwise
    """
    folder, f = Path(path_to_file).parts[-2:]
    return f == 'dbn_annotations.csv' and cfg.train_set_check(folder)
    

def is_test_annotations_file(path_to_file):
    """Checks whether the given path leads to the annotations file of a test scene. This function fully relies on naming conventions.

    Args:
        path_to_file (str or pathlib.Path): path of the fule concerned

    Returns:
        bool: True if the file contains the annotations of a test scene, False otherwise
    """
    folder, f = Path(path_to_file).parts[-2:]
    return f == 'dbn_annotations.csv' and cfg.test_set_check(folder)


def is_test_prediction_file(path_to_file):
    """Checks whether the given path leads to the prediction file of a test scene. This function fully relies on naming conventions.

    Args:
        path_to_file (str or pathlib.Path): path of the file concerned

    Returns:
        bool: True if file contains the prediction of a test scene, False otherwise
    """
    folder, f = Path(path_to_file).parts[-2:]
    return f == 'dbn_prediction.csv' and cfg.test_set_check(folder)



def get_max_distance(dataset_root):
    """Goes through all training data scenes and returns the maximal distance between two pedestrians. This is needed to set the values of the distance variable in the DBN.

    Args:
        dataset_root (str): directory containing all scenes directories

    Returns:
        int: maximal distance of all training data scenes
    """
    max_distance = 0
    for root, _, files in os.walk(dataset_root):
        for f in files:
            path_to_file = Path(root) / f
            if is_train_annotations_file(path_to_file):
                df = pd.read_csv(path_to_file)
                scene_max_distance = int(df.max(axis=0)['distance'])
                if max_distance < scene_max_distance:
                    max_distance = scene_max_distance
    assert max_distance > 0
    return max_distance


def choose_trianing_scenes():
    """Function to randomly choose subset of OpenDS-CTS02.
    """
    root_dir = '/Users/nora/Downloads/6Jun2020 Complete Dataset'
    scenes = {}

    for root, _, files in os.walk(root_dir):
        if 'dbn_annotations.csv' in files:
            directory = Path(root).name
            scene, refl, num, _ = directory.split('_')
            if scene + '_' + refl in scenes:
                scenes[scene + '_' + refl].append(num)
            else:
                scenes[scene + '_' + refl] = [num]

    scenes = {key: sorted(val) for key, val in scenes.items()}
    sorted_scenes = dict(sorted(scenes.items()))
    scenes_len = {key: len(val) for key, val in sorted_scenes.items()}

    # with open('scenes_dict.txt', 'w') as out:
    #     for key, val in sorted_scenes.items():
    #         out.write(key + ' : ' + str(val) + '\n')
    #     out.write('\n\n')
    #     for key, val in scenes_len.items():
    #         out.write(key + ' : ' + str(val) + '\n')
    
    scenario_num = {'01' : (8,4), '02' : (24,4), '03' : (4,36), '04' : (16,4), '05' : (12,4), '06' : (49,0), '08' : (24,4)}

    chosen_scenes = {}

    for scenario in scenario_num:
        for key, val in sorted_scenes.items():
            if key.startswith(scenario):
                n = scenario_num[scenario][0]
                if scenario_num[scenario][1] > 0:
                    n += 1
                    scenario_num[scenario] = (scenario_num[scenario][0], scenario_num[scenario][1]-1)
                if len(val) < n:
                    diff = n - len(val)
                    scenario_num[scenario] = (scenario_num[scenario][0], scenario_num[scenario][1]+diff)
                    n = len(val)
                chosen_scenes[key] = random.sample(val, n)

    print(scenario_num)

    for scenario, (_, remaining) in scenario_num.items():
        while remaining > 0:
            for key, val in sorted_scenes.items():
                if key.startswith(scenario):
                    if set(val) != set(chosen_scenes[key]):
                        scene = random.choice(val)
                        while scene in chosen_scenes[key]:
                            scene = random.choice(val)
                        chosen_scenes[key].append(scene)
                        remaining -= 1
                        if remaining == 0:
                            break
                    

    with open('scenes_dict2.txt', 'w') as out:
        for key, val in chosen_scenes.items():
            out.write('\'' + key + '\' : ' + str(val) + ',\n')

    chosen_scenes_len = {key: len(val) for key, val in chosen_scenes.items()}

    scenario_amount = {'01' : 0, '02' : 0, '03' : 0, '04' : 0, '05' : 0, '06' : 0, '08' : 0}
    for scenario in scenario_amount:
        for key, val in chosen_scenes_len.items():
            if key.startswith(scenario):
                scenario_amount[scenario] += val
    print(scenario_amount)


def calculate_correlation(directory):
    """Function that computes PCC of the predictions and ground truth.

    Args:
        directory (pathlib.Path): directory containing the scenes that should be used fot the PCC computation
    """
    data = {'p1_imas' : {'x' : [], 'y1' : [], 'y2' : []}, # x: ground truth, y1: highest prob, y2: computed value
            'p2_imas' : {'x' : [], 'y1' : [], 'y2' : []},
            'p1_wti' : {'x' : [], 'y1' : [], 'y2' : []},
            'p2_wti' : {'x' : [], 'y1' : [], 'y2' : []}}

    for root, _, files in os.walk(directory):
        if 'dbn_prediction.csv' in files:
            df = pd.read_csv(Path(root) / 'dbn_prediction.csv', usecols=(lambda col : 'imas' in col or 'wti' in col))
            for target in ['p1_imas', 'p2_imas', 'p1_wti', 'p2_wti']:
                subset = [target + '_' + str(i) + '_pred' for i in range(1,6)]
                factors = list(range(1,6))
                df[target + '_most_prob'] = df[subset].apply(lambda row: int(row.idxmax()[len(target)+1:-5]), axis=1)
                df[target + '_val'] = df[subset].apply(lambda row: sum([row[subset[i]]*factors[i] for i in range(len(factors))]), axis=1)

                data[target]['x'].extend(list(df[target]))
                data[target]['y1'].extend(list(df[target + '_most_prob']))
                data[target]['y2'].extend(list(df[target + '_val']))
   
    with open('pcc-values.txt', 'a') as f:
        for key in data:
            f.write(directory + '\n')
            f.write(key + ' pcc(ground_truth, highest_prob)' + '\n')
            f.write(str(pearsonr(data[key]['x'], data[key]['y1'])) + '\n')
            f.write(key + ' pcc(ground_truth, aggregated_value)' + '\n')
            f.write(str(pearsonr(data[key]['x'], data[key]['y2'])) + '\n')


def select_predictions(directory_str, small=False):
    """Since the predictions of each EMIDAS_DBN were saved in separate directory, this function copies the predictions files into the OpenDS-CTS02 folder.

    Args:
        directory_str (str): name of "lowest" directory containing the scenes (slice[1-6]0) 
        small (bool, optional): whether the results of the subset should be used, defaults to False.
    """
    num_folders = 0
    dataset = Path(cfg.dataset_root)
    for folder in dataset.glob('*'):
        if (folder / 'dbn_prediction.csv').exists():
            (folder / 'dbn_prediction.csv').unlink()
        num_folders += 1

    num_predictions = 0
    predictions = dataset.parent / 'predictions'
    if small:
        scenarios = ['01-small', '02-small', '03-small', '04-small', '05-small', '06-small', '08-small']
    else:
        scenarios =  ['01', '02', '03', '04', '05', '06', '08']
    for scenario in scenarios:
        scenario_predictions = predictions / scenario / directory_str
        for folder in scenario_predictions.glob('*_1'): # Hack damit nur die Ordner gematched werden, wie geht es rihtig????
            assert (dataset / folder.name).exists(), str(dataset / folder.name) + ' does not exist.'
            # print(str(folder / 'dbn_prediction.csv'))
            # print(str(dataset / folder.name / 'dbn_prediction.csv'))
            copyfile(str(folder / 'dbn_prediction.csv'), str(dataset / folder.name / 'dbn_prediction.csv'))
            num_predictions += 1
    
    print(num_folders, num_predictions) # should be 16880 15949


def confusion_matrix(root):
    """Given a directory containing predictions, the functions creates confusion matices fot the variables ['p1_imas', 'p2_imas', 'p1_wti', 'p2_wti'].

    Args:
        root (str): directory containing predictions
    """
    matrices = {'p1_imas': np.zeros(shape=(5,5)), 'p2_imas': np.zeros(shape=(5,5)), 'p1_wti': np.zeros(shape=(5,5)), 'p2_wti': np.zeros(shape=(5,5))}
    for subdir, _, files in os.walk(root):
        for f in files:
            path_to_f = Path(subdir) / f
            if is_test_prediction_file(path_to_f):
                df_pred = pd.read_csv(path_to_f, usecols=(lambda col : 'imas' in col or 'wti' in col))
                for target in ['p1_imas', 'p2_imas', 'p1_wti', 'p2_wti']:
                    for _, row in df_pred.iterrows():
                        predictions = {}
                        for outcome in range(1,6):
                            predictions[outcome] = row[target + '_' + str(outcome) + '_pred']
                        pred = max(predictions, key=predictions.get)
                        ground_truth = int(row[target])
                        matrices[target][ground_truth-1, pred-1] +=1
                        
    if cfg.train_set == 'small':
        conf_matrix_path = 'output_files/confusion_matrix_' + str(Path(cfg.dbn_path).stem) + '_' + cfg.test_scenario + '-small.txt'
    else:
        conf_matrix_path = 'output_files/confusion_matrix_' + str(Path(cfg.dbn_path).stem) + '_' + cfg.test_scenario + '.txt'
    with open(conf_matrix_path, 'w') as f:
        for target, matrix in matrices.items():
            f.write(target + '\n')
            f.write('rows: ground truth, columns: prediction, 0: very_low ... 4: very_high\n')
            np.savetxt(f, matrix, fmt='%d')
            f.write('\n\n')

            