import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import math

def merge_csv(path):

    all_files = glob.glob(os.path.join(path, "*.csv"))

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_files], axis=1, sort=False)
    # export to csv
    combined_csv.to_csv(path+"combined_csv.csv", index=False)


#merge_csv('data/Generated EMIDAS Dataset 2020-04-17 21h 38m 47s/')

def extract_data(file):
    imas_values = [5, 4, 1, 0.5, 0.2]
    p1_xy_ = []
    p2_xy_ = []
    p1_pull = []
    p2_pull = []

    data = np.genfromtxt(file, delimiter=',')
    for sample in range(1, len(data)):
        #pedestrians 1,2 imas values dot product
        p1_imas = np.dot(data[sample][2:7], imas_values)
        p2_imas = np.dot(data[sample][8:13], imas_values)
        p1_xy = np.asarray([data[sample][-5], data[sample][-4]])
        p2_xy = np.asarray([data[sample][-2], data[sample][-1]])
        p1_imas_pull = p2_xy - p1_xy
        p1_imas_pull_norm =  p1_imas_pull / np.linalg.norm(p1_imas_pull)
        p1_imas_pull_norm *= p1_imas
        p2_imas_pull = p1_xy - p2_xy
        p2_imas_pull_norm =  p2_imas_pull / np.linalg.norm(p2_imas_pull)
        p2_imas_pull_norm *= p2_imas
        p1_xy_.append(p1_xy)
        p2_xy_.append(p2_xy)
        p1_pull.append(p1_imas_pull_norm)
        p2_pull.append(p2_imas_pull_norm)

    p1_xy_ = np.reshape(p1_xy_, [len(p1_xy_), 2])
    p2_xy_ = np.reshape(p2_xy_, [len(p2_xy_), 2])
    p1_pull = np.reshape(p1_pull, [len(p1_pull), 2])
    p2_pull = np.reshape(p2_pull, [len(p2_pull), 2])

    return p1_xy_, p2_xy_, p1_pull, p2_pull

def get_obs_pred(data, observed_frame_num, predicting_frame_num, pos=True):
    obs = []
    pred = []
    count = 0

    if len(data) >= observed_frame_num + predicting_frame_num:
        seq = int((len(data) - (observed_frame_num + predicting_frame_num)) / observed_frame_num) + 1

        for k in range(seq):
            obs_pedIndex = []
            pred_pedIndex = []
            count += 1
            for i in range(observed_frame_num):
                obs_pedIndex.append(data[i + k * observed_frame_num])
            for j in range(predicting_frame_num):
                pred_pedIndex.append(data[k * observed_frame_num + j + observed_frame_num])

            if pos == True:
                obs_pedIndex = np.reshape(obs_pedIndex, [observed_frame_num, 2])
                pred_pedIndex = np.reshape(pred_pedIndex, [predicting_frame_num, 2])
            else:
                obs_pedIndex = np.reshape(obs_pedIndex, [observed_frame_num, 1])
                pred_pedIndex = np.reshape(pred_pedIndex, [predicting_frame_num, 1])

            obs.append(obs_pedIndex)
            pred.append(pred_pedIndex)

    if pos==True:
        obs = np.reshape(obs, [count, observed_frame_num, 2])
        pred = np.reshape(pred, [count, predicting_frame_num, 2])
    else:
        obs = np.reshape(obs, [count, observed_frame_num, 1])
        pred = np.reshape(pred, [count, predicting_frame_num, 1])

    return np.expand_dims(obs[0], axis=0), np.expand_dims(pred[0], axis=0)


def plot_data(pos1, pos2, pull1, pull2):
    plt.scatter(pos1[:, 1], pos1[:, 0], label='pedestrian 1 location')
    plt.scatter(pos2[:, 1], pos2[:, 0], label='pedestrian 2 location')
    origin = [pos1[:,1]], [pos1[:,0]]  # origin point
    origin1 = [pos2[:, 1]], [pos2[:, 0]]  # origin point
    q =plt.quiver(*origin, pull1[:, 1], pull1[:, 0], scale=21, width=0.001)
    plt.quiver(*origin1, pull2[:, 1], pull2[:, 0], scale=21, width = 0.001)
    plt.quiverkey(q, X=0.9, Y=0.9, U=1,
             label='Pull towards other', labelpos='E')
    plt.legend()
    plt.show()

def get_final_data():
    pos1, pos2, pull1, pull2 = extract_data('data/Generated EMIDAS Dataset 2020-04-17 21h 38m 47s/combined_csv.csv')
    obs, pred = get_obs_pred(pos1, 10, 10)

    return obs, pred

#pos1, pos2, pull1, pull2 = extract_data('data/Generated EMIDAS Dataset 2020-04-17 21h 38m 47s/combined_csv.csv')
#plot_data(pos1, pos2, pull1, pull2)

def read_imas(file, pos1, pos2):

    data = np.genfromtxt(file, delimiter=',')
    imas_1_index = 1
    imas_2_index = 7
    imas = [5, 4, 3, 2, 1]
    imas1_predictions = [2, 3, 4, 5, 6]
    imas2_predictions = [8, 9, 10, 11, 12]
    imas1_values = []
    imas2_values = []
    for i in range(len(data)-1):
        #
        #p1_imas = data[i + 1][imas_1_index]
        #p2_imas = data[i + 1][imas_2_index]
        #
        imas1 = data[i+1][imas1_predictions[:]]
        imas2 = data[i+1][imas2_predictions[:]]

        #
        # p1_imas_pull = np.asarray(pos2[i]) - np.asarray(pos1[i])
        # p1_imas_pull_norm = p1_imas_pull / np.linalg.norm(p1_imas_pull)
        # p1_imas_pull_norm *= imas1
        # p2_imas_pull = np.asarray(pos1[i]) - np.asarray(pos2[i])
        # p2_imas_pull_norm = p2_imas_pull / np.linalg.norm(p2_imas_pull)
        # p2_imas_pull_norm *= imas2
        #

        ##
        p1_imas = np.dot(imas1, imas)
        p2_imas = np.dot(imas2, imas)

        #imas1_val = imas[np.argmax(imas1)]
        #imas2_val = imas[np.argmax(imas2)]

        ##

        imas1_values.append(p1_imas)
        imas2_values.append(p2_imas)

    #imas1_values = np.reshape(imas1_values, [len(imas1_values), 2])
    #imas2_values = np.reshape(imas2_values, [len(imas2_values), 2])
    #imas1_values -= imas1_values[0]
    #imas2_values -= imas2_values[0]

    return imas1_values, imas2_values

def read_positions(file):

    data = np.genfromtxt(file, delimiter=',')
    pos1_x_index = 2
    pos1_y_index = 3
    pos2_x_index = 5
    pos2_y_index = 6
    pos1_values = []
    pos2_values = []
    for i in range(len(data)-1):
        pos1_values.append([data[i + 1][pos1_x_index], data[i + 1][pos1_y_index]])
        pos2_values.append([data[i + 1][pos2_x_index], data[i + 1][pos2_y_index]])

    pos1_values = np.reshape(pos1_values, [len(pos1_values), 2])
    pos2_values = np.reshape(pos2_values, [len(pos2_values), 2])

    return pos1_values, pos2_values

def get_data(path):
    observed_frame_num = 40
    predicting_frame_num = 30
    #imas_obs = []
    #pos_obs = []
    #pred_obs = []
    files = []
    count = 0
    for r, d, f in os.walk(path):
        for file in f:
            if file  == "dbn_prediction.csv":
                imas_file = os.path.join(r, file)
                positions_file = os.path.join(r, "pedestrian_positions.csv")
                pos1, pos2 = read_positions(positions_file)
                imas1, imas2 = read_imas(imas_file, pos1, pos2)
                obs_imas1, _ = get_obs_pred(imas1, observed_frame_num, predicting_frame_num, pos=False)
                obs_imas2, _ = get_obs_pred(imas2, observed_frame_num, predicting_frame_num, pos=False)

                obs_pos1, pred_pos1 = get_obs_pred(pos1, observed_frame_num, predicting_frame_num, pos=True)
                obs_pos2, pred_pos2 = get_obs_pred(pos2, observed_frame_num, predicting_frame_num, pos=True)
                for i in range(obs_pos1.shape[0]*2):
                    files.append(r)
                if count==0:
                    imas_obs = obs_imas1
                    imas_obs = np.concatenate((imas_obs, obs_imas2), axis=0)
                    pos_obs = obs_pos1
                    pos_obs = np.concatenate((pos_obs, obs_pos2), axis=0)
                    pred_obs = pred_pos1
                    pred_obs = np.concatenate((pred_obs, pred_pos2), axis=0)
                    count += 1
                    continue
                imas_obs = np.concatenate((imas_obs, obs_imas1), axis=0)
                imas_obs = np.concatenate((imas_obs, obs_imas2), axis=0)
                pos_obs = np.concatenate((pos_obs, obs_pos1), axis=0)
                pos_obs = np.concatenate((pos_obs, obs_pos2), axis=0)
                pred_obs = np.concatenate((pred_obs, pred_pos1), axis=0)
                pred_obs = np.concatenate((pred_obs, pred_pos2), axis=0)


    #print(imas_obs.shape)
    #print(pos_obs.shape)
    #print(pred_obs.shape)
    #print(len(files))
    np.save('data/imas_obs_40.npy', imas_obs)
    np.save('data/pos_obs_40.npy', pos_obs)
    np.save('data/pos_pred_30.npy', pred_obs)
    np.save('data/files_obs40_pred30.npy', files)


def train_val_split(files):

    sets = ['01', '02', '03', '04', '05', '06', '08']
    data = []
    for f in files:
        set = os.path.basename(os.path.normpath(f))[0:2]
        data.append(set)

    for s in sets:
        test_indices = [i for i, x in enumerate(data) if x == s]
        train_indices = [i for i, x in enumerate(data) if x != s]
        yield train_indices, test_indices


def normalize(trajectories, shift_x=None, shift_y=None, scale=None):
    """
    This is the forward transformation function which transforms the
    trajectories to a square region around the origin such as
    [-1,-1] - [1,1] while the original trajectories could be in
    pixel unit or in metres. The transformation would be applied
    to the trajectories as follows:
      new_trajectories = (old_trajectories - shift) / scale
    :param trajectories: must be a 3D Python array of the form
           Ntrajectories x Ntime_step x 2D_x_y
    :param shift_x, shift_y: the amount of desired translation.
           If not given, the centroid of trajectory points
           would be used.
    :param scale: the desirable scale factor. If not given, the scale
          would be computed so that the trajectory points fall inside
          the [-1,-1] to [1,1] region.
    :return new_trajectories: the new trajectories after the
            transformation.
    :return shift_x, shift_y, scale: if these arguments were not
            supplied, then the function computes and returns them.
    The function assumes that either all the optional parameters
    are given or they are all not given.
    """
    if shift_x is not None:
        shift = np.array([shift_x, shift_y]).reshape(1, 2)
        new_trajectories = deepcopy((trajectories - shift) / scale)
        return new_trajectories
    else:
        shift = np.mean(trajectories, axis=(0,1)).reshape(1, 2)
        new_trajectories = deepcopy(trajectories - shift)
        minxy = np.min(new_trajectories, axis=(0,1))
        maxxy = np.max(new_trajectories, axis=(0,1))
        scale = np.max(maxxy - minxy) / 2.0
        new_trajectories /= scale
        return new_trajectories, shift[0,0], shift[0,1], scale

def unnormalize(trajectories, shift_x, shift_y, scale):
    """
    This function does the inverse transformation to bring the trajectories
    back to their original coordinate system (in pixels or in metres).
    :param trajectories: must be a 3D Python array of the form
           Ntrajectories x Ntime_step x 2D_x_y
    :param shift_x, shift_y, scale: these should be the same
           parameters as the function 'normalise' above.
    :return new_trajectories: the new trajectories after the
            transformation.
    """
    shift = np.array([shift_x, shift_y])
    new_trajectories = deepcopy((trajectories * scale) + shift)
    return new_trajectories


def calc_fde_meters(pred, gt):

    #1 prediction
    fde_ = []
    final_frame = gt.shape[1] - 1
    #final_frame = 0

    if len(pred.shape) > 3:
        for i in range(pred.shape[0]):
            best_fde = 999999
            for j in range(pred.shape[1]):
                fde = math.sqrt((pred[i][j][final_frame][0] - gt[i][final_frame][0]) ** 2 + (pred[i][j][final_frame][1] - gt[i][final_frame][1]) ** 2)

                if fde < best_fde:
                    best_fde = fde

            fde_.append(best_fde)
        av_fde = np.mean(fde_)

    else:
        #for i in range(pred.shape[0]):
            #fde = math.sqrt((pred[i][final_frame][0] - gt[i][final_frame][0]) ** 2 + (pred[i][final_frame][1] - gt[i][final_frame][1]) ** 2)
            #fde_.append(fde)

        #av_fde = np.mean(fde_)
        av_fde = np.mean(np.sqrt(np.sum((pred[:, -1, :] - gt[:, -1, :]) ** 2, axis=-1)))

    return av_fde

def calc_ade_meters(pred, gt):

    #1 prediction
    ade_ = []
    if len(pred.shape) > 3:
        for i in range(pred.shape[0]):
            best_ade = 999999
            for j in range(pred.shape[1]):
                ade_temp = []
                for k in range(pred.shape[2]):
                    ade = math.sqrt((pred[i][j][k][0] - gt[i][k][0]) ** 2 + (pred[i][j][k][1] - gt[i][k][1]) ** 2)
                    ade_temp.append(ade)

                if np.mean(ade_temp) < best_ade:
                    best_ade = np.mean(ade_temp)

            ade_.append(best_ade)
        av_ade = np.mean(ade_)

    else:
        #for i in range(pred.shape[0]):
            #for k in range(pred.shape[1]):
                #ade = math.sqrt((pred[i][k][0] - gt[i][k][0]) ** 2 + (pred[i][k][1] - gt[i][k][1]) ** 2)
                #ade_.append(ade)

        #av_ade = np.mean(ade_)
        av_ade = np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=-1)))

    return av_ade

#get_data('data/6Jun2020 Complete Dataset/')
#files = np.load('data/files_obs20_pred10.npy')

# d = np.load('data/pos_obs.npy')
# print(d[123])
# n, shift1, shift2, scale = normalize(d)
# print(n[1233])
# u = unnormalize(n, shift1, shift2, scale)
# print(u[123])
#pos1, pos2 = read_positions("data/6Jun2020 Complete Dataset/0101_horizontal-reflection_100_1/pedestrian_positions.csv")
#obs_pos1, pred_pos1 = get_obs_pred(pos1, 50, 20, pos=True)
#imas1, imas2 = read_imas("data/6Jun2020 Complete Dataset/03a02_horizontal-reflection_1_1/dbn_prediction.csv", pos1, pos2)

# print(imas1)
# print(imas2)
#plot_data(pos1, pos2, imas1, imas2)