import numpy as np
import os
import glob
import argparse

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

        imas1 = data[i+1][imas1_predictions[:]]
        imas2 = data[i+1][imas2_predictions[:]]

        ##
        p1_imas = np.dot(imas1, imas)
        p2_imas = np.dot(imas2, imas)

        imas1_values.append(p1_imas)
        imas2_values.append(p2_imas)

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


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Path to dataset")
args = parser.parse_args()
get_data(args.data)
