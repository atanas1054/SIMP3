import keras
from utils import *
from keras.backend.tensorflow_backend import set_session
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Dense, Lambda, RepeatVector
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
print(keras.__version__)
print(tf.__version__)

stop = EarlyStopping(monitor='val_loss',min_delta=0.000000000001,patience=2)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)

def concat_layers(l):
    l1 = l[0]
    l2 = l[1]
    c = tf.stack([l1, l2], axis=2)
    c = tf.reshape(c, [-1, 256])
    return c

observed_frame_num = 40
predicting_frame_num = 30
lstm_hidden = 128
fc_hidden = 128
scenario = 1

observed_traj = Input(shape=(observed_frame_num, 2))
imas = Input(shape=(observed_frame_num, 1))

attention_traj = TimeDistributed(Dense(128, activation='relu'))(observed_traj)
attention_imas = TimeDistributed(Dense(128, activation='relu'))(imas)

encoder_traj = LSTM(lstm_hidden, return_sequences=False)(attention_traj)
encoder_imas = LSTM(lstm_hidden, return_sequences=False)(attention_imas)
con = Lambda(concat_layers)([encoder_traj, encoder_imas])

decoder = Dense(fc_hidden, activation='relu')(con)
decoder = RepeatVector(predicting_frame_num)(decoder)
decoder = LSTM(lstm_hidden, return_sequences=True)(decoder)
prediction = TimeDistributed(Dense(2))(decoder)

model = Model([observed_traj, imas], prediction)
model.compile(loss='mse', optimizer=Adam())
model.summary()

ws = model.get_weights()

obs = np.load('data/pos_obs_40.npy')
pred = np.load('data/pos_pred_30.npy')
imas_obs = np.load('data/imas_obs_40_predictions.npy')
files = np.load('data/files_obs40_pred30.npy')

###TRAINING/TESTING loop####
for train_index, test_index in train_val_split(files):

    obs_train_un, pred_train_un, imas_train = obs[train_index], pred[train_index], imas_obs[train_index]
    obs_test_un, pred_test, imas_test = obs[test_index], pred[test_index], imas_obs[test_index]

    #normalize data
    obs_train, _, _, _ = normalize(obs_train_un)
    pred_train, _, _, _ = normalize(pred_train_un)
    obs_test, shift1, shift2, scale = normalize(obs_test_un)
    pred_test_, _, _, _ = normalize(pred_test)

    model.set_weights(ws)
    rates = [0.001, 0.0001, 0.00001]
    #imas_train = imas_train / 5
    #imas_test = imas_test / 5

    for rate in rates:
        #print('training with lr = ' + str(rate))
        model.compile(loss='mse', optimizer=Adam(lr=rate))
        model.fit([obs_train, imas_train], pred_train, validation_data=([obs_test, imas_test], pred_test_), nb_epoch=1, callbacks=[stop], batch_size= 128, verbose=0, shuffle=True)

    preds = model.predict([obs_test, imas_test])
    preds = unnormalize(preds, shift1, shift2, scale)

    ade = calc_ade_meters(preds, pred_test)
    print("Scenario ", scenario)
    print("ADE: ", ade)
    fde = calc_fde_meters(preds, pred_test)
    print("FDE: ", fde)
    scenario +=1

