# Running on GPU?
import setGPU

import getpass
import h5py
import os
import pickle

from tqdm import tqdm

# Get permission to access EOS (Insert your NICE password)
os.system("echo %s | kinit" % getpass.getpass())

# ## Load data, and labels

import json
import numpy as np
import pandas as pd

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Lambda
from keras.layers.advanced_activations import PReLU
from keras.activations import sigmoid, linear, selu
from keras.models import Model, load_model

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PDs  = {1: 'BTagCSV',
        2: 'BTagMu',
        3: 'Charmonium',
        4: 'DisplacedJet',
        5: 'DoubleEG',
        6: 'DoubleMuon',
        7: 'DoubleMuonLowMass',
        8: 'FSQJets',
        9: 'HighMultiplicityEOF',
        10: 'HTMHT',
        11: 'JetHT',
        12: 'MET',
        13: 'MinimumBias',
        14: 'MuonEG',
        15: 'MuOnia',
        16: 'NoBPTX',
        17: 'SingleElectron',
        18: 'SingleMuon',
        19: 'SinglePhoton',
        20: 'Tau',
        21: 'ZeroBias'}

# Select PD
nPD = 11

data_directory = "/eos/cms/store/user/fsiroky/consistentlumih5/"
label_file = "/afs/cern.ch/user/t/tkrzyzek/Documents/Data-Certification/JetHT.json"
model_directory = "/eos/user/t/tkrzyzek/autoencoder/lumi_dep/split01_selu/"
model_name = "model"

def get_file_list(directory, pds, npd, typeof, extension):
    files = []
    parts = ["C", "D", "E", "F", "G", "H"]
    for p in parts:
        files.append("%s%s_%s_%s%s" % (directory, pds[npd], p, typeof, extension))
    return files

files = get_file_list(data_directory, PDs, nPD, "background", ".h5")
files = files + get_file_list(data_directory, PDs, nPD, "signal", ".h5")

# Load good and bad jets
def get_data(files):
    readout = np.empty([0,2813])
    
    for file in files:
        jet = file.split("/")[-1][:-3]
        print("Reading: %s" % jet)
        try:
            h5file = h5py.File(file, "r")
            readout = np.concatenate((readout, h5file[jet][:]), axis=0)
        except OSError as error:
            print("This Primary Dataset doesn't have %s. %s" % (jet, error))
            continue

    return readout

data = pd.DataFrame(get_data(files))

data["run"] = data[2807].astype(int)
data["lumi"] = data[2808].astype(int)
data["inst_lumi"] = data[2809].astype(float)

# Drop unnecessary meta data
data.drop([2807, 2808, 2809, 2810, 2811, 2812], axis=1, inplace=True)

# Sort by runID and then by lumiID
data = data.sort_values(["run", "lumi"], ascending=[True,True])

# Reset index
data = data.reset_index(drop=True)  

runIDs  = data["run"].astype(int)
lumiIDs = data["lumi"].astype(int)
luminosity = data["inst_lumi"].astype(float)

# Apply labels
output_json = json.load(open(label_file))

def json_checker(json_file, orig_runid, orig_lumid):
    try:
        for i in json_file[str(int(orig_runid))]:
            if orig_lumid >= i[0] and orig_lumid <= i[1]:
                return 0
    except KeyError:
        pass
    return 1

def add_flags(sample):
    return json_checker(output_json, sample["run"], sample["lumi"])

data["label"] = data.apply(add_flags, axis=1)

# Split the data
SPLIT_FACTOR = 0.1

split = round(SPLIT_FACTOR*len(data))

runIDs = runIDs[split:]
lumiIDs = lumiIDs[split:]
luminosity = luminosity[split:]

train = data.iloc[:split]
X_train = train.iloc[:, 0:2806]
y_train = train["label"]

test = data.iloc[split:]
X_test = test.iloc[:, 0:2806]
y_test = test["label"]
    
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# Train only on good

X_train = X_train[y_train == 0]

input_dim = X_train.shape[1]

# Define the model

input_layer = Input(shape=(input_dim, ))
prellll = Lambda(selu)
encoded = Dense(2000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(input_layer)
encoded = prellll(encoded)

prellll = Lambda(selu)
encoded = Dense(1000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
encoded = prellll(encoded)

prellll = Lambda(selu)
encoded = Dense(500, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
encoded = prellll(encoded)

prellll = Lambda(selu)
encoded = Dense(1000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
encoded = prellll(encoded)

prellll = Lambda(selu)
encoded = Dense(2000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
encoded = prellll(encoded)

prellll = Lambda(selu)
decoder = Dense(input_dim)(encoded)
decoder = prellll(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.summary()

adamm = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

early_stopper = EarlyStopping(monitor="val_loss",
                              patience=32,
                              verbose=True,
                              mode="auto")

autoencoder.compile(optimizer=adamm, loss='mean_squared_error')

checkpoint_callback = ModelCheckpoint(("%s%s.h5" % (model_directory, model_name)),
                                      monitor="val_loss",
                                      verbose=False,
                                      save_best_only=True,
                                      mode="min")

autoencoder.fit(X_train,
                X_train,
                epochs=2048,
                batch_size=256,
                shuffle=True,
                validation_split=0.25,
                verbose=2,
                callbacks=[early_stopper, checkpoint_callback])

# Reload saved model
autoencoder = load_model("%s%s.h5" % (model_directory, model_name))

# Run predictions
predictions = autoencoder.predict(X_test)

def get_error_df(X_test, predictions, mode="allmean", n_highest = 100):
    
    if mode == "allmean":
        return np.mean(np.power(X_test - predictions, 2), axis=1)
    
    elif mode == "topn":
        temp = np.partition(-np.power(X_test - predictions, 2), n_highest)
        result = -temp[:,:n_highest]
        return np.mean(result, axis=1)
    
    elif mode == "perobj":
        mses = []
        for l in legend:
            mse = np.mean(
                np.power(X_test[:,l["start"]:l["end"]] - predictions[:,l["start"]:l["end"]], 2),
                axis=1)
            mses.append(mse)
     
        return np.maximum.reduce(mses)
    
ae_error = get_error_df(X_test, predictions, mode="topn")

pickle.dump(predictions, open(model_directory + "ae_pred.p", "wb"))
pickle.dump(ae_error, open(model_directory + "ae_error.p", "wb"))