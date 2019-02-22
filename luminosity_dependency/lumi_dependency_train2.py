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
from keras.layers import Input, Dense
from keras.layers.advanced_activations import PReLU
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
label_file = "./JetHT.json"
model_directory = "/eos/user/t/tkrzyzek/autoencoder/lumi_dep/split01/"
model_name = "model01"

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

print("Number of inliers in test set: %s" % sum((y_test == 0).values))
print("Number of anomalies in the test set: %s" % sum((y_test == 1).values))

ae_scores = []
ms_scores = []
rf_scores = []
true_labels = []
inliers = []
outliers = []
    
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

inliers.append(sum((y_test == 0).values))
outliers.append(sum((y_test == 1).values))

# Calculate class weights
#classes = np.unique(y_train)
#weights = class_weight.compute_class_weight('balanced', classes, y_train)
#cw = {int(cls): weight for cls, weight in zip(classes, weights)}

params = {
    "max_depth": 7,
    "n_estimators": 64, 
    "random_state": 42, 
    "n_jobs": -1,
    "verbose" : 1,
#    "class_weight": cw
}

rf = RandomForestClassifier(**params)
rf.fit(X_train, y_train)
rf_score = rf.predict_proba(X_test)[:, 1]

# Train only on good

X_train = X_train[y_train == 0]

input_dim = X_train.shape[1]

# Define the model

input_layer = Input(shape=(input_dim, ))
prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(2000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(input_layer)
#encoded = Dense(2000)(input_layer)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(1000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
#encoded = Dense(1000)(encoded)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(500, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
#encoded = Dense(500)(encoded)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(1000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
#encoded = Dense(1000)(encoded)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(2000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
#encoded = Dense(2000)(encoded)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
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

history = autoencoder.fit(X_train,
                          X_train,
                          epochs=2048,
                          batch_size=256,
                          shuffle=True,
                          validation_split=0.25,
                          verbose=2,
                          callbacks=[early_stopper, checkpoint_callback]).history

# Reload saved model
autoencoder = load_model("%s%s.h5" % (model_directory, model_name))

# Run predictions
predictions = autoencoder.predict(X_test)

# Mean square
mean_square = np.mean(np.power(X_test, 2), axis=1)

ms_scores.append(mean_square)
rf_scores.append(rf_score)
true_labels.append(y_test.values)

pickle.dump(X_test, open(model_directory + "x_test.p", "wb"))
pickle.dump(predictions, open(model_directory + "ae_pred.p", "wb"))
pickle.dump(ms_scores, open(model_directory + "ms_scores.p", "wb"))
pickle.dump(rf_scores, open(model_directory + "rf_scores.p", "wb"))
pickle.dump(true_labels, open(model_directory + "true_labels.p", "wb"))
pickle.dump(inliers, open(model_directory + "inliers.p", "wb"))
pickle.dump(outliers, open(model_directory + "outliers.p", "wb"))
pickle.dump(luminosity, open(model_directory + "luminosity.p", "wb"))
   