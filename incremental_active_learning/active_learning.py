import setGPU

import getpass
import h5py
import os
import pickle

from tqdm import tqdm
from sklearn.metrics import accuracy_score

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

CUTOFF_ACC = 1000
# CUTOFF_ACC = 10

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
golden_file = "/afs/cern.ch/user/t/tkrzyzek/Documents/Data-Certification/labels/Golden2016.json"
label_file = "/afs/cern.ch/user/t/tkrzyzek/Documents/Data-Certification/JetHT.json"
model_directory = "/eos/user/t/tkrzyzek/autoencoder/active_learning/"
model_name = "model"
model_base_name = "model_base"

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

def get_error_df(X_test, predictions, mode="allmean", n_highest = 100):
    
    if mode == "allmean":
        return np.median(np.power(X_test - predictions, 2), axis=1)
    
    elif mode == "topn":
        temp = np.partition(-np.power(X_test - predictions, 2), n_highest)
        result = -temp[:,:n_highest]
        return np.median(result, axis=1)
    
    elif mode == "perobj":
        mses = []
        for l in legend:
            mse = np.median(np.power(X_test[:,l["start"]:l["end"]] - predictions[:,l["start"]:l["end"]], 2),
                axis=1)
            mses.append(mse)
     
        return np.maximum.reduce(mses)
    
def find_optimal_cutoff(scores, y_true):
    step_factor = CUTOFF_ACC
    max_acc = 0
    best_threshold = None
    for threshold in tqdm(np.geomspace(min(scores), max(scores), step_factor)):
        y_pred = [1 if e > threshold else 0 for e in scores]
        acc = accuracy_score(y_true, y_pred)
        if acc > max_acc:
            max_acc = acc
            best_threshold = threshold
    return best_threshold, max_acc

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
data_train = data

output_json = json.load(open(label_file))
golden_json = json.load(open(golden_file))

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

def add_golden_flags(sample):
    return json_checker(golden_json, sample["run"], sample["lumi"])

data["label"] = data.apply(add_flags, axis=1)
data_train["label"] = data.apply(add_golden_flags, axis=1)

# Split the data
PRE_TRAIN = 0.1

split = round(PRE_TRAIN*len(data))

runIDs = runIDs[split:]
lumiIDs = lumiIDs[split:]
luminosity = luminosity[split:]

train = data_train.iloc[:split]
X_train = train.iloc[:, 0:2806]
y_train = train["label"]

test = data.iloc[split:]
X_test = test.iloc[:, 0:2806]
y_test = np.asarray(test["label"])

normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

inliers = []
outliers = []


# In[44]:

######################
# DEFINE MODELS
######################
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

# Train only on good
X_train_good = X_train[y_train == 0]

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim, ))
prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(2000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(input_layer)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(1000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(500, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(1000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
encoded = Dense(2000, kernel_regularizer=keras.regularizers.l1_l2(10e-5))(encoded)
encoded = prellll(encoded)

prellll = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
decoder = Dense(input_dim)(encoded)
decoder = prellll(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.summary()

adamm = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adamm_base = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

early_stopper = EarlyStopping(monitor="val_loss",
                              patience=32,
                              verbose=True,
                              mode="auto")
early_stopper_base = EarlyStopping(monitor="val_loss",
                              patience=32,
                              verbose=True,
                              mode="auto")

autoencoder.compile(optimizer=adamm, loss='mean_squared_error')

checkpoint_callback = ModelCheckpoint(("%s%s.h5" % (model_directory, model_name)),
                                      monitor="val_loss",
                                      verbose=False,
                                      save_best_only=True,
                                      mode="min")
checkpoint_callback_base = ModelCheckpoint(("%s%s.h5" % (model_directory, model_base_name)),
                                      monitor="val_loss",
                                      verbose=False,
                                      save_best_only=True,
                                      mode="min")

ae_scores = []
ae_cutoffs = []
ae_base_scores = []
ae_base_cutoffs = []
ae_orig_scores = []
ms_scores = []
rf_scores = []
true_labels = []

######################
# PRE-TRAIN
######################
autoencoder = load_model("%s%s.h5" % (model_directory, model_name))

#history = autoencoder.fit(X_train_good,
#                          X_train_good,
#                          epochs=2048,
#                          batch_size=256,
#                          shuffle=True,
#                          validation_split=0.25,
#                          verbose=2,
#                          callbacks=[early_stopper, checkpoint_callback]).history

ae_pred = autoencoder.predict(X_train)
ae_score = get_error_df(X_train, ae_pred, mode="topn")
ae_cutoff, ae_acc = find_optimal_cutoff(ae_score, y_train)
ae_base_cutoff = ae_cutoff

# Mean square
mean_square = np.mean(np.power(X_test, 2), axis=1)

# Random forest
rf.fit(X_train, y_train)
X_rf = X_train
y_rf = y_train

######################
# ACTIVE LEARNING LOOP
######################
FOLDS_NO = 9

ranges = np.linspace(0, len(X_test), FOLDS_NO+1)

autoencoder = load_model("%s%s.h5" % (model_directory, model_name))
autoencoder.save("%s%s.h5" % (model_directory, model_base_name))

autoencoder_orig = autoencoder

for i in range(FOLDS_NO):
    print("FOLD:", i+1)
    
    # PREDICT
    X_orig = X_test[int(ranges[i]):int(ranges[i+1])]
    X = X_orig
    y = y_test[int(ranges[i]):int(ranges[i+1])]

    inliers.append(sum((y == 0)))
    outliers.append(sum((y == 1)))
    print("Number of inliers in the subset: %s" % inliers[-1])
    print("Number of anomalies in the subset: %s" % outliers[-1])
    
    autoencoder = load_model("%s%s.h5" % (model_directory, model_name))
    autoencoder_base = load_model("%s%s.h5" % (model_directory, model_base_name))
    
    ae_pred = autoencoder.predict(X)
    ae_score = get_error_df(X, ae_pred, mode="topn")
    
    mean_square = np.mean(np.power(X, 2), axis=1)    
    
    # CHOOSE MOST INFORMATIVE SAMPLES
    RATIO = 0.2
        
    ae_cutoffs.append(ae_cutoff)
    ae_base_cutoffs.append(ae_base_cutoff)
    
    ae_diff = pd.DataFrame({'diff': np.abs(ae_score - ae_cutoff)})
    ae_diff = ae_diff.sort_values('diff').iloc[:round(RATIO*len(X))]
    ae_idx = ae_diff.index.values
    X_known_ae = X[ae_idx]
    y_known_ae = y[ae_idx]
    X_ae = X_known_ae[y_known_ae == 0]
    
    ae_base_idx = np.random.choice(len(X), round(RATIO*len(X)))
    X_known_ae_base = X[ae_base_idx]
    y_known_ae_base = y[ae_base_idx]
    X_ae_base = X_known_ae_base[y_known_ae_base == 0]

    print(len(ae_base_idx))
    # RETRAIN
    autoencoder.fit(X_ae,
                    X_ae,
                    epochs=256,
                    #epochs=2048,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.25,
                    verbose=2,
                    callbacks=[early_stopper, checkpoint_callback])
    
    autoencoder_base.fit(X_ae_base,
                    X_ae_base,
                    epochs=256,
                    #epochs=2048,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.25,
                    verbose=2,
                    callbacks=[early_stopper_base, checkpoint_callback_base])

    # PREDICT AGAIN
    ae_orig_pred = autoencoder_orig.predict(X)
    ae_orig_scores.append(ae_orig_pred)
    
    ae_pred = autoencoder.predict(X)
    ae_score = get_error_df(X, ae_pred, mode="topn")
    ae_scores.append(ae_score)
    
    ae_pred = autoencoder.predict(X_known_ae)
    ae_score = get_error_df(X_known_ae, ae_pred, mode="topn")
    ae_cutoff, ae_acc = find_optimal_cutoff(ae_score, y_known_ae)
    
    ae_pred_base = autoencoder_base.predict(X)
    ae_base_score = get_error_df(X, ae_pred_base, mode="topn")
    ae_base_scores.append(ae_base_score)
    
    ae_pred_base = autoencoder.predict(X_known_ae_base)
    ae_base_score = get_error_df(X_known_ae_base, ae_pred_base, mode="topn")
    ae_base_cutoff, ae_acc = find_optimal_cutoff(ae_base_score, y_known_ae_base)
    print(ae_base_cutoff)
    mean_square = np.mean(np.power(X, 2), axis=1)
    ms_scores.append(mean_square)

    true_labels.append(y)

# pickle.dump(X_test, open(model_directory + "x_test.p", "wb"))
pickle.dump(ae_scores, open(model_directory + "ae_scores.p", "wb"))
pickle.dump(ae_cutoffs, open(model_directory + "ae_cutoffs.p", "wb"))
pickle.dump(ae_base_scores, open(model_directory + "ae_base_scores.p", "wb"))
pickle.dump(ae_orig_scores, open(model_directory + "ae_orig_scores.p", "wb"))
pickle.dump(ae_base_cutoffs, open(model_directory + "ae_base_cutoffs.p", "wb"))
pickle.dump(ms_scores, open(model_directory + "ms_scores.p", "wb"))
# pickle.dump(rf_scores, open(model_directory + "rf_scores.p", "wb"))
pickle.dump(true_labels, open(model_directory + "true_labels.p", "wb"))
pickle.dump(inliers, open(model_directory + "inliers.p", "wb"))
pickle.dump(outliers, open(model_directory + "outliers.p", "wb"))
# pickle.dump(luminosity, open(model_directory + "luminosity.p", "wb"))