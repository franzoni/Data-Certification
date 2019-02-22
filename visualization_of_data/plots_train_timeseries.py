import setGPU

import getpass
import h5py
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from tqdm import tnrange, tqdm_notebook
import os
import getpass
# Get permission to access EOS (Insert your NICE password)
os.system("echo %s | kinit" % getpass.getpass())

import json
import numpy as np
import pandas as pd
from pprint import pprint

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Lambda, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.activations import sigmoid, linear, relu
from keras.models import Model, load_model
from keras.regularizers import l1, l2, l1_l2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

plt.rcParams.update({'font.size': 22})

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

legend = [{"name": 'pf_jets', "start": 0, "end": 776},
          {"name": 'cal_jet_mets', "start": 777, "end": 944},
          {"name": 'pho', "start": 945, "end": 1280},
          {"name": 'muons', "start": 1281, "end": 1784},
          {"name": 'pf_jets2', "start": 1785, "end": 1889},
          {"name": 'pf_mets', "start": 1890, "end": 1917},
          {"name": 'nvtx', "start": 1918, "end": 1924},
          {"name": 'cal_jet_mets2', "start": 1925},
          {"name": 'sc', "start": 2037, "end": 2127},
          {"name": 'cc', "start": 2128, "end": 2169},
          {"name": 'pho2', "start": 2170, "end": 2365},
          {"name": 'muons2', "start": 2366, "end": 2491},
          {"name": 'ebs', "start": 2492, "end": 2701},
          {"name": 'hbhef', "start": 2702, "end": 2764},
          {"name": 'presh', "start": 2765, "end": 2806},
          {"name": 'inst_lumi', "start": 2807, "end": 2808}]

# Feature description
feature_names = ['qPFJetPt', 'qPFJetEta', 'qPFJetPhi', 'qPFJet0Pt', 'qPFJet1Pt', 'qPFJet2Pt', 'qPFJet3Pt', 'qPFJet4Pt', 'qPFJet5Pt', 'qPFJet0Eta', 'qPFJet1Eta', 'qPFJet2Eta', 'qPFJet3Eta', 'qPFJet4Eta', 'qPFJet5Eta', 'qPFJet0Phi', 'qPFJet1Phi', 'qPFJet2Phi', 'qPFJet3Phi', 'qPFJet4Phi', 'qPFJet5Phi', 'qPFJet4CHS0Pt', 'qPFJet4CHS1Pt', 'qPFJet4CHS2Pt', 'qPFJet4CHS3Pt', 'qPFJet4CHS4Pt', 'qPFJet4CHS5Pt', 'qPFJet4CHS0Eta', 'qPFJet4CHS1Eta', 'qPFJet4CHS2Eta', 'qPFJet4CHS3Eta', 'qPFJet4CHS4Eta', 'qPFJet4CHS5Eta', 'qPFJet4CHS0Phi', 'qPFJet4CHS1Phi', 'qPFJet4CHS2Phi', 'qPFJet4CHS3Phi', 'qPFJet4CHS4Phi', 'qPFJet4CHS5Phi', 'qPFJet8CHS0Pt', 'qPFJet8CHS1Pt', 'qPFJet8CHS2Pt', 'qPFJet8CHS3Pt', 'qPFJet8CHS4Pt', 'qPFJet8CHS5Pt', 'qPFJet8CHS0Eta', 'qPFJet8CHS1Eta', 'qPFJet8CHS2Eta', 'qPFJet8CHS3Eta', 'qPFJet8CHS4Eta', 'qPFJet8CHS5Eta', 'qPFJet8CHS0Phi', 'qPFJet8CHS1Phi', 'qPFJet8CHS2Phi', 'qPFJet8CHS3Phi', 'qPFJet8CHS4Phi', 'qPFJet8CHS5Phi', 'qPFJetEI0Pt', 'qPFJetEI1Pt', 'qPFJetEI2Pt', 'qPFJetEI3Pt', 'qPFJetEI4Pt', 'qPFJetEI5Pt', 'qPFJetEI0Eta', 'qPFJetEI1Eta', 'qPFJetEI2Eta', 'qPFJetEI3Eta', 'qPFJetEI4Eta', 'qPFJetEI5Eta', 'qPFJetEI0Phi', 'qPFJetEI1Phi', 'qPFJetEI2Phi', 'qPFJetEI3Phi', 'qPFJetEI4Phi', 'qPFJetEI5Phi', 'qPFJet8CHSSD0Pt', 'qPFJet8CHSSD1Pt', 'qPFJet8CHSSD2Pt', 'qPFJet8CHSSD3Pt', 'qPFJet8CHSSD4Pt', 'qPFJet8CHSSD5Pt', 'qPFJet8CHSSD0Eta', 'qPFJet8CHSSD1Eta', 'qPFJet8CHSSD2Eta', 'qPFJet8CHSSD3Eta', 'qPFJet8CHSSD4Eta', 'qPFJet8CHSSD5Eta', 'qPFJet8CHSSD0Phi', 'qPFJet8CHSSD1Phi', 'qPFJet8CHSSD2Phi', 'qPFJet8CHSSD3Phi', 'qPFJet8CHSSD4Phi', 'qPFJet8CHSSD5Phi', 'qPFJetTopCHS0Pt', 'qPFJetTopCHS1Pt', 'qPFJetTopCHS2Pt', 'qPFJetTopCHS3Pt', 'qPFJetTopCHS4Pt', 'qPFJetTopCHS5Pt', 'qPFJetTopCHS0Eta', 'qPFJetTopCHS1Eta', 'qPFJetTopCHS2Eta', 'qPFJetTopCHS3Eta', 'qPFJetTopCHS4Eta', 'qPFJetTopCHS5Eta', 'qPFJetTopCHS0Phi', 'qPFJetTopCHS1Phi', 'qPFJetTopCHS2Phi', 'qPFJetTopCHS3Phi', 'qPFJetTopCHS4Phi', 'qPFJetTopCHS5Phi', 'qCalJet0Pt', 'qCalJet1Pt', 'qCalJet2Pt', 'qCalJet3Pt', 'qCalJet4Pt', 'qCalJet5Pt', 'qCalJet0Eta', 'qCalJet1Eta', 'qCalJet2Eta', 'qCalJet3Eta', 'qCalJet4Eta', 'qCalJet5Eta', 'qCalJet0Phi', 'qCalJet1Phi', 'qCalJet2Phi', 'qCalJet3Phi', 'qCalJet4Phi', 'qCalJet5Phi', 'qCalJet0En', 'qCalJet1En', 'qCalJet2En', 'qCalJet3En', 'qCalJet4En', 'qCalJet5En', 'qPho0Pt', 'qPho1Pt', 'qPho2Pt', 'qPho3Pt', 'qPho4Pt', 'qPho5Pt', 'qPho0Eta', 'qPho1Eta', 'qPho2Eta', 'qPho3Eta', 'qPho4Eta', 'qPho5Eta', 'qPho0Phi', 'qPho1Phi', 'qPho2Phi', 'qPho3Phi', 'qPho4Phi', 'qPho5Phi', 'qPho0En', 'qPho1En', 'qPho2En', 'qPho3En', 'qPho4En', 'qPho5En', 'qgedPho0Pt', 'qgedPho1Pt', 'qgedPho2Pt', 'qgedPho3Pt', 'qgedPho4Pt', 'qgedPho5Pt', 'qgedPho0Eta', 'qgedPho1Eta', 'qgedPho2Eta', 'qgedPho3Eta', 'qgedPho4Eta', 'qgedPho5Eta', 'qgedPho0Phi', 'qgedPho1Phi', 'qgedPho2Phi', 'qgedPho3Phi', 'qgedPho4Phi', 'qgedPho5Phi', 'qgedPho0En', 'qgedPho1En', 'qgedPho2En', 'qgedPho3En', 'qgedPho4En', 'qgedPho5En', 'qMu0Pt', 'qMu1Pt', 'qMu2Pt', 'qMu3Pt', 'qMu4Pt', 'qMu5Pt', 'qMu0Eta', 'qMu1Eta', 'qMu2Eta', 'qMu3Eta', 'qMu4Eta', 'qMu5Eta', 'qMu0Phi', 'qMu1Phi', 'qMu2Phi', 'qMu3Phi', 'qMu4Phi', 'qMu5Phi', 'qMu0En', 'qMu1En', 'qMu2En', 'qMu3En', 'qMu4En', 'qMu5En', 'qMuCosm0Pt', 'qMuCosm1Pt', 'qMuCosm2Pt', 'qMuCosm3Pt', 'qMuCosm4Pt', 'qMuCosm5Pt', 'qMuCosm0Eta', 'qMuCosm1Eta', 'qMuCosm2Eta', 'qMuCosm3Eta', 'qMuCosm4Eta', 'qMuCosm5Eta', 'qMuCosm0Phi', 'qMuCosm1Phi', 'qMuCosm2Phi', 'qMuCosm3Phi', 'qMuCosm4Phi', 'qMuCosm5Phi', 'qMuCosm0En', 'qMuCosm1En', 'qMuCosm2En', 'qMuCosm3En', 'qMuCosm4En', 'qMuCosm5En', 'qMuCosmLeg0Pt', 'qMuCosmLeg1Pt', 'qMuCosmLeg2Pt', 'qMuCosmLeg3Pt', 'qMuCosmLeg4Pt', 'qMuCosmLeg5Pt', 'qMuCosmLeg0Eta', 'qMuCosmLeg1Eta', 'qMuCosmLeg2Eta', 'qMuCosmLeg3Eta', 'qMuCosmLeg4Eta', 'qMuCosmLeg5Eta', 'qMuCosmLeg0Phi', 'qMuCosmLeg1Phi', 'qMuCosmLeg2Phi', 'qMuCosmLeg3Phi', 'qMuCosmLeg4Phi', 'qMuCosmLeg5Phi', 'qMuCosmLeg0En', 'qMuCosmLeg1En', 'qMuCosmLeg2En', 'qMuCosmLeg3En', 'qMuCosmLeg4En', 'qMuCosmLeg5En', 'qPFJet4CHSPt', 'qPFJet4CHSEta', 'qPFJet4CHSPhi', 'qPFJet8CHSPt', 'qPFJet8CHSEta', 'qPFJet8CHSPhi', 'qPFJetEIPt', 'qPFJetEIEta', 'qPFJetEIPhi', 'qPFJet8CHSSDPt', 'qPFJet8CHSSDEta', 'qPFJet8CHSSDPhi', 'qPFJetTopCHSPt', 'qPFJetTopCHSEta', 'qPFJetTopCHSPhi', 'qPFChMetPt', 'qPFChMetPhi', 'qPFMetPt', 'qPFMetPhi', 'qNVtx', 'qCalJetPt', 'qCalJetEta', 'qCalJetPhi', 'qCalJetEn', 'qCalMETPt', 'qCalMETPhi', 'qCalMETEn', 'qCalMETBEPt', 'qCalMETBEPhi', 'qCalMETBEEn', 'qCalMETBEFOPt', 'qCalMETBEFOPhi', 'qCalMETBEFOEn', 'qCalMETMPt', 'qCalMETMPhi', 'qCalMETMEn', 'qSCEn', 'qSCEta', 'qSCPhi', 'qSCEtaWidth', 'qSCPhiWidth', 'qSCEnhfEM', 'qSCEtahfEM', 'qSCPhihfEM', 'qSCEn5x5', 'qSCEta5x5', 'qSCPhi5x5', 'qSCEtaWidth5x5', 'qSCPhiWidth5x5', 'qCCEn', 'qCCEta', 'qCCPhi', 'qCCEn5x5', 'qCCEta5x5', 'qCCPhi5x5', 'qPhoPt', 'qPhoEta', 'qPhoPhi', 'qPhoEn_', 'qPhoe1x5_', 'qPhoe2x5_', 'qPhoe3x3_', 'qPhoe5x5_', 'qPhomaxenxtal_', 'qPhosigmaeta_', 'qPhosigmaIeta_', 'qPhor1x5_', 'qPhor2x5_', 'qPhor9_', 'qgedPhoPt', 'qgedPhoEta', 'qgedPhoPhi', 'qgedPhoEn_', 'qgedPhoe1x5_', 'qgedPhoe2x5_', 'qgedPhoe3x3_', 'qgedPhoe5x5_', 'qgedPhomaxenxtal_', 'qgedPhosigmaeta_', 'qgedPhosigmaIeta_', 'qgedPhor1x5_', 'qgedPhor2x5_', 'qgedPhor9_', 'qMuPt', 'qMuEta', 'qMuPhi', 'qMuEn_', 'qMuCh_', 'qMuChi2_', 'qMuCosmPt', 'qMuCosmEta', 'qMuCosmPhi', 'qMuCosmEn_', 'qMuCosmCh_', 'qMuCosmChi2_', 'qMuCosmLegPt', 'qMuCosmLegEta', 'qMuCosmLegPhi', 'qMuCosmLegEn_', 'qMuCosmLegCh_', 'qMuCosmLegChi2_', 'qSigmaIEta', 'qSigmaIPhi', 'qr9', 'qHadOEm', 'qdrSumPt', 'qdrSumEt', 'qeSCOP', 'qecEn', 'qUNSigmaIEta', 'qUNSigmaIPhi', 'qUNr9', 'qUNHadOEm', 'qUNdrSumPt', 'qUNdrSumEt', 'qUNeSCOP', 'qUNecEn', 'qEBenergy', 'qEBtime', 'qEBchi2', 'qEBiEta', 'qEBiPhi', 'qEEenergy', 'qEEtime', 'qEEchi2', 'qEEix', 'qEEiy', 'qESenergy', 'qEStime', 'qESix', 'qESiy', 'qHBHEenergy', 'qHBHEtime', 'qHBHEauxe', 'qHBHEieta', 'qHBHEiphi', 'qHFenergy', 'qHFtime', 'qHFieta', 'qHFiphi', 'qPreShEn', 'qPreShEta', 'qPreShPhi', 'qPreShYEn', 'qPreShYEta', 'qPreShYPhi', 'inst_luminosity']

data_directory = "/eos/cms/store/user/fsiroky/consistentlumih5/"
label_file = "/afs/cern.ch/user/t/tkrzyzek/Documents/Data-Certification/JetHT.json"
pileup_file = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/pileup_latest.txt"
model_directory = "/eos/user/t/tkrzyzek/autoencoder/time_series_comp/"
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
data.drop([2808, 2809, 2810, 2811, 2812], axis=1, inplace=True)

# Append inst. luminosity at the end as well
data[2807] = data["inst_lumi"]

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

with open(pileup_file) as f:
    pileup = json.load(f)

def pileup_from_json(json_file, orig_runid, orig_lumid):
    try:
        for i in json_file[str(int(orig_runid))]:
            if orig_lumid == i[0]:
                return i[3]
    except KeyError:
        print('key error')
    return 1

def add_pileup(sample):
    return pileup_from_json(pileup, sample["run"], sample["lumi"])

data["pileup"] = data.apply(add_pileup, axis=1)

data[2808] = data["pileup"]

INPUT_DIM = 2809

X = data.iloc[:, 0:INPUT_DIM]
y = data['label']

tss = TimeSeriesSplit(max_train_size=None, n_splits=10)

ae_orig_scores = []
ae_scores = []
ms_scores = []
rf_scores = []
xgb_scores = []

for train_idx, test_idx in tss.split(X):
    X_train = X[train_idx]
    y_train = y[train_idx]
    
    X_test = X[test_idx]
    y_test = y[test_idx]

    normalizer = StandardScaler()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)

    # Train only on good
    X_train = X_train[y_train == 0]

    input_layer = Input(shape=(INPUT_DIM, ))

    x = Dense(2048, kernel_regularizer=l1_l2(10e-5))(input_layer)
    x = PReLU()(x)

    x = Dense(1024, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(512, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(256, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(128, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(64, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(128, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(256, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(512, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(1024, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(2048, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(INPUT_DIM)(x)

    autoencoder = Model(inputs=input_layer, outputs=x)

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
                    epochs=1024,
                    batch_size=256,
                    validation_split=0.25,
                    verbose=2,
                    callbacks=[early_stopper, checkpoint_callback])

    # Reload saved model
    autoencoder = load_model("%s%s.h5" % (model_directory, model_name))

    # Run predictions
    ae_pred = autoencoder.predict(X_test)
    ae_pred_train = autoencoder.predict(X)

    # Exclude validation set
    ae_pred_train = ae_pred_train[:int(0.75 * len(ae_pred_train)), :]

    # Exclude validation set
    X_train_wo_valid = X_train[:int(0.75 * len(X_train)), :]

    def get_error_df(X_test, predictions, mode="allmean", n_highest = 100):

        if mode == "allmean":
            return np.mean(np.power(X_test - predictions, 2), axis=1)

        elif mode == "topn":
            temp = np.partition(-np.power(X_test - predictions, 2), n_highest)
            result = -temp[:,:n_highest]
            return np.mean(result, axis=1)

        elif mode == "topn_median":
            temp = np.partition(-np.power(X_test - predictions, 2), n_highest)
            result = -temp[:,:n_highest]
            return np.median(result, axis=1)

        elif mode == "perobj":
            mses = []
            for l in legend:
                mse = np.mean(
                    np.power(X_test[:,l["start"]:l["end"]] - predictions[:,l["start"]:l["end"]], 2),
                    axis=1)
                mses.append(mse)

            return np.maximum.reduce(mses)

        elif mode == "bottomn":
            temp = np.partition(np.power(X_test - predictions, 2), n_highest)
            result = temp[:,:n_highest]
            return np.mean(result, axis=1)
        
    ae_error = get_error_df(X_test, ae_pred, mode="topn", n_highest=100)
    ae_error_train = get_error_df(X_train_wo_valid, ae_pred_train, mode="topn", n_highest=100)
    
    ae_orig_scores.append(ae_error)

    start_legend = 1414
    var_legend = [{'start': start_legend, 'end': start_legend, 'name': 'mean'},
                  {'start': start_legend+1, 'end': start_legend+1, 'name': 'RMS'},
                  {'start': start_legend+2, 'end': start_legend+2, 'name': 'Q1'},
                  {'start': start_legend+3, 'end': start_legend+3, 'name': 'Q2'},
                  {'start': start_legend+4, 'end': start_legend+4, 'name': 'Q3'},
                  {'start': start_legend+5, 'end': start_legend+5, 'name': 'Q4'},
                  {'start': start_legend+6, 'end': start_legend+6, 'name': 'Q5'}]

    corr = [pearsonr(X_train_wo_valid[:, i].reshape(-1), ae_pred_train[:, i].reshape(-1)) for i in range(INPUT_DIM)]
    corr = np.array(corr)[:, 0]
    corr[np.isnan(corr)] = 0
    corr_filter = corr > 0.5
    variance = np.var(data.iloc[:, 0:INPUT_DIM])
    var_filter = variance != 0
    feature_filter = corr_filter & var_filter
    X_train_filtered = X_train[:, feature_filter]
    X_train_filtered.shape
    X_train_filtered_wo_valid = X_train.iloc[:round(0.75*len(X_train)), :]
    X_train_filtered_wo_valid = np.array(X_train_filtered_wo_valid)[:, feature_filter]

    input_dim = X_train_filtered.shape[1]
    input_layer = Input(shape=(input_dim, ))

    x = Dense(1024, kernel_regularizer=l1_l2(10e-5))(input_layer)
    x = PReLU()(x)

    x = Dense(512, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(256, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(128, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(64, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(32, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(64, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(128, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(256, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(512, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(1024, kernel_regularizer=l1_l2(10e-5))(x)
    x = PReLU()(x)

    x = Dense(input_dim)(x)

    autoencoder = Model(inputs=input_layer, outputs=x)

    autoencoder.summary()

    model_name2 = 'ae2'

    adamm = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    early_stopper = EarlyStopping(monitor="val_loss",
                                  patience=32,
                                  verbose=True,
                                  mode="auto")

    autoencoder.compile(optimizer=adamm, loss='mean_squared_error')

    checkpoint_callback = ModelCheckpoint(("%s%s.h5" % (model_directory, model_name2)),
                                          monitor="val_loss",
                                          verbose=False,
                                          save_best_only=True,
                                          mode="min")

    autoencoder.fit(X_train_filtered,
                    X_train_filtered,
                    epochs=1024,
                    batch_size=256,
                    validation_split=0.25,
                    verbose=2,
                    callbacks=[early_stopper, checkpoint_callback])

    # Reload saved model
    autoencoder = load_model("%s%s.h5" % (model_directory, model_name2))

    X_test_filtered = X_test_norm[:, feature_filter]

    # Exclude validation set
    X_train_wo_valid_filtered = X_train_wo_valid[:, feature_filter]

    # Run predictions
    ae_pred_filtered = autoencoder.predict(X_test_filtered)
    ae_pred_filtered_train = autoencoder.predict(X_train_wo_valid_filtered)

    ae_error_filtered = get_error_df(X_test_filtered, ae_pred_filtered, mode="topn", n_highest=100)
    ae_error_filtered_train = get_error_df(X_train_wo_valid_filtered, ae_pred_filtered_train, mode="topn", n_highest=100)
    
    ae_scores.append(ae_error_filtered)
    
    params = {
    "max_depth": 7,
    "n_estimators": 64, 
    "random_state": 42, 
    "n_jobs": -1,
    "verbose" : 1,
#    "class_weight": cw
    }
    
    #rf = RandomForestClassifier(**params)
    
    #rf.fit(X_train, y_train)
    
pickle.dump(ae_scores, open(model_directory + "ae_scores.p", "wb"))
pickle.dump(ae_orig_scores, open(model_directory + "ae_orig_scores.p", "wb"))
pickle.dump(feature_filter, open(model_directory + "feature_filter.p", "wb"))

