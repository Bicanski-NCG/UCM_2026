
# Bicanski 2026 - UCM: universal cognitive maps
# https://doi.org/10.1016/j.cub.2026.08.064
# correspondence: bicanski@cbs.mpg.de


import numpy as np
import pandas as pd


def load_GCs():
    GCs       = np.load("GCmaps/GCmaps.npy")
    GCdims    = GCs.shape
    GCs       = GCs.reshape((GCdims[0] ,GCdims[1], GCdims[2]*GCdims[3]), order='F')
    res       = GCdims[0]
    N_GCs     = GCdims[2]*GCdims[3]
    N_mod     = GCdims[3]
    N_per_mod = GCdims[2]
    return GCs, res, N_GCs, N_mod, N_per_mod


def load_oasis():
    data_scaled = np.load("datasets/oasis/OASIS_data_rescaled.npy")
    data        = pd.read_csv("datasets/oasis/ExpData/OASIS.csv", sep=None, engine="python")
    IDs         = data.iloc[:, 0].to_numpy()
    valence     = data["Valence_mean"].to_numpy()
    arousal     = data["Arousal_mean"].to_numpy()
    mask        = ((arousal > 2.5) & (arousal < 4.75) & (valence > 3) & (valence < 5.5))
    aro_sub     = arousal[mask]
    val_sub     = valence[mask]
    return data, data_scaled, IDs, valence, arousal, aro_sub, val_sub


def load_oasis_mapped(configname):
    filename1 = f"datasets/oasis/{configname}_anchored_stimuli.npy"
    filename2 = f"datasets/oasis/{configname}_PVs.npy"
    filename3 = f"datasets/oasis/{configname}_PVcoordinates.npy"
    filename4 = f"datasets/oasis/{configname}_center.npy"
    Stims     = np.load(filename1)    
    PVs       = np.load(filename2)
    PVs_xy    = np.load(filename3)
    S1ind     = np.load(filename4)  
    return Stims, PVs, PVs_xy, S1ind