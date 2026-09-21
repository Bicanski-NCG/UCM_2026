
# Bicanski 2026 - UCM: universal cognitive maps
# https://doi.org/10.1016/j.cub.2026.08.064
# correspondence: bicanski@cbs.mpg.de

######


import numpy as np
import matplotlib.pyplot as plt


def prep_oasis(data,padding,res):
    
    valence = data["Valence_mean"].to_numpy()
    arousal = data["Arousal_mean"].to_numpy()

    # select valence range 3-5.5, arousal range 2.5-4.75. similar to Qasim et a. 2025
    # I assume they did this for even coverage of the square 2D space in that range
    
    mask = (
        (arousal > 2.5) & (arousal < 4.75) &
        (valence > 3) & (valence < 5.5)
    )
    #subsets
    aro_sub = arousal[mask]
    val_sub = valence[mask]

    # check distribution
    plt.scatter(valence, arousal, s=30, facecolors='none', edgecolors=(130/255, 130/255, 130/255))
    plt.scatter(val_sub, aro_sub, s=10, facecolors='black', edgecolors='black')
    plt.xlabel("valence", fontsize=12, color='black')
    plt.ylabel("Arousal", fontsize=12, color='black')
    plt.title("Oasis dataset", fontsize=14, fontweight='bold')
    plt.show()

    # rescale distances in data, this is like modulating velocity integration gain in grid cell models
    val  = padding+(val_sub-np.min(val_sub))/(np.max(val_sub)-np.min(val_sub))*(res-2*padding)
    aro  = padding+(aro_sub-np.min(aro_sub))/(np.max(aro_sub)-np.min(aro_sub))*(res-2*padding)
    data = np.transpose(np.array([val,aro])) # this transpose is needed to plot it same as Kunz paper

    return data, val, aro


###


def make_oasis_synthetic(val,aro):
    
    # synthetic data
    rng = np.random.default_rng()
    N_new = 5000
    x_new = rng.choice(val, size=N_new, replace=True)
    y_new = rng.choice(aro, size=N_new, replace=True)
    surrogate = np.column_stack([x_new, y_new])
    data = surrogate
    val = data[:,0]
    aro = data[:,1]
    
    return data, val, aro


###


def get_distances(data,dim1,dim2):
    
    # precompute all posible distances
    # this is just done to speed up simulation. look at figure S1B for what would
    # happen sequentially
    
    eucl_data_dist = np.linalg.norm(data[np.newaxis,:,:] - data[:,np.newaxis,:],axis=2)
    dim1_dist = dim1[:, None] - dim1[None, :]   # signed distances
    dim2_dist = dim2[:, None] - dim2[None, :]

    eucl_data_dist = np.transpose(eucl_data_dist)  # transpose such that indexing anchor vs target fits below
    dim1_dist = np.transpose(dim1_dist)
    dim2_dist = np.transpose(dim2_dist)
    
    return eucl_data_dist, dim1_dist, dim2_dist

