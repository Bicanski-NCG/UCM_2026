
# Bicanski 2026 - UCM: universal cognitive maps
# https://doi.org/10.1016/j.cub.2026.08.064
# correspondence: bicanski@cbs.mpg.de

######



import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.ndimage.interpolation import rotate
from scipy.signal import correlate2d
import math



def generate_grid_cells(
    shift=False, rotate=False, num_modules=6, num_neurons_per_mod=100
):
# generate static grid cell rate maps as standin for real grid cells.
# python adaptation of the same Matlab code published as part of 
# Bicanski and Burgess 2019, Current Biology

    if num_modules == 7:
        frequencies_of_modules = (np.array([0.002, 0.0028, 0.004, 0.0057, 0.0080, 0.0113, 0.016]) * 2 * np.pi)
        scaling_offsets        = np.array([0.0911, 0.128, 0.182, 0.26, 0.365, 0.515, 0.73])
        orientations           = np.array([0, np.pi / 3, np.pi / 4, np.pi / 2, np.pi / 6, 1.2 * np.pi, 1.7 * np.pi])
        print(orientations)
        # orientations = np.linspace(0, 2 * np.pi, num_modules, endpoint=False)
    
    if num_modules == 9:
        frequencies_of_modules = (
            np.array(
                [0.0014, 0.002, 0.0028, 0.004, 0.0055, 0.0077, 0.0108, 0.0151, 0.0211]
            ) * 2 * np.pi
        )
        scaling_offsets = np.array(
            [0.0638, 0.0911, 0.128, 0.182, 0.26, 0.365, 0.515, 0.73, 1.0429]
        )
        # orientations = np.linspace(0, 2 * np.pi, num_modules, endpoint=False)
        orientations = np.array(
            [
                0,
                np.pi / 3,
                np.pi / 4,
                np.pi / 2,
                np.pi / 6,
                1.2 * np.pi,
                1.7 * np.pi,
                1.9 * np.pi,
                np.pi / 5,
            ]
        )
    
    if num_modules == 4:
        frequencies_of_modules = np.array([0.0108, 0.0151, 0.0211, 0.028]) * 2 * np.pi
        scaling_offsets        = np.array([0.515, 0.73, 1.0429, 1.5])
        orientations           = np.linspace(0, 2 * np.pi, num_modules, endpoint=False)
        print(orientations)    
    
    if num_modules == 6:
        frequencies_of_modules = np.array([0.002, 0.0028, 0.004, 0.0057, 0.0080, 0.0113]) * 2 * np.pi
        scaling_offsets        = np.array([0.0911, 0.128, 0.182, 0.26, 0.365, 0.515])
        orientations           = np.array([0, np.pi / 3, np.pi / 4, np.pi / 2, np.pi / 6, 1.2 * np.pi])
        print(orientations)
    
    if rotate:
        orientation_addons = np.random.uniform(0, 0.25, num_modules) * np.pi
        orientations += orientation_addons
    
    neurons_per_module = num_neurons_per_mod  # offsets per module

    resolution = 10
    x_max = 440
    y_max = 440

    x = np.arange(0, x_max, resolution)
    y = np.arange(0, y_max, resolution)

    gc_rate_maps = np.zeros((len(x), len(y), neurons_per_module, num_modules))

    xg, yg = np.meshgrid(x, y)
    
    xy = np.column_stack((xg.ravel(), yg.ravel()))
    b0 = [np.cos(0), np.sin(0)]
    b1 = [np.cos(np.pi / 3), np.sin(np.pi / 3)]
    b2 = [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)]

    if shift:
        addon_offsets = np.random.uniform(-10, 10, num_modules)
    else:
        addon_offsets = np.zeros(num_modules)

    for i in tqdm(range(len(orientations))):
        frequency = frequencies_of_modules[i]
        offset = scaling_offsets[i]
        orientation = orientations[i]
        rotation_matrix = np.array(
            [
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)],
            ]
        )

        x_off_base1 = 0
        y_off_base1 = offset * 1 / frequency

        x_off_base2 = offset * (1 / frequency) * np.cos(np.pi / 6)
        y_off_base2 = offset * (1 / frequency) * np.sin(np.pi / 6)

        off_vec1 = rotation_matrix @ np.array([x_off_base1, y_off_base1]).reshape(-1, 1)
        off_vec2 = rotation_matrix @ np.array([x_off_base2, y_off_base2]).reshape(-1, 1)
        side_rhombus = int(np.sqrt(neurons_per_module))
        
        for w in range(0, side_rhombus):
            for j in range(0, side_rhombus):
                
                off = (j / side_rhombus * off_vec1.T + w / side_rhombus * off_vec2.T).flatten()

                z0 = np.sum((rotation_matrix @ b0).reshape(1, -1) * (frequency * xy + off + addon_offsets[i]),axis=1,)
                z1 = np.sum((rotation_matrix @ b1).reshape(1, -1) * (frequency * xy + off + addon_offsets[i]),axis=1,)
                z2 = np.sum((rotation_matrix @ b2).reshape(1, -1) * (frequency * xy + off + addon_offsets[i]),axis=1,)

                gc_rate_map = np.cos(z0) + np.cos(z1) + np.cos(z2)
                gc_rate_map /= np.max(gc_rate_map)
                gc_rate_map[gc_rate_map < 0] = 0
                gc_rate_map = gc_rate_map.reshape(len(x), len(y))
                gc_rate_maps[:, :, j * int(np.sqrt(neurons_per_module)) + w, i] = (gc_rate_map)

    return gc_rate_maps



def AutoCorr(GCs, N_GCs, res, h=1):

    correlograms = []
    xed = []
    yed = []

    for i in range(N_GCs):

        ratemap = GCs[:,:,i]
        
        ratemap = ratemap - np.mean(np.reshape(ratemap, (1, ratemap.size)))
        
        corr = correlate2d(ratemap, ratemap, mode="full")

        nyc, nxc = corr.shape

        xedges = (np.arange(nxc) - nxc // 2) * h
        yedges = (np.arange(nyc) - nyc // 2) * h

        X, Y = np.meshgrid(xedges, yedges)
        mask = np.sqrt(X**2 + Y**2) > res

        correlograms.append(ma.masked_array(corr, mask=mask))
        xed.append(xedges)
        yed.append(yedges)

    return np.stack(correlograms, axis=2), np.stack(xed, axis=1), np.stack(yed, axis=1)



def Gridness(correlograms, xed, yed, N_GCs):
    
    # this function is modified from 
    # https://github.com/DehongXu/grid-cell-rnn
    # Conformal Isometry of Lie Group Representation in Recurrent Network of Grid Cells
    
    # Calculate gridness score according to:
    
    # Hafting, T. et al., 2005. Microstructure of a spatial map in the
    # entorhinal cortex. Nature, 436(7052), pp.801-806.
    
    # The autocorrelation of the firing rate map is rotated in 3 degree steps. 
    # The resulting gridness score is the difference between a minimum of cross
    # correlations at 60 and 90 degrees, and a maximum of cross correlations at
    # 30, 90 and 150 degrees.

    # The center of the auto correlation map given by cutRmin is removed
    # from the map.

    h = 1
    cnt = 0;
    #radii = [52, 38, 26, 18, 13, 10, 6]  # heuristic, because I am not calculating scales ...
    #radii = [26, 19, 13, 9, 7, 5, 3]
    inner_radii = [26, 22, 13, 9, 8, 6, 4]
    outer_radii = [44, 42, 40, 28, 20, 15, 9]
    
    G = np.zeros([N_GCs,1])  # largest grids have a single peak in the autocorr

    for i in range(N_GCs):
        
        cutRmin = inner_radii[cnt]
        cutRmax = outer_radii[cnt]
        
        if (np.mod(i, 100) == 0) and (i<N_GCs) and (i>0):
            cnt += 1
        
        ac = correlograms[:,:,i]
        ac_xed = xed[:,i]
        ac_yed = yed[:,i]

        # Remove the center point
        X, Y = np.meshgrid(ac_xed, ac_yed)
        #ac_ori = ac.copy()
        ac[np.sqrt(X ** 2 + Y ** 2) < cutRmin] = 0
        ac[np.sqrt(X ** 2 + Y ** 2) > cutRmax] = 0
        #scale, orientation, peaks = compute_scale_orientation(ac)
        
        da = 3   # 3 delta angle
        angles = list(range(0, 180 + da, da))
        crossCorr = []
        # Rotate and compute correlation coefficient
        
        for angle in angles:
            ac_Rot = rotate(ac, angle, reshape=False)
            C = np.corrcoef(np.reshape(ac, (1, ac.size)),
                            np.reshape(ac_Rot, (1, ac_Rot.size)))
            crossCorr.append(C[0, 1])

        max_angles_i = (np.array([30, 90, 150]) / da).astype(np.int32)
        min_angles_i = (np.array([60, 120]) / da).astype(np.int32)

        maxima = np.max(np.array(crossCorr)[max_angles_i])
        minima = np.min(np.array(crossCorr)[min_angles_i])
        
        G[i,0] = minima - maxima

    return G