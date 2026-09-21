
# Bicanski 2026 - UCM: universal cognitive maps
# https://doi.org/10.1016/j.cub.2026.08.064
# correspondence: bicanski@cbs.mpg.de

######



# generates distance cell model from Bicanski & Burgess 2019, Current Biology, 
# original model from Bush et al. 2015, Neuron.

import numpy as np
import matplotlib.pyplot as plt

def generate_vna():
    
    GCs = np.load("GCmaps/GCmaps.npy")
    
    GCdims = GCs.shape
    GCs    = GCs.reshape((GCdims[0] ,GCdims[1] ,GCdims[2]*GCdims[3]), order='F')
    N_DCs  = GCdims[0]   
    N_GCs  = GCdims[2]*GCdims[3] 

    # same as above for all positions
    AllPosGC2Dy_wts = np.zeros([N_DCs,N_GCs])
    AllPosGC2Dx_wts = np.zeros([N_DCs,N_GCs])
    
    for i in range(N_DCs):

        AllPosGC2Dy_wts[i,:] = np.sum(GCs[i,:,:], axis=0)     
        AllPosGC2Dx_wts[i,:] = np.sum(GCs[:,i,:], axis=0)
        
    LINwts = np.linspace(0,N_DCs-1,N_DCs)   # template for weigths to readout cells

    yDcurr2yUPrdout_wts = np.flip(LINwts)   # weights to readout cells
    yDgoal2yUPrdout_wts = LINwts
    yDcurr2yDOrdout_wts = LINwts
    yDgoal2yDOrdout_wts = np.flip(LINwts)

    xDcurr2yUPrdout_wts = np.flip(LINwts)   # for x, "UP" means going right
    xDgoal2yUPrdout_wts = LINwts
    xDcurr2yDOrdout_wts = LINwts
    xDgoal2yDOrdout_wts = np.flip(LINwts)

    np.savez("NAVwts/DCab2d.npz", 
        AllPosGC2Dx_wts=AllPosGC2Dx_wts, 
        AllPosGC2Dy_wts=AllPosGC2Dy_wts, 
        yDcurr2yUPrdout_wts=yDcurr2yUPrdout_wts, 
        yDgoal2yUPrdout_wts=yDgoal2yUPrdout_wts, 
        yDcurr2yDOrdout_wts=yDcurr2yDOrdout_wts, 
        yDgoal2yDOrdout_wts=yDgoal2yDOrdout_wts, 
        xDcurr2yUPrdout_wts=xDcurr2yUPrdout_wts,
        xDgoal2yUPrdout_wts=xDgoal2yUPrdout_wts, 
        xDcurr2yDOrdout_wts=xDcurr2yDOrdout_wts, 
        xDgoal2yDOrdout_wts=xDgoal2yDOrdout_wts)       


#####


import numpy as np

# calc_vectors is an old version, works, but can have small pixel-scale errors 
# at longer distances, because of cross-axis coupling, but could be resolved
# by regenerating grid cells with different orientation spread

def calc_vectors(cGCs,tGCs,wts):

    # s1 to target

    Dy_curr_rates = wts["AllPosGC2Dy_wts"] @ cGCs   # look up distance rates at target and current location
    Dy_goal_rates = wts["AllPosGC2Dy_wts"] @ tGCs
    Dx_curr_rates = wts["AllPosGC2Dx_wts"] @ cGCs
    Dx_goal_rates = wts["AllPosGC2Dx_wts"] @ tGCs

    Dy_curr_rates[Dy_curr_rates<np.max(Dy_curr_rates)] = 0   # simple winner take all
    Dy_goal_rates[Dy_goal_rates<np.max(Dy_goal_rates)] = 0
    Dx_curr_rates[Dx_curr_rates<np.max(Dx_curr_rates)] = 0
    Dx_goal_rates[Dx_goal_rates<np.max(Dx_goal_rates)] = 0

# Winner-take-all
    Dy_curr_rates_wta = np.zeros_like(Dy_curr_rates)
    Dy_curr_rates_wta[np.argmax(Dy_curr_rates)] = 1
    Dy_goal_rates_wta = np.zeros_like(Dy_goal_rates)
    Dy_goal_rates_wta[np.argmax(Dy_goal_rates)] = 1
    Dx_curr_rates_wta = np.zeros_like(Dx_curr_rates)
    Dx_curr_rates_wta[np.argmax(Dx_curr_rates)] = 1
    Dx_goal_rates_wta = np.zeros_like(Dx_goal_rates)
    Dx_goal_rates_wta[np.argmax(Dx_goal_rates)] = 1
    
    Dy_curr_rates = Dy_curr_rates_wta
    Dy_goal_rates = Dy_goal_rates_wta
    Dx_curr_rates = Dx_curr_rates_wta
    Dx_goal_rates = Dx_goal_rates_wta

    yUPrdout_rate = wts["yDcurr2yUPrdout_wts"] @ Dy_curr_rates + wts["yDgoal2yUPrdout_wts"] @ Dy_goal_rates    # readout
    yDOrdout_rate = wts["yDcurr2yDOrdout_wts"] @ Dy_curr_rates + wts["yDgoal2yDOrdout_wts"] @ Dy_goal_rates
    xUPrdout_rate = wts["xDcurr2yUPrdout_wts"] @ Dx_curr_rates + wts["xDgoal2yUPrdout_wts"] @ Dx_goal_rates
    xDOrdout_rate = wts["xDcurr2yDOrdout_wts"] @ Dx_curr_rates + wts["xDgoal2yDOrdout_wts"] @ Dx_goal_rates

    print(f"Dx_curr winner: {np.argmax(Dx_curr_rates)}, Dx_goal winner: {np.argmax(Dx_goal_rates)}")
    print(f"Dy_curr winner: {np.argmax(Dy_curr_rates)}, Dy_goal winner: {np.argmax(Dy_goal_rates)}")

    x_diff = (xUPrdout_rate-xDOrdout_rate)/2   # map readout rate to position on plane
    y_diff = (yUPrdout_rate-yDOrdout_rate)/2

    return x_diff, y_diff


#####


# sub-pixel version
def calc_vectors_subpix(cGCs, tGCs, wts):
    
    Dy_curr_rates = wts["AllPosGC2Dy_wts"] @ cGCs
    Dy_goal_rates = wts["AllPosGC2Dy_wts"] @ tGCs
    Dx_curr_rates = wts["AllPosGC2Dx_wts"] @ cGCs
    Dx_goal_rates = wts["AllPosGC2Dx_wts"] @ tGCs

    # used to do winner take all here, see above, but analytical grid cell peaks
    # don't necessarily align with discrete index values, see grid_cell.py
    # Instead of winner-take-all we can find peaks with sub-pixel precision
    
    def find_peak_interpolated(rates):
        max_ind = np.argmax(rates)
        if max_ind > 0 and max_ind < len(rates) - 1:
            y1 = rates[max_ind-1] # before max
            y2 = rates[max_ind]   # at max
            y3 = rates[max_ind+1] # after max
            offset = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3) # quadratic interpolation around the peak
            return max_ind + offset
        return float(max_ind)
    
    dy_curr_ind = find_peak_interpolated(Dy_curr_rates)
    dy_goal_ind = find_peak_interpolated(Dy_goal_rates)
    dx_curr_ind = find_peak_interpolated(Dx_curr_rates)
    dx_goal_ind = find_peak_interpolated(Dx_goal_rates)

    # sub-pixel displacement with interpolated indices
    x_diff = dx_goal_ind - dx_curr_ind
    y_diff = dy_goal_ind - dy_curr_ind
    
    # though we loose the readout weights here. But we can always
    # choose to accept some small 1-pixel errors due to numerical 
    # imprecision of the model. The key mechanisms are not affected
    
    return x_diff, y_diff



# simplest version, a lookup table. See Bush et al. 2015, Neuron, for several more neural implementations of VNAs
# any of those or any as of yet unknown vna architecture for that matter could be substituted. It is modular

def calc_vectors_2d(cGCs, tGCs, GCs):

    # Decode current position via 2D correlation
    corr_curr = np.tensordot(GCs, cGCs, axes=(2, 0))  # (res, res)
    cy, cx = np.unravel_index(np.argmax(corr_curr), corr_curr.shape)
    
    # Decode goal position
    corr_goal = np.tensordot(GCs, tGCs, axes=(2, 0))
    gy, gx = np.unravel_index(np.argmax(corr_goal), corr_goal.shape)
        
    return gx - cx, gy - cy