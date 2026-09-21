
# Bicanski 2026 - UCM: universal cognitive maps
# https://doi.org/10.64898/2026.03.07.705326
# correspondence: bicanski@cbs.mpg.de

# Reviewer code - do not share outside of this review.

######



import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import sparse

# first implemented in matlab
def generate_pin(
):
    
    GCs = np.load("GCmaps/GCmaps.npy")
    
    GCdims = GCs.shape
    GCs    = GCs.reshape((GCdims[0] ,GCdims[1] ,GCdims[2]*GCdims[3]), order='F')
    N_DCs  = GCdims[0]   
    N_GCs  = GCdims[2]*GCdims[3] 

    dc_x_plus  = np.zeros(N_DCs) # for 1-hot vectors
    dc_x_minus = np.zeros(N_DCs) 
    dc_y_plus  = np.zeros(N_DCs)
    dc_y_minus = np.zeros(N_DCs)

    DCxp_2_tGCs_WTS = np.zeros((N_DCs,N_DCs,N_GCs,N_DCs))   
    DCxm_2_tGCs_WTS = np.zeros((N_DCs,N_DCs,N_GCs,N_DCs))
    DCyp_2_tGCs_WTS = np.zeros((N_DCs,N_DCs,N_GCs,N_DCs))
    DCym_2_tGCs_WTS = np.zeros((N_DCs,N_DCs,N_GCs,N_DCs))

    # array explanation:
    # dim 1: index of copy for 2D
    # dim 2: location index along relevant dim (of GCs)
    # dim 3: GC pop
    # dim 4: DX or DY extent
    # N_DC locations per axis, means 4 N_DC DCs, and N_DC^2 weights for each DC

# now we essentially create weights for associations between grid cell population vectors
# we assume distance can gate the selection

    # first Dim
    for sGC2ndDind in tqdm(range(N_DCs)):         # repeat what is below for all points along second dim (copies)
        # source GCs 2nd Dim index
        # meaning: as we cycle through 2nd dim, we actually look up all distances along first dim
        # that is what targets are. So, loop along y, for each y coord, look up all targets along x
        # in both directions. if sender larger than target, then DX is negative
        # if sender equal to target, then DX zero, and DX 0 to link back to sender GC 
        for sGCind in range(N_DCs):         # sources

            #print([sGC2ndDind, sGCind]) # left over old Matlab implemention of this from 2023

            for tGCind in range(N_DCs):     # targets

                #sGC_pv = np.squeeze(GCs[sGC2ndDind,sGCind,:])
                tGC_pv = np.squeeze(GCs[sGC2ndDind,tGCind,:])

                DXind = tGCind-sGCind 

                if DXind>=0:  # use 0 distance in the plus array by convention
                    dc_x_plus = dc_x_plus * 0
                    dc_x_plus[DXind] = 1   # 1-hot distance encoding
                    DCxp_2_tGCs_WTS[sGC2ndDind,sGCind,:,:] = DCxp_2_tGCs_WTS[sGC2ndDind,sGCind,:,:] + np.outer(tGC_pv,dc_x_plus);

                if DXind<=0:  # but no harm writing it to the minus array too
                    dc_x_minus = dc_x_minus * 0
                    dc_x_minus[np.absolute(DXind)] = 1   # 1-hot distance encoding, but distance is negative
                    DCxm_2_tGCs_WTS[sGC2ndDind,sGCind,:,:] = DCxm_2_tGCs_WTS[sGC2ndDind,sGCind,:,:] + np.outer(tGC_pv,dc_x_minus)

    # 2nd Dim
    for sGC2ndDind in tqdm(range(N_DCs)):   # repeat what is below for all points along second dim (copies)

        for sGCind in range(N_DCs):         # sources

            #print([sGC2ndDind, sGCind])

            for tGCind in range(N_DCs):     # targets

                #sGC_pv = np.squeeze(GCs[sGCind,sGC2ndDind,:])
                tGC_pv = np.squeeze(GCs[tGCind,sGC2ndDind,:])

                DYind = tGCind-sGCind

                if DYind>=0:
                    dc_y_plus = dc_y_plus * 0
                    dc_y_plus[DYind] = 1
                    DCyp_2_tGCs_WTS[sGCind,sGC2ndDind,:,:] = DCyp_2_tGCs_WTS[sGCind,sGC2ndDind,:,:] + np.outer(tGC_pv,dc_y_plus)

                if DYind<=0:
                    dc_y_minus = dc_y_minus * 0
                    dc_y_minus[np.absolute(DYind)] = 1
                    DCym_2_tGCs_WTS[sGCind,sGC2ndDind,:,:] = DCym_2_tGCs_WTS[sGCind,sGC2ndDind,:,:] + np.outer(tGC_pv,dc_y_minus)

    np.savez("NAVwts/DCad2b_final.npz", 
        DCxp_2_tGCs_WTS=DCxp_2_tGCs_WTS, 
        DCxm_2_tGCs_WTS=DCxm_2_tGCs_WTS, 
        DCyp_2_tGCs_WTS=DCyp_2_tGCs_WTS, 
        DCym_2_tGCs_WTS=DCym_2_tGCs_WTS)


###


def pin(
    DCxpW, DCxmW, DCypW, DCymW, res, vect, xc, yc, GCs
):  
    
    import warnings
    warnings.filterwarnings('ignore', message='divide by zero encountered in matmul')
    warnings.filterwarnings('ignore', message='invalid value encountered in matmul')
    warnings.filterwarnings('ignore', message='overflow encountered in matmul')
    # Note, there is a well-known NumPy quirk, where there can be a 
    # divide by zero waring in matrix multiplication.  
    # This warning may or may not reproduce on your system, see e.g. here
    # https://github.com/numpy/numpy/issues/29820 
    # It can be ignored, it is not a bug in the code but a spurious warning. 
    # Regardless, as a rule of thumb it is best to Restard the python kernel for each run
    
    # yc, xc, are current x and y.
    # convention is 0 diff is + for x and y, index 0 in   
        
    x_diff = vect[0]
    y_diff = vect[1]
        
    yc_yd = int(yc + y_diff)
    xc_xd = int(xc + x_diff)
        
    DsimVec_X = np.zeros(res)
    DsimVec_Y = np.zeros(res)

    # diff is distance IS similarity! Similarity IS distance!
    DsimVec_X[int(np.round(abs(x_diff)))] = 1  # abs faster than np.absolute
    DsimVec_Y[int(np.round(abs(y_diff)))] = 1

    xpWTS = np.squeeze(DCxpW[yc_yd,xc,:,:])
    xmWTS = np.squeeze(DCxmW[yc_yd,xc,:,:])
    ypWTS = np.squeeze(DCypW[yc,xc_xd,:,:])
    ymWTS = np.squeeze(DCymW[yc,xc_xd,:,:])

    # this can be used to check that the above warnings a spurious
    #print(np.any(np.isnan(xmWTS)),  np.any(np.isinf(xmWTS)))
    #print(np.any(np.isnan(DsimVec_X)), np.any(np.isinf(DsimVec_X)))
    #print(xmWTS.dtype, DsimVec_X.dtype)

    if x_diff>=0:  # because of above convention
        tGC_X = xpWTS @ DsimVec_X
    if x_diff<0:
        tGC_X = xmWTS @ DsimVec_X
    if y_diff>=0:
        tGC_Y = ypWTS @ DsimVec_Y
    if y_diff<0:
        tGC_Y = ymWTS @ DsimVec_Y
    
    # Now we always have tGC_Y and tGC_X

    #print('diffs', y_diff, x_diff)

    # this is slow
    #tGC_Y_rep = np.tile(tGC_Y, (res, res, 1))
    #ty = np.sum(GCs-tGC_Y_rep,2)
    # this is fast
    
    #ty = np.sum(GCs - tGC_Y, axis=2)   # tGC_Y shape (N,) broadcasts over (res,res,N)
    
    ty = np.sum(np.abs(GCs - tGC_Y), axis=2)
    
    #m, n = np.where(ty == 0)
    #m, n = np.where(np.abs(ty) < 1e-10) 
    
    # this is slow
    #tGC_X_rep = np.tile(tGC_X, (res, res, 1))
    #tx = np.sum(GCs-tGC_X_rep,2)
    # this is fast
    
    #tx = np.sum(GCs - tGC_X, axis=2)
    
    tx = np.sum(np.abs(GCs - tGC_X), axis=2)
    
    #k, l = np.where(tx == 0)
    #k, l = np.where(np.abs(tx) < 1e-10) 
    
    min_ty = np.min(np.abs(ty))
    if min_ty > 1e-10:
        return np.array([np.nan, np.nan])  # the script that calls this one could check for this
    m, n = np.where(np.abs(ty) == min_ty)

    min_tx = np.min(np.abs(tx))
    if min_tx > 1e-10:
        return np.array([np.nan, np.nan])
    k, l = np.where(np.abs(tx) == min_tx)  
    
    xy_tmp_y = np.array([n[0], m[0]])
    xy_tmp_x = np.array([l[0], k[0]])
    
    xy_avg = np.round(np.mean([xy_tmp_x, xy_tmp_y], axis=0)).astype(int)

    return xy_avg