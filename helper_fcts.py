

# Bicanski 2026 - UCM: universal cognitive maps
# https://doi.org/10.1016/j.cub.2026.08.064
# correspondence: bicanski@cbs.mpg.de


##### helper functions

import numpy as np



##### check range alongs dimensions, find point closest to center
def check_dataset_range(data, dim1, dim2, mapsize):
    
    # data has been scaled to the range of the GC rate maps
    # before this function call
    
    geo_cen_x = 22
    geo_cen_y = 22
    
    data_dist2cen = np.sqrt((data[:,0]-geo_cen_x)**2 + (data[:,1]-geo_cen_y)**2)
    min_dist      = np.min(data_dist2cen)
    min_indices   = np.where(data_dist2cen == min_dist)[0]
    S_cen_ind     = min_indices[0]
    S1coords      = data[S_cen_ind]  # Shape (2,)
    
    dim1range = np.max(dim1)-np.min(dim1)
    dim2range = np.max(dim2)-np.min(dim2)
    drange    = np.max([dim1range,dim2range])

    prop  = 1 #mapsize/drange # left over, prop was supposed to scale maps, but algorithmically scaling the data before is equivalent.
    N_pts = len(dim1)
    
    return dim1range, dim2range, prop, S_cen_ind, S1coords, N_pts



##### self-explanatory
def calc_mean_gridness(G):
    
    G_m1 = np.nanmean(G[0:99])
    G_m2 = np.nanmean(G[100:199])
    G_m3 = np.nanmean(G[200:299])
    G_m4 = np.nanmean(G[300:399])
    G_m5 = np.nanmean(G[400:499])
    G_m6 = np.nanmean(G[500:599])
    
    G_std1 = np.nanstd(G[0:99])
    G_std2 = np.nanstd(G[100:199])
    G_std3 = np.nanstd(G[200:299])
    G_std4 = np.nanstd(G[300:399])
    G_std5 = np.nanstd(G[400:499])
    G_std6 = np.nanstd(G[500:599])
    
    #G_m7 = np.nanmean(G[600:699])

    Gm = [G_m1, G_m2, G_m3, G_m4, G_m5, G_m6]
    Gstd = [G_std1, G_std2, G_std3, G_std4, G_std5, G_std6]

    return Gm, Gstd



##### Gaussian kernel to multipply infered GC PVs by
# to estimate what recorded GCs would look like
def Gauss_kernel(sigma, sigrange, x_cen, y_cen):
    
    if sigrange.size==1:
        sigrange = np.zeros([6,1])+sigma
    
    GCs       = np.load("GCmaps/GCmaps.npy")
    GCdims    = GCs.shape
    N_GCs     = GCdims[2]*GCdims[3]
    N_mod     = GCdims[3]
    N_per_mod = GCdims[2]

    x      = np.arange(GCdims[0]) - x_cen
    y      = np.arange(GCdims[0]) - y_cen    
    xx, yy = np.meshgrid(x, y)

    Gk = {}
    
    for i in range(N_mod): # should this be N_GCs?
        Gk[f'Gk{i+1}'] = np.exp(-(xx**2 + yy**2) / (2 * sigrange[i]**2))
        Gk[f'Gk{i+1}'] = Gk[f'Gk{i+1}'][:, :, None].repeat(N_per_mod, axis=2)

    Gk_stacked = np.concatenate([Gk[f'Gk{j+1}'] for j in range(N_mod)], axis=2)
    Gk_stacked = Gk_stacked / np.max(Gk_stacked)
    
    return Gk_stacked



##### place panels on plots relative to axes
def Panel_Labels(ax, label, dx, dy):
    
    ax.text(
        dx, dy,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontfamily='sans-serif',
        fontweight='bold',
        va='top',
        ha='right'
    )



##### name says it all
def load_dataset_and_GCs(configname,directory):

    filename1 = f"{directory}{configname}_anchored_stimuli.npy"
    Stims = np.load(filename1)
    filename2 = f"{directory}{configname}_PVs.npy"
    PVs = np.load(filename2)
    filename3 = f"{directory}{configname}_PVcoordinates.npy"
    PVs_xy = np.load(filename3)       
    filename33 = f"{directory}{configname}_PVcoordinates_data.npy"
    PVs_xy_data = np.load(filename33)    
        
    GCs       = np.load("GCmaps/GCmaps.npy")
    GCdims    = GCs.shape
    GCs       = GCs.reshape((GCdims[0] ,GCdims[1], GCdims[2]*GCdims[3]), order='F')
    res       = GCdims[0]
    N_GCs     = GCdims[2]*GCdims[3]
    N_mod     = GCdims[3]
    N_per_mod = GCdims[2]

    return Stims, PVs, PVs_xy, PVs_xy_data, GCs, res, N_GCs, N_mod, N_per_mod




# MN means there were M iterations, and N anchor pairs
def load_dataset_MN(configname, directory, M, N):
    
    filename1  = f"{directory}{configname}_N{N}_M{M}_anchored_stimuli.npy"
    filename2  = f"{directory}{configname}_N{N}_M{M}_PVs.npy"
    filename3  = f"{directory}{configname}_N{N}_M{M}_PVcoordinates.npy"
    filename33 = f"{directory}{configname}_N{N}_M{M}_PVcoordinates_data.npy"
    
    Stims       = np.load(filename1)
    PVs         = np.load(filename2)
    PVs_xy      = np.load(filename3)
    PVs_xy_data = np.load(filename33)
    
    return Stims, PVs, PVs_xy, PVs_xy_data




##### if we just need to load grid cells
def load_GCs():

    GCs    = np.load("GCmaps/GCmaps.npy")

    GCdims    = GCs.shape
    GCs       = GCs.reshape((GCdims[0] ,GCdims[1], GCdims[2]*GCdims[3]), order='F')
    res       = GCdims[0]
    N_GCs     = GCdims[2]*GCdims[3]
    N_mod     = GCdims[3]
    N_per_mod = GCdims[2]

    return GCs, res, N_GCs, N_mod, N_per_mod



##### we get the inferred GC PV via two routes. 
# here, check if they link to the same coordinates
def check_overlap(xy_tmp1,xy_tmp2,tolerance):

    if (abs(xy_tmp1[0]-xy_tmp2[0]) <= tolerance) and (abs(xy_tmp1[1]-xy_tmp2[1]) <= tolerance):
        coords = [round(np.mean([xy_tmp1[0], xy_tmp2[0]])), round(np.mean([xy_tmp1[1], xy_tmp2[1]]))]
    else:
        coords = xy_tmp1 
        
    return coords



def mean_overlap(PVs_xy_tmp1, PVs_xy_tmp2, tolerance, G):

    # Filter tmp1
    mask1 = ~((PVs_xy_tmp1[:,0] == -1) & (PVs_xy_tmp1[:,1] == -1))
    mask2 = ~((PVs_xy_tmp2[:,0] == -1) & (PVs_xy_tmp2[:,1] == -1))

    stack = np.vstack((PVs_xy_tmp1[mask1], PVs_xy_tmp2[mask2]))
    avg_all = np.mean(stack, axis=0)
    final_coords = np.rint(avg_all).astype(int)

    # kind of confidence: how many points cluster near the mean
    mean_dist = np.sqrt(((stack - avg_all)**2).sum(axis=1))
    n_near_mean = np.sum(mean_dist <= tolerance)
    near_mean = int(n_near_mean > 1) 
        
    avgPVs = 1 # dummy
    
    return final_coords, avgPVs, near_mean




##### anchors are stimuli already placed on the UCM
# then find a target (stimulus not yet anchored)
def find_anchors_and_target(Stims, eucl_data_dist, tolerance, v_dist, a_dist,PVs_xy):

    valid_anchor_mask = ((Stims[:,0] > 0) & (PVs_xy[:,0] >= 0) & (PVs_xy[:,1] >= 0))    
    candidate_anchors = np.where(valid_anchor_mask)[0]
            
    if len(candidate_anchors) < 2:
        return None, None, np.nan, np.nan, np.nan, np.nan, np.nan
        
    # pick two of the candidates
    tmp = np.random.choice(len(candidate_anchors), size=2, replace=False) 
    
    # use their indices
    A1ind = candidate_anchors[tmp[0]]
    A2ind = candidate_anchors[tmp[1]]

    # to check constraints, need a target
    candidate_T_mask = (Stims[:, 0] == 0)   # not anchored targets
    valid_T_A1_mask  = (eucl_data_dist[A1ind,:] > 2*tolerance)
    valid_T_A2_mask  = (eucl_data_dist[A2ind,:] > 2*tolerance)
    combined_mask    = (candidate_T_mask & valid_T_A1_mask & valid_T_A2_mask)
    valid_targets    = np.where(combined_mask)[0]

    if valid_targets.size == 0:
        return None, None, np.nan, np.nan, np.nan, np.nan, np.nan

    # there should be at least one remaining target
    if valid_targets.size > 0:
        STind     = np.random.choice(valid_targets)
        vect_A1_T = np.array([v_dist[A1ind,STind], a_dist[A1ind,STind]])
        dist_A1_T = eucl_data_dist[A1ind,STind]
        vect_A2_T = np.array([v_dist[A2ind,STind], a_dist[A2ind,STind]])
        dist_A2_T = eucl_data_dist[A2ind,STind]

    return A1ind, A2ind, STind, vect_A1_T, dist_A1_T, vect_A2_T, dist_A2_T




##### found a traget, select new anchors. This can be done
# since target is the stim, not the GC PV
def find_anchors_for_given_target_v2(Stims, eucl_data_dist, tolerance, v_dist, a_dist, STind, PVs_xy):

    # if we have a target (in stimulus space) and want it to be reached 
    # multiple times from different anchors, when we have a target,
    # need to find anchors that satisfy constraints

    # there are always enough anchors far enough from the target. at a very minimum 2,
    # the ones we picked in the last call of the above find_anchors_and_target function.
    
    T_tmp          = np.zeros(Stims.shape)
    T_tmp[STind,0] = 1  # mark current target to exclude as anchor
    Tmask          = T_tmp[:,0] == 1
    Amask          = Stims[:,0] > 0 # Stim has been placed

    candidate_A_mask = Amask & ~Tmask 
    # this excludes the target from being an anchor,
    
    # now only need to check constraints on anchors relative to maintained target
    # we know there are always enough, at the very least 2, since we already placed
    # the target stim at least once if we enter this function
    
    valid_anchor_mask = ((Stims[:,0] > 0) & (PVs_xy[:,0] >= 0) & (PVs_xy[:,1] >= 0))
    
    valid_A1_mask    = (eucl_data_dist[:,STind] > 2*tolerance) # dist to target, all pts
    combined_mask    = (candidate_A_mask & valid_A1_mask & valid_anchor_mask) # dist for potential A
    valid_anchors    = np.where(combined_mask)[0]

    if valid_anchors.size < 2:
        return None, None, STind, np.nan, np.nan, np.nan, np.nan

    tmp = np.random.choice(valid_anchors, size=2, replace=False) 
    A1ind = tmp[0]
    A2ind = tmp[1]
    
    vect_A1_T = np.array([v_dist[A1ind,STind], a_dist[A1ind,STind]])
    dist_A1_T = eucl_data_dist[A1ind,STind]
    
    vect_A2_T = np.array([v_dist[A2ind,STind], a_dist[A2ind,STind]])
    dist_A2_T = eucl_data_dist[A2ind,STind]
    
    # nothing else needed, because anchors can always be found        
    
    return A1ind, A2ind, STind, vect_A1_T, dist_A1_T, vect_A2_T, dist_A2_T




def valid_anchor_indices(Stims, PVs_xy):
    return np.where(
        (Stims[:,0] > 0) &
        (PVs_xy[:,0] >= 0) &
        (PVs_xy[:,1] >= 0))[0]




def check_output_bounds(xy, GCs):
    return (
        xy[0] < 0 or xy[1] < 0 or 
        xy[0] >= GCs.shape[1] or 
        xy[1] >= GCs.shape[0])
    
        
        


def clip_vect_to_map(xc, yc, vect, res):
    total_x = xc + vect[0]
    total_y = yc + vect[1]
    total_x_clipped = np.clip(total_x, 1, res-1)
    total_y_clipped = np.clip(total_y, 1, res-1)
    return np.array([total_x_clipped - xc, total_y_clipped - yc])



def get_index_in_interval(coords, dx, dy):

    # used for reasoning sims

    x_min, x_max = dx
    y_min, y_max = dy
    
    mask = (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) & \
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
        
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0:
        return None
    
    return np.random.choice(valid_indices)
