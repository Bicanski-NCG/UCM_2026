
# Bicanski 2026 - UCM: universal cognitive maps
# https://doi.org/10.1016/j.cub.2026.08.064
# correspondence: bicanski@cbs.mpg.de


import numpy as np
import matplotlib.pyplot as plt
from helper_fcts import Gauss_kernel



def make_data_grids(N_GCs, res, Stims, PVs_xy, PVs, sig, sigr, cleanup=0):

    dataset_GCs = np.zeros([res,res,N_GCs])

    print(sigr)

    test_cnt = 0

    for i in range(len(Stims[:,0])):
                
        if Stims[i,1]>0:  # exclude what was deemed low confidence during mapping
            x      = PVs_xy[i,0]
            y      = PVs_xy[i,1]

            pv = PVs[i,:]  # shape (N_GCs,)
            Gk = Gauss_kernel(sigma=sig, sigrange=sigr, x_cen=x, y_cen=y)
            dataset_GCs += pv * Gk  
        
            test_cnt += 1
                    
        if cleanup:
            dataset_GCs[dataset_GCs < cleanup * np.max(dataset_GCs)] = 0
            
    return dataset_GCs



def plot_data_grids(dataset_GCs, N_plots, cells2show):

    # plot a few cells
    fig, axes = plt.subplots(2, int(N_plots/2), figsize=(2*N_plots, 4*2))
    axes = axes.flatten() 
    
    for i in range(N_plots):
        gc = dataset_GCs[:,:,cells2show[i]]
        ax = axes[i]
        im = ax.imshow(gc, cmap='viridis')
        ax.set_title(f'grid cell {cells2show[i]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    return



def plot_correlograms(correlograms, N_plots, cells2show):

    # plot a few correlograms
    N_plots = 10

    fig, axes = plt.subplots(2, int(N_plots/2), figsize=(2*N_plots, 4*2))
    axes = axes.flatten() 

    for i in range(N_plots):
        ac = correlograms[:,:,cells2show[i]]
        ax = axes[i]
        im = ax.imshow(ac, cmap='magma')
        ax.set_title(f'Autocorr {cells2show[i]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    
    return



def plot_gridness(G):

    if len(G) == 600:
        G_m1 = np.mean(G[0:100])
        G_m2 = np.mean(G[100:200])
        G_m3 = np.mean(G[200:300])
        G_m4 = np.mean(G[300:400])
        G_m5 = np.mean(G[400:500])
        G_m6 = np.mean(G[500:600])        
        G_m_vec = [G_m1, G_m2, G_m3, G_m4, G_m5, G_m6]
        N_mod_vec = [1,2,3,4,5,6]
        
    if len(G) == 700:
        G_m1 = np.mean(G[0:100])
        G_m2 = np.mean(G[100:200])
        G_m3 = np.mean(G[200:300])
        G_m4 = np.mean(G[300:400])
        G_m5 = np.mean(G[400:500])
        G_m6 = np.mean(G[500:600])
        G_m7 = np.mean(G[600:700])
        G_m_vec = [G_m1, G_m2, G_m3, G_m4, G_m5, G_m6, G_m7]
        N_mod_vec = [1,2,3,4,5,6,7]

    plt.plot(G)
    plt.show()

    plt.bar(N_mod_vec,G_m_vec)
    plt.show()
    
    return



def save_all_data_grids(dataset_GCs, subfolder="data_grids"):

    import os

    os.makedirs(subfolder, exist_ok=True)
    
    N_cells = dataset_GCs.shape[2]
    
    print(f"Saving {N_cells} grid cell maps to '{subfolder}'...")
    
    for i in range(N_cells):
        
        fig_s, ax_s = plt.subplots(figsize=(4, 4))
        gc = dataset_GCs[:,:,i]/np.max(dataset_GCs[:,:,i])
        im = ax_s.imshow(gc, cmap='viridis')
        ax_s.set_title(f'grid cell {i}')
        ax_s.set_xlabel('X')
        ax_s.set_ylabel('Y')
        plt.colorbar(im, ax=ax_s, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        fig_s.savefig(os.path.join(subfolder, f"GC_{i:04d}.png"), dpi=150)
        
        plt.close(fig_s)
    
    print("Done.")
    
    return