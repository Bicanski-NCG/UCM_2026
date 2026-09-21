# Bicanski 2026 - UCM: universal cognitive maps

https://doi.org/10.1016/j.cub.2026.08.064
correspondence: bicanski@cbs.mpg.de

basic setup

To run the model download the oasis dataset first, link to Kurdi et al. in the paper. 
Not hosting the dataset here. 

Code was run and developed with Python. 3.12

To run the code without changes, create the following folder structure:

datasets/oasis/ExpData (this is where the real data goes)
datasets/oasis/
datasets/oasis/data_grids
datasets/oasis/NM
datasets/oasis/NM/Noise1,2,3
datasets/birds/
GCmaps/
NAVwts/

NM folder holds the data for repeated runs with N anchor pairs, run for M iterations each (Figure 3 last panel in the paper). takes a while.

Everything can be run from python notebooks. 

Paper order: 
UCM_oasis
UCM_oasis_noise
Makebirds etc
UCM_birds
UCM_reasoning
UCM_magnitude_detect
