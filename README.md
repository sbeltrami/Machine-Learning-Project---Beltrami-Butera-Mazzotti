**CNN-based prediction of the SON Dipole Index**

This repository contains the codes for the training, validation and testing of a Convolutional Neural Network (CNN) - based prediction of the September-November (SON) Dipole Index (DI). The architecture is based on the work of Tao (2024), DOI: 10.1088/1748-9326/ad7522

**Data**

Two observational datasets are used as input:
- HadISST Sea Surface Temperature (SST), downloaded from https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html
- NCEP NCAR Sea Level Pressure (SLP), downloaded from https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html

**Repository content**
- **Codes**
  - data_preparation_obs.ipynb: HadISST SST and NCEP NCAR SLP data are prepared for use as CNN inputs.
  - cnn_train.py: training and validation of the CNN for each lead time. The code is parallelized for each lead time due to the large size of the data.
  - cnn_test.ipynb: testing of the CNN for each lead time and evaluation of the performance metrics for training, validation and testing.
  - cnn_test_no_spike.ipynb: same as cnn_test.ipynb but the best models and the no spike models are evaluated separately.
  - results_plots.ipynb: plots the obtained results and compares them with SON observations of the DI.
- **Output figures**
  - acc_son_di_obs_cnn_lead_time.pdf:  plot of the Anomaly Correlation Coefficient (ACC) computed between the observed HadISST SON Dipole Index and the CNN ensemble prediction, as a function of lead time.
  - son_di_hadisst_testing.pdf: plot of the SON Dipole Index (DI) from HadISST observations (solid curve) and CNN ensemble predictions at all lead times (dashed curves) over the testing period.
  - train_val_loss.pdf: plot of the training and validation loss for the top 10 models at lead time of 1 month.

The files should be run in the following order
1) data_preparation_obs.ipynb
2) cnn_train.py
3) cnn_test.ipynb
4) results_plots.ipynb
5) cnn_test_no_spike.ipynb
