# Battery-state-estimation

Estimation of the Remaining Useful Life (RUL) of Lithium-ion batteries using Autoencoders + LSTMs and Autoencoders + CNNs.

## Introduction

This repository provides the implementation of deep networks for RUL estimation. The experiments have been performed on two datasets: the [**NASA Randomized Battery Usage Data Set**](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and the [**UNIBO Powertools Dataset**](https://doi.org/10.17632/n6xg5fzsbv.1).

## Paper (Energies publication)
If you use this repo, please cite our paper:

*To Charge or to Sell? EV Pack Useful Life Estimation via LSTMs, CNNs, and Autoencoders* [[URL](https://www.mdpi.com/1996-1073/16/6/2837#)]

```
@Article{en16062837,
    AUTHOR = {Bosello, Michael and Falcomer, Carlo and Rossi, Claudio and Pau, Giovanni},
    TITLE = {To Charge or to Sell? EV Pack Useful Life Estimation via LSTMs, CNNs, and Autoencoders},
    JOURNAL = {Energies},
    VOLUME = {16},
    YEAR = {2023},
    NUMBER = {6},
    ARTICLE-NUMBER = {2837},
    URL = {https://www.mdpi.com/1996-1073/16/6/2837},
    ISSN = {1996-1073},
    DOI = {10.3390/en16062837}
}
```

## Source code structure

The package *data_processing* contains the scripts that load the data from the two sets. *unibo_powertools_data.py* loads the data from the UNIBO dataset and compute the derived columns like the SOC one, while *model_data_handler.py* prepare the time series. *nasa_random_data.py* both loads and prepares the data of the NASA Randomized set. *prepare_rul_data.py* is used for both datasets; it calculates the integral of the current to obtain the RUL based on Ah, and it format the time series for the neural network.

The *experiments* directory contains the Jupyter notebooks defining the various experiments and models used. The *results* directory shows the plots of the results and the measurements like RMSE, MAE, etc.

The trained models are available in the GitHub release section.

## Quick start

### 1) Install requirements

#### Python packages

    pip install tensorflow
    pip install pandas sklearn scipy
    pip install plotly
    pip install jupyter notebook ipykernel jupyterlab


### 2) Download the datasets

Download the [NASA Randomized Battery Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and put its content in the directory `battery-state-estimation/data/nasa-randomized/`

Download the [**UNIBO dataset**](https://doi.org/10.17632/n6xg5fzsbv.1) and put its content in the directory `battery-state-estimation/data/unibo-powertools-dataset/`

### 3) Run one of the notebooks

Run one of the notebooks in the *experiments* directory. You can switch between training a new model or loading an existing one by toggling the value of *IS_TRAINING* at the top of the notebook (just define the model file name in *RESULT_NAME*).

Check out the *results* directory if you want to see the results obtained by us (you can find the trained models in the release section of GitHub).

If you want to run the notebook on Google Colab, load the repository in your Google Drive and set to True the variable *IS_COLAB* at the top of the notebook. This will allow the notebook to find the datasets and to save the results in your Drive. 
