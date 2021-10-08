# Battery-state-estimation

Estimation of the Remaining Useful Life (RUL) of Lithium-ion batteries using Deep LSTMs.

## Introduction

This repository provides the implementation of deep LSTMs for RUL estimation. The experiments have been performed on two datasets: the [**NASA Randomized Battery Usage Data Set**](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#batteryrnddischarge) and the [**UNIBO Powertools Dataset**](https://doi.org/10.17632/n6xg5fzsbv.1).

## Paper
If you use this repo, please cite our paper:

*To Charge or To Sell? EV Pack Useful Life Estimation via LSTMs and Autoencoders* [[URL](https://arxiv.org/abs/2110.03585)]

```
@misc{bosello2021charge,
      title={To Charge or To Sell? EV Pack Useful Life Estimation via LSTMs and Autoencoders}, 
      author={Michael Bosello and Carlo Falcomer and Claudio Rossi and Giovanni Pau},
      year={2021},
      eprint={2110.03585},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Source code structure

The package *data_processing* contains the scripts that load the data from the two sets. *unibo_powertools_data.py* loads the data from the UNIBO dataset and compute the derived columns like the SOC one, while *model_data_handler.py* prepare the time series. *nasa_random_data.py* both loads and prepares the data of the NASA Randomized set. *prepare_rul_data.py* is used for both datasets; it calculates the integral of the current to obtain the RUL based on Ah, and it format the time series for the neural network.

The *experiments* directory contains the Jupyter notebooks defining the various experiments and LSTM models used. The *results* directory shows the plots of the results and the measurements like RMSE, MAE, etc.

## Quick start

### 1) Install requirements

#### Python packages

    pip install tensorflow
    pip install pandas
    pip install sklearn
    pip install scipy
    pip install plotly
    pip install jupyterlab

#### Let Plotly work in Jupyterlab

1) [Install node](https://nodejs.org/en/download/package-manager)


2) `jupyter labextension install jupyterlab-plotly`

### 2) Download the datasets

Download the [NASA Randomized Battery Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#batteryrnddischarge) and put its content in the directory `battery-state-estimation/data/nasa-randomized/`

Download the [**UNIBO dataset**](https://doi.org/10.17632/n6xg5fzsbv.1) and put its content in the directory `battery-state-estimation/data/unibo-powertools-dataset/`

### 3) Run one of the notebooks

Run one of the notebooks in the *experiments* directory. You can switch between training a new model or loading an existing one by toggling the value of *IS_TRAINING* at the top of the notebook (just define the model file name in *RESULT_NAME*).

Check out the *results* directory if you want to see the results obtained by us.

If you want to run the notebook on Google Colab, load the repository in your Google Drive and set to True the variable *IS_COLAB* at the top of the notebook. This will allow the notebook to find the datasets and to save the results in your Drive. 
