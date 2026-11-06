# Forecasting Extreme Events in Time Series: Introducing the Extreme Loss Function for Neural Networks

Xiao Xia Liang, Dany Lauzon, Julien Straubhaar, Erwan Gloaguen, Maxime Claprood, Philippe Renard, and Reed Maxwell

This repository contains the code for the **extreme loss function**, a loss function that helps deep neural networks (DNNs) forecast extreme values in regression problems. We test it on two hydrogeological forecasting problems:

- **Milandre** (Switzerland): discharge (l/s) of 4 stations within the Milandre catchment, at hourly (`H`) and 4-hourly (`4H`) resolution.
- **Yamaska** (Québec, Canada): groundwater levels (masl) in 18 monitoring wells, at daily (`D`) resolution.

Two models are used, and each is trained with six loss functions (the extreme loss and five others for comparison):

| Model | Framework | Folder |
|---|---|---|
| [GRU auto-encoder](https://arxiv.org/pdf/1406.1078) | TensorFlow / Keras | [gru/](gru/) |
| [Graph WaveNet](https://arxiv.org/pdf/1906.00121) ([original code](https://github.com/sshleifer/Graph-WaveNet)), adapted for hydrogeological data | PyTorch | [gwn/](gwn/) |

You don't need to know Python to run the experiments. You pick a model, a dataset, a forecast length and the loss function(s) in a short text file (YAML), and run one command. The code fills in the hyperparameters used in the paper.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Installation](#2-installation)
3. [Data](#3-data)
4. [Running experiments with a YAML file (recommended)](#4-running-experiments-with-a-yaml-file-recommended)
5. [Running one model directly with `run.py`](#5-running-one-model-directly-with-runpy)
6. [Outputs](#6-outputs)
7. [Reproducing the paper](#7-reproducing-the-paper)
8. [Loss functions](#8-loss-functions)
9. [Using your own data](#9-using-your-own-data)
10. [Troubleshooting](#10-troubleshooting)
11. [Repository layout](#11-repository-layout)

---

## 1. Quick start

```bash
# 1. get the code and install the packages (see section 2)
git clone https://github.com/XiaoXia10/ExtremeLossTimeseriesForecast.git
cd ExtremeLossTimeseriesForecast
conda create -n extreme_loss python=3.12 -y
conda activate extreme_loss
pip install -r requirements.txt

# 2. download the data and place it as described in section 3

# 3. train and evaluate the GRU on Yamaska, 6-day forecast, extreme loss, paper settings
python run_experiment.py --config configs/yamaska_D.yaml
```

The last command trains the model, predicts the train, validation and test sets, computes the metrics, plots the results, and prints a summary that puts your test-set metrics next to the paper's:

```
   loss status   RMSE    NSE    KGE  paper_RMSE  paper_NSE  paper_KGE
extreme     ok 0.0991 0.8936 0.9015      0.0977     0.8895      0.924
```

Your numbers will differ slightly from the paper's because neural network training is random (see [section 7](#7-reproducing-the-paper)). On a laptop CPU this run takes about 5 minutes.

All commands in this README are run from the **repository root** (the folder that contains `run_experiment.py`).

---

## 2. Installation

You need Python 3.10 or newer. We recommend [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda) to keep the packages of this project separate from the rest of your system.

```bash
conda create -n extreme_loss python=3.12 -y
conda activate extreme_loss
pip install -r requirements.txt
```

[requirements.txt](requirements.txt) lists the exact versions we tested with:

| Package | Version | Used for |
|---|---|---|
| tensorflow / keras | 2.20.0 / 3.12.0 | GRU auto-encoder |
| torch | 2.8.0 | Graph WaveNet |
| numpy, pandas, scipy, scikit-learn | 2.0.1, 2.2.3, 1.15.2, 1.7.1 | data handling |
| matplotlib | 3.10.1 | figures |
| permetrics | 2.0.0 | metrics (RMSE, MAE, NSE, KGE, ...) |
| denseweight | 0.1.2 | DenseLoss |
| fastprogress | 1.0.3 | progress bars (Graph WaveNet) |
| pyyaml | 6.0.2 | reading the YAML configuration files |

**GPU (optional).** Everything runs on a CPU. Graph WaveNet uses an NVIDIA GPU (CUDA) or an Apple-silicon GPU (MPS) if one is available. For an NVIDIA GPU, install the CUDA build of PyTorch first by following <https://pytorch.org/get-started/locally/>, then run `pip install -r requirements.txt`.

To check the installation:

```bash
python -c "import tensorflow, torch, permetrics, yaml; print('ok')"
```

---

## 3. Data

The datasets are available here: **[data download (Dropbox)](https://www.dropbox.com/scl/fo/9z66v6y2qmxzldoqebr5a/AL9RviqcOfNo-lQHMh7QuGs?rlkey=6yxpkicxyuotd7xb3ts84gjig&st=bfdq708q&dl=0)**

Copy the `milandre_data` and `yamaska_data` folders into **both** `gru/` and `gwn/`, so that the repository looks like this:

```
ExtremeLossTimeseriesForecast/
├── gru/
│   ├── milandre_data/
│   │   ├── data_H/     train_val_data.csv  test_data.csv  df.csv
│   │   └── data_4H/    train_val_data.csv  test_data.csv  df.csv
│   └── yamaska_data/
│       └── data_D/     train_val_data.csv  test_data.csv  df.csv  well_names.csv
└── gwn/
    ├── milandre_data/
    │   ├── adj_mat/    adj_mx.npy
    │   ├── data_H/     (same files as above)
    │   └── data_4H/
    └── yamaska_data/
        ├── adj_mat/    adj_mx.npy
        └── data_D/
```

| File | Content |
|---|---|
| `train_val_data.csv` | Training and validation period. The first column is the date/time, then one column per station (spring or well). |
| `test_data.csv` | Test period, with the same columns. |
| `df.csv` | Full record, used only by the plotting-position (`pp`) loss to rank values. |
| `adj_mat/adj_mx.npy` | Adjacency matrix for Graph WaveNet. Graph convolution is switched off in this study (as in the paper), so the matrix is loaded but does not change the results. If it is missing, an identity matrix is written. |

| Dataset | Resolution | Stations | Variable | Train/validation | Test | Training fraction |
|---|---|---|---|---|---|---|
| `milandre_data` | `H`, `4H` | 4 springs | discharge (l/s) | 1990–2014 | 2015 | 0.9 |
| `yamaska_data` | `D` | 18 wells | groundwater level (masl) | 2011–2017 | 2018 | 0.8 |

The training fraction is the share of `train_val_data.csv` used for training. The rest, at the end of the period, is used for validation (early stopping). The paper settings load these fractions automatically.

Gaps in the Yamaska records were filled with multiple-point statistics (MPS) using the DeeSse algorithm of the [geone](https://github.com/randlab/geone) package. An example notebook is [here](https://www.dropbox.com/scl/fi/c0zdpxx0zvjt9sql8sigu/MPS_Yamaska_example.ipynb?rlkey=c2pzw0vq1a04oplqcg7l5vuw3&st=p18xejis&dl=0). geone is **not** needed to run the forecasting code.

---

## 4. Running experiments with a YAML file (recommended)

[run_experiment.py](run_experiment.py) is the single entry point for both models. You describe what to run in a YAML file, and the script:

1. checks that your choices are valid (dataset, resolution, forecast length, loss names, settings);
2. loads the hyperparameters used in the paper for each combination;
3. runs the full pipeline once per loss function;
4. prints the test-set metrics next to the paper's.

```bash
python run_experiment.py --config configs/yamaska_D.yaml            # run
python run_experiment.py --config configs/yamaska_D.yaml --dry-run  # only show what would run
```

Three ready-to-edit templates are provided, one per dataset and resolution:

| File | Dataset | Resolution |
|---|---|---|
| [configs/yamaska_D.yaml](configs/yamaska_D.yaml) | Yamaska groundwater levels | daily |
| [configs/milandre_H.yaml](configs/milandre_H.yaml) | Milandre spring discharge | hourly |
| [configs/milandre_4H.yaml](configs/milandre_4H.yaml) | Milandre spring discharge | 4-hourly |

Copy a template (for example to `configs/my_run.yaml`), edit it in any text editor, and run it. In YAML, everything after `#` is a comment, and a list item starts with `-`.

### 4.1 The settings

```yaml
model: gru                 # gru = GRU auto-encoder | gwn = Graph WaveNet
dataset: yamaska_data      # yamaska_data | milandre_data
freq: D                    # yamaska_data: D | milandre_data: H or 4H
forecast: 6                # forecast length in time steps: 6 | 9 | 12 (as in the paper)

loss:                      # one or more; uncomment the ones you want
  - extreme
  # - mae
  # - focal
  # - dense
  # - gumbel
  # - pp

steps: [prepare, train, predict, metrics, plot]

hyperparameters: paper     # paper | default

overrides:                 # optional: change single settings
  seed: 0
  # epochs: 5
```

| Key | Required | Meaning |
|---|---|---|
| `model` | yes | `gru` or `gwn`. |
| `dataset` | yes | `yamaska_data` or `milandre_data`. |
| `freq` | yes | Time resolution: `D` for Yamaska; `H` or `4H` for Milandre. |
| `forecast` | yes | Number of future time steps forecast. The model also sees this many past steps as input. The paper used 6, 9 and 12. |
| `loss` | yes | One loss or a list of losses. Each one is trained and evaluated separately, with its own paper settings. See [section 8](#8-loss-functions). |
| `steps` | no | Pipeline steps to run, in order (see below). Default: `[prepare, train, predict, metrics]`. |
| `loaders` | no | Which sets the `metrics` and `plot` steps use: any of `train`, `val`, `test`. Default: all three. |
| `hyperparameters` | no | `paper` (default) uses the exact settings of the paper run for this model, dataset, resolution, forecast and loss. `default` uses the built-in defaults of the scripts, which you need for a forecast length not in the paper. |
| `overrides` | no | Change individual settings on top of the above, such as `epochs`, `seed` or `latent_dim`. The allowed names are the options of each model's `run.py` ([section 5.2](#52-all-options)), written with `_` or `-`. |
| `data_root` | no | Folder that holds `<dataset>/data_<freq>/` if your data is not in `gru/` or `gwn/`. |
| `exp_suffix` | no | Text added to the output folder name (e.g. `_seed1`), so a new run does not overwrite an earlier one. |
| `keep_going` | no | `true` = carry on with the next loss if one fails. Default: `false`. |

**Pipeline steps**

| Step | What it does |
|---|---|
| `prepare` | Reads the CSV files, standardises them (z-score with the mean and standard deviation of the train/validation period), and cuts them into input/output windows. |
| `train` | Trains the model and saves the best model (lowest validation loss). Training stops early when the validation loss has not improved for `patience` epochs. |
| `predict` | Predicts the train, validation and test sets with the saved model and converts the predictions back to the original units. |
| `metrics` | Computes RMSE, MAE, MSE, MAPE, NSE and KGE for each station and averages them over the stations, as in the paper. |
| `plot` | Plots measured against predicted values for each station. |

The configuration is checked before anything runs. A typo such as `loss: extrme`, a forecast length not in the paper, or an unknown override stops the script with a message that lists the allowed values.

### 4.2 Examples

**Compare all loss functions** (GRU, Yamaska, 6-day forecast): uncomment every loss under `loss:` and run

```bash
python run_experiment.py --config configs/yamaska_D.yaml
```

The summary at the end shows one line per loss with your test metrics and the paper's.

**Use Graph WaveNet instead of the GRU:** set `model: gwn` and run the same command. Graph WaveNet also accepts `mse` as a loss.

**Quick test of the installation** (a few epochs, takes about a minute; the results are not meaningful):

```yaml
overrides:
  epochs: 5
exp_suffix: _test
```

**Repeat a run with another random seed** without overwriting the first one:

```yaml
overrides:
  seed: 3
exp_suffix: _seed3
```

**Evaluate a model that is already trained** (for example a model you trained earlier) without retraining it:

```yaml
steps: [prepare, predict, metrics, plot]
```

**Another forecast length** (not in the paper, for example 24 steps):

```yaml
forecast: 24
hyperparameters: default
```

---

## 5. Running one model directly with `run.py`

Each model has its own pipeline script: [gru/src/run.py](gru/src/run.py) and [gwn/src/run.py](gwn/src/run.py). `run_experiment.py` calls these scripts. You can also call them directly, which is handy for a single run or inside your own scripts. They take the same settings as command-line options.

### 5.1 Examples

```bash
# GRU, Yamaska, 6-day forecast, extreme loss, paper hyperparameters
python gru/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme --paper-settings

# same, plus the figures
python gru/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme --paper-settings \
    --steps prepare,train,predict,metrics,plot

# Graph WaveNet, Milandre hourly, 12-hour forecast, MAE loss, paper hyperparameters
python gwn/src/run.py --dataset milandre_data --freq H --forecast 12 --loss mae --paper-settings

# paper settings, but only 5 epochs (quick test) in a separate folder
python gru/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme --paper-settings \
    --epochs 5 --exp-suffix _test

# re-score an existing model (no training), test set only
python gru/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme --paper-settings \
    --steps prepare,predict,metrics --loaders test

# list every option with its default value
python gru/src/run.py --help
```

With `--paper-settings`, the hyperparameters of the paper run are loaded from `gru/configs/paper_runs.csv` or `gwn/configs/paper_runs.csv`. Any option you also type on the command line wins over the paper value. In the quick-test example above, everything comes from the paper except `--epochs 5`.

Without `--paper-settings`, every option you don't give takes the default value in the table below. These defaults are **not** the paper's settings for every case. To reproduce the paper, always use `--paper-settings` (or `hyperparameters: paper` in YAML).

### 5.2 All options

**Common to both models**

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | `milandre_data` | `milandre_data` or `yamaska_data` |
| `--freq` | `4H` | `H`, `4H` (Milandre) or `D` (Yamaska) |
| `--forecast` | `12` | forecast length = input length, in time steps |
| `--loss` | `extreme` | `mae`, `extreme`, `focal`, `dense`, `gumbel`, `pp` (GWN also `mse`) |
| `--alpha` | GRU `2.0`, GWN `1.5` | threshold of the extreme loss, in standard deviations from the mean ([section 8](#8-loss-functions)) |
| `--steps` | `prepare,train,predict,metrics` | comma-separated list among `prepare,train,predict,metrics,plot` |
| `--loaders` | `train,val,test` | sets used by `metrics` and `plot` |
| `--paper-settings` | off | load the paper hyperparameters (see above) |
| `--exp-suffix` | (none) | added to the output folder name |
| `--data-root` | `gru/` or `gwn/` | folder that holds `<dataset>/data_<freq>/` |
| `--epochs` | GRU `1000`, GWN `500` | maximum number of epochs |
| `--batch-size` | `16` | |
| `--learning-rate` | `1e-4` | Adam learning rate |
| `--dropout` | GRU `0.0`, GWN `0.7` | |
| `--seed` | `0` | random seed (Python, NumPy, TensorFlow/PyTorch) |
| `--keep-going` | off | run the next steps even if one fails |
| `--force` | off | skip the dataset/resolution check |

**GRU only**

| Option | Default | Meaning |
|---|---|---|
| `--latent-dim` | `120` | number of GRU units |
| `--recurrent-dropout` | `0.7` | dropout on the recurrent connections |
| `--patience` | `10` | early stopping patience (epochs) |
| `--train-percent` | `0.8` | training fraction of `train_val_data.csv` (paper: 0.9 Milandre, 0.8 Yamaska) |

**Graph WaveNet only**

| Option | Default | Meaning |
|---|---|---|
| `--weight-decay` | `1e-5` | Adam weight decay |
| `--lr-decay-rate` | `0.98` | learning-rate decay per epoch |
| `--clip` | `1` | gradient clipping |
| `--nhid` | `= --forecast` | number of internal convolution channels |
| `--es-patience` | `10` | early stopping patience (epochs) |
| `--train-split` | `0.8` | training fraction of `train_val_data.csv` (paper: 0.9 Milandre, 0.8 Yamaska) |
| `--device` | `auto` | `auto` = GPU (CUDA or Apple MPS) if available, otherwise CPU; `cpu` = always CPU |

### 5.3 Paper hyperparameters

[gru/configs/paper_runs.csv](gru/configs/paper_runs.csv) and [gwn/configs/paper_runs.csv](gwn/configs/paper_runs.csv) hold one row for each of the 54 paper runs per model (3 dataset/resolution combinations × forecasts 6, 9, 12 × 6 losses). Each row gives:
- the hyperparameters, read from the settings file (`args.pkl`) saved with the original run;
- the test set RMSE, NSE and KGE of that run (`saved_rmse`, `saved_nse`, `saved_kge`);
- the original run folder (`source_run`).

You can open these files in Excel to look up a configuration.

---

## 6. Outputs

All outputs are written inside the data folder of the run (`<model>/<dataset>/data_<freq>/`).

**GRU** (`gru/<dataset>/data_<freq>/GRU<fff>/`, where `<fff>` is the forecast written three times, e.g. `GRU666` for 6 steps)

```
GRU666/
├── x_*.npy, y_*.npy, *_time.npy           input/output windows (prepare)
├── GRU_metrics_6_D_extreme.csv            test-set metrics (mean over stations)
├── GRU_metrics_6_D_extreme_train.csv      training-set metrics
├── GRU_metrics_6_D_extreme_val.csv        validation-set metrics
├── GRU_metrics_6_D_extreme*.json          metrics for every station
└── experiment_GRU666_extreme/
    ├── GRU_best_model.h5                  trained model
    ├── {train,val,test}_predy.csv         predictions (original units)
    ├── {train,val,test}_realy.csv         matching measured values
    ├── plot_{train,val,test}.png          measured vs predicted, one panel per station
    ├── loss_history.png                   training and validation loss per epoch
    ├── args.pkl, run_config.yaml          every setting used for the run
    └── run_report.txt                     summary: settings, steps passed/failed, paper metrics
```

**Graph WaveNet** (`gwn/<dataset>/data_<freq>/GWN_<f>/`)

```
GWN_6/
├── {train,val,test}.npz, ...              input/output windows (prepare)
└── experiment_6_extreme/
    ├── best_model.pth                     trained model
    ├── GWN_metrics_6_D_extreme.csv        test-set metrics (+ _train, _val, .json per station)
    ├── {train,val,test}_{predy,realy}.csv predictions and measured values
    ├── plot_*.png, loss_history.png
    ├── args.pkl, run_config.yaml
    └── run_report.txt
```

The metric CSV files have one header line and one line of values, in this order: RMSE, MAE, MSE, MAPE, NSE, KGE. The header gives the units: RMSE and MAE are in the units of the data (l/s for Milandre, m for Yamaska), MSE in those units squared, and MAPE, NSE and KGE have no units. Each value is the mean over all stations.

Running the same configuration again **overwrites** its folder. Use `exp_suffix` (YAML) or `--exp-suffix` to keep several runs.

---

## 7. Reproducing the paper

### 7.1 Re-scoring the trained paper models

The 108 trained models behind the paper's tables (54 GRU + 54 Graph WaveNet, forecasts 6, 9 and 12) were re-scored with this code. The input windows were rebuilt from the CSV files, and each saved model was then run through `predict` and `metrics`. All 108 models give **exactly** the metrics reported with the original runs. To re-score a released model yourself:
1. Put its file (`GRU_best_model.h5` or `best_model.pth`) in the matching experiment folder ([section 6](#6-outputs)).
2. Run with `steps: [prepare, predict, metrics]`.

### 7.2 Retraining: run-to-run variability

Neural network training depends on random numbers: the initial weights, the order of the batches and dropout. Retraining with a different seed, a different computer or a different package version therefore gives slightly different numbers. To measure how much, [review_test_runs.py](review_test_runs.py) retrains one paper configuration several times with different seeds:

```bash
python review_test_runs.py                 # GRU, Yamaska, D, forecast 6, extreme loss, 10 runs
python review_test_runs.py --n-runs 3      # quicker check
python review_test_runs.py --help          # other datasets, forecasts, losses
```

It writes to `gru/<dataset>/data_<freq>/review_test_runs/`:
- one folder per run (`GRU_Yamaska_1` ... `GRU_Yamaska_10`), with the model, predictions, metrics and settings;
- `metrics_table.csv`: the metrics of every run, their mean and standard deviation, the metrics of the averaged prediction, and the paper run;
- `plot_test_runs.png`: measured values against all runs and their mean, for each well;
- `plot_metrics.png`: the spread of RMSE, NSE and KGE over the runs, with the paper value.

Finished runs are skipped, so you can stop the script and start it again. Each GRU Yamaska run takes about 5 minutes on a laptop CPU.

Our result for 10 runs (GRU, Yamaska, daily, 6-day forecast, extreme loss, paper settings, seeds 1–10). Values are test-set metrics, averaged over the 18 wells:

| | RMSE (m) | NSE | KGE |
|---|---|---|---|
| 10 runs, mean ± standard deviation | 0.097 ± 0.002 | 0.893 ± 0.003 | 0.914 ± 0.007 |
| 10 runs, range | 0.093 – 0.101 | 0.886 – 0.897 | 0.902 – 0.926 |
| Paper (single run) | 0.098 | 0.889 | 0.924 |

The paper's run lies within the spread of the retrained runs.

### 7.3 Note on earlier versions of this repository

In earlier versions, the scripts' default hyperparameters were not the paper's settings. For example, for the GRU on Yamaska with a 6-day forecast:
- `latent_dim` defaulted to 120 instead of 150;
- the test script had `dropout` and `recurrent_dropout` swapped;
- the test script trained for only a few epochs;
- the training fraction was not always the one used in the paper.

Running those scripts gave clearly worse results than the paper. The current version stores the paper settings for every run ([section 5.3](#53-paper-hyperparameters)) and loads them by default in the YAML workflow.

---

## 8. Loss functions

| Name | Description |
|---|---|
| `extreme` | **Extreme loss (this study).** A weighted mean absolute error. Values more than `alpha` standard deviations from the mean are treated as extremes. Their errors and those of the other values are weighted differently, with weights computed from the spread of the measured data (see the paper). In the paper, `alpha` = 2.0 for the GRU and 1.5 for Graph WaveNet. |
| `mae` | Mean absolute error (baseline). |
| `focal` | Focal-R loss: absolute error multiplied by a factor that grows with the error, so large errors count more. |
| `dense` | DenseLoss: squared error weighted by the rarity of the measured value (inverse density, via `denseweight`), so rare values count more. |
| `gumbel` | Gumbel loss: squared error multiplied by `(1 - exp(-squared error))`, inspired by the Gumbel extreme-value distribution. |
| `pp` | Plotting-position loss: errors are weighted according to how rare the measured value is in the full record `df.csv` (ranked in 1-standard-deviation bins). |

The extreme loss is implemented twice: in TensorFlow in [gru/src/model.py](gru/src/model.py), and in PyTorch in [gwn/src/util.py](gwn/src/util.py). To use it in your own model, copy the function for your framework.

---

## 9. Using your own data

The pipeline can run on other time series that have the same format:
- `train_val_data.csv` and `test_data.csv`;
- the date/time in the first column, then one column per station;
- regular time steps and no missing values.

The dataset names are fixed to `milandre_data` and `yamaska_data`, so the simplest way is:

1. Create a folder, for example `my_data/yamaska_data/data_D/`, with your two CSV files. Use `yamaska_data` for daily data and `milandre_data` with `data_H` or `data_4H` for hourly data.
2. Point the code to it with `data_root: /full/path/to/my_data` (YAML) or `--data-root` (`run.py`).
3. Use `hyperparameters: default`, because the paper settings were tuned for our data. Choose `epochs`, `latent_dim` and the other settings under `overrides`.
4. For the `pp` loss, also provide `df.csv` (the full record). For Graph WaveNet, an identity adjacency matrix is written automatically if `adj_mat/adj_mx.npy` is missing.

The y-axis labels of the figures still say "Discharge (l/s)" or "Groundwater level (masl)". You will need to change the labels for your dataset(s).

---

## 10. Troubleshooting

| Problem | Solution |
|---|---|
| `PyYAML is required` | `pip install pyyaml` (or `pip install -r requirements.txt`). |
| `train_val_data.csv not found` | The data is not in the expected folder. Check the layout in [section 3](#3-data), or set `data_root`. |
| `forecast X was not used in the paper` | Use 6, 9 or 12, or set `hyperparameters: default`. |
| `... cannot be overridden` | The option name is wrong. The message lists the allowed names. |
| `df.csv` missing with `loss: pp` | The `pp` loss needs `df.csv` in the data folder. |
| TensorFlow prints many warnings at start-up | These are normal messages about the CPU or GPU and can be ignored. |
| Results differ slightly from the paper | This is expected when retraining ([section 7.2](#72-retraining-run-to-run-variability)). Check that you used the paper settings: `run_report.txt` in the experiment folder lists every setting and the paper metrics. |
| Results differ a lot from the paper | Make sure `hyperparameters: paper` (or `--paper-settings`) is set and that `epochs` is not overridden. |

---

## 11. Repository layout

```
ExtremeLossTimeseriesForecast/
├── run_experiment.py          single entry point (YAML configuration)
├── review_test_runs.py        run-to-run variability check (several seeds)
├── requirements.txt           Python packages
├── configs/                   YAML templates: yamaska_D, milandre_H, milandre_4H
├── gru/
│   ├── configs/paper_runs.csv hyperparameters and metrics of the 54 GRU paper runs
│   └── src/
│       ├── run.py             GRU pipeline (prepare, train, predict, metrics, plot)
│       ├── generate_data.py   windows from the CSV files
│       ├── model.py           GRU auto-encoder and loss functions (TensorFlow)
│       ├── train.py           training
│       ├── compile_model_pred_data.py   predictions
│       ├── get_metrics.py     metrics
│       ├── plot_results.py    figures
│       ├── data_scaler.py     standardisation
│       └── get_parser.py      default settings
├── gwn/
│   ├── configs/paper_runs.csv hyperparameters and metrics of the 54 Graph WaveNet paper runs
│   └── src/
│       ├── run.py             Graph WaveNet pipeline
│       ├── generate_training_data.py
│       ├── model.py           Graph WaveNet (PyTorch)
│       ├── engine.py, train.py
│       ├── util.py            data loaders, loss functions (PyTorch), metrics
│       ├── compile_model_pred.py, get_metrics.py, plot_results.py, data_scaler.py
│       └── exp_results.py
└── Naive-Models/              climatology and persistence baselines
```

---

## Citation

If you use this code, please cite:

> Liang, X. X., Lauzon, D., Straubhaar, J., Gloaguen, E., Claprood, M., Renard, P., & Maxwell, R. Forecasting Extreme Events in Time Series: Introducing the Extreme Loss Function for Neural Networks. *JGR: Machine Learning and Computation* (under review).
