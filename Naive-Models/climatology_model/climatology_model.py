#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:17:54 2026

@author: xl3138
"""

"""
Climatology baseline for the ELF manuscript (Milandre karst system, hourly data).

Fits the climatology on the first `train_frac` of the time series (the same split used
to train the DNNs) and produces:

  1. climatology_table.csv          -- lookup table (day-of-year x hour-of-day) per station
  2. climatology_forecast_test.csv  -- climatology forecast at every test timestamp
  2b. climatology_metrics.csv       -- NSE / KGE / RMSE per station + cross-station MEAN
  3. (optional) windowed forecasts of shape (n_windows, horizon) per station that align
     with your test_predy files -- see `windowed_forecast()`

Usage
-----
Edit the CONFIG block below, then run:  python climatology_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================== CONFIG ==============================
###### Change datasets and frequency here to match your DNN training setup

DATASET = "milandre_data"        # milandre_data | yamaska_data
FREQ = "4H"                       # H | 4H | D

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / DATASET / f"data_{FREQ}"

CSV = _DATA_DIR / "train_val_data.csv"   # train_val csv; first TRAIN_FRAC is used to fit
TEST = _DATA_DIR / "test_data.csv"       # separate test csv; set to None to use the held-out tail
TRAIN_FRAC = 0.8
OUT = _HERE / "climatology_out" / f"{DATASET}_{FREQ}"

# Input/target window. Set both to also write windowed forecasts that align
# with the DNN test_predy files; leave as None to skip.
SEQ_LEN_X = None
HORIZON = None
SHIFT = 1
# ==================================================================


def calendar_key(index: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year x hour-of-day key. Maps Feb 29 onto Feb 28 to avoid sparse bins."""
    doy = index.dayofyear.to_numpy().copy()
    leap = index.is_leap_year & (index.month > 2)
    doy[leap] -= 1                       # collapse leap years onto 365-day calendar
    return doy * 100 + index.hour.to_numpy()


def fit_climatology(train: pd.DataFrame) -> pd.DataFrame:
    """Mean of each station by calendar key over the training period."""
    keys = calendar_key(train.index)
    clim = train.groupby(keys).mean()
    clim.index.name = "doy_hour"
    return clim


def predict(clim: pd.DataFrame, index: pd.DatetimeIndex,
            fallback: pd.Series) -> pd.DataFrame:
    """Climatology forecast at arbitrary timestamps."""
    keys = calendar_key(index)
    out = clim.reindex(keys)
    out = out.fillna(fallback)           # any unseen key -> training mean
    out.index = index
    return out


def windowed_forecast(clim_series_test: pd.Series, seq_len_x: int, horizon: int,
                      shift: int = 1) -> np.ndarray:
    """
    Slice a per-timestamp climatology forecast into (n_windows, horizon) target windows
    that mirror the DNN sliding-window construction:
      window i covers inputs [i*shift, i*shift+seq_len_x) and
      targets [i*shift+seq_len_x, i*shift+seq_len_x+horizon).
    Adjust `shift` / offsets to match your dataloader exactly.
    """
    v = clim_series_test.to_numpy()
    n = (len(v) - seq_len_x - horizon) // shift + 1
    idx = (np.arange(n)[:, None] * shift) + seq_len_x + np.arange(horizon)[None, :]
    return v[idx]


def nse(o, p):
    return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


def kge(o, p):
    r = np.corrcoef(o, p)[0, 1]
    beta = p.mean() / o.mean()
    gamma = (p.std() / p.mean()) / (o.std() / o.mean())
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


def rmse(o, p):
    return np.sqrt(np.mean((o - p) ** 2))


def evaluate(obs: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Per-station NSE/KGE/RMSE plus the cross-station mean (the reported value)."""
    rows = []
    for col in obs.columns:
        o, pr = obs[col].to_numpy(), pred[col].to_numpy()
        rows.append({"station": col, "NSE": nse(o, pr), "KGE": kge(o, pr), "RMSE": rmse(o, pr)})
    tab = pd.DataFrame(rows).set_index("station")
    tab.loc["MEAN"] = tab.mean()
    return tab


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV, index_col=0, parse_dates=True).sort_index()
    n_train = int(len(df) * TRAIN_FRAC)
    train, held = df.iloc[:n_train], df.iloc[n_train:]
    print(f"train : {train.index[0]} -> {train.index[-1]}  ({len(train)} rows)")
    print(f"held  : {held.index[0]} -> {held.index[-1]}  ({len(held)} rows)")

    clim = fit_climatology(train)
    clim.to_csv(OUT / "climatology_table.csv")
    print(f"climatology table: {clim.shape[0]} calendar bins x {clim.shape[1]} stations")

    if TEST is not None:
        test = pd.read_csv(TEST, index_col=0, parse_dates=True).sort_index()[df.columns]
        print(f"test  : {test.index[0]} -> {test.index[-1]}  ({len(test)} rows)")
    else:
        test = held  # fallback: evaluate on the held-out tail of the train_val file

    fc = predict(clim, test.index, fallback=train.mean())
    fc.to_csv(OUT / "climatology_forecast_test.csv")

    tab = evaluate(test, fc)
    tab.to_csv(OUT / "climatology_metrics.csv")
    print(tab.round(3))

    if SEQ_LEN_X and HORIZON:
        for col in df.columns:
            w = windowed_forecast(fc[col], SEQ_LEN_X, HORIZON, SHIFT)
            np.savetxt(OUT / f"climatology_{col}_L{SEQ_LEN_X}_H{HORIZON}.csv",
                       w, delimiter=",")
        print(f"windowed forecasts written: shape {w.shape} per station")


if __name__ == "__main__":
    main()