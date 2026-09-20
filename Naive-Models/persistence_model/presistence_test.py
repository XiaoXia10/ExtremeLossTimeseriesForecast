"""
Decile comparison: persistence baseline vs. MAE-loss GRU-AE and GWN models,
for Milandre 4-hour, Milandre 1-hour, and Yamaska (12-step forecast, all three).

Uses your existing destandardize_pred pipeline for the DNN predictions, and
computes a matching persistence baseline directly from test_data.csv.
Stratifies error into deciles (0-10, 10-20, ..., 90-100) using the SAME
per-station/per-well method as your Panel 4 quantile code: percentile
thresholds are computed independently within each station's own
distribution, error is computed within that per-station bin, and only the
final per-station errors are averaged across stations.

ADJUST THE CONFIG SECTION BELOW before running — in particular the GRU path
pattern, which I inferred from your GWN example but have not seen directly.
"""

import os
from os.path import join
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# --- Your existing helper functions (unchanged) -------------------- #
# ------------------------------------------------------------------ #
def _get_stats(df_path):
    df = pd.read_csv(join(df_path, "train_val_data.csv"), parse_dates=True, index_col=0)
    std = df.std()
    mean = df.mean()
    return std, mean


def _load_loader_data(output_dir, loader):
    yhat = pd.read_csv(join(output_dir, loader + "_predy.csv"))
    realy = pd.read_csv(join(output_dir, loader + "_realy.csv"))
    return yhat, realy


def _is_already_raw(realy, std):
    col_std = realy.std()
    dist_standardized = (col_std - 1).abs()
    dist_raw = (col_std - std.values).abs()
    return (dist_raw < dist_standardized).mean() > 0.5


def destandardize_pred(df_path, output_dir, time_dir, loader):
    yhat, realy = _load_loader_data(output_dir, loader)
    std, mean = _get_stats(df_path)

    if _is_already_raw(realy, std):
        yhat = yhat.copy()
        realy = realy.copy()
    else:
        yhat = (yhat * std.values) + mean.values
        realy = (realy * std.values) + mean.values

    time = np.load(join(time_dir, loader + "_time.npy"), allow_pickle=True)
    time = pd.to_datetime(time.flatten())

    yhat.index = time
    realy.index = time

    return yhat, realy


# ------------------------------------------------------------------ #
# --- Persistence baseline, computed directly from test_data.csv --- #
# ------------------------------------------------------------------ #
def persistence_predictions(df_path, h, n, shift):
    """
    Builds persistence obs/pred in the SAME (n_windows, n) per-station
    structure as the DNN predy/realy, so the decile stratification code
    below can treat it identically.
    Returns (pred_df, real_df), each shaped (n_windows*n, n_stations),
    matching the flattened layout of your predy/realy CSVs.
    """
    test = pd.read_csv(join(df_path, "test_data.csv"), parse_dates=True, index_col=0).sort_index()
    stations = test.columns

    n_windows = (len(test) - h - n) // shift + 1
    pred_rows = np.zeros((n_windows * n, len(stations)))
    real_rows = np.zeros((n_windows * n, len(stations)))

    for s_idx, col in enumerate(stations):
        v = test[col].to_numpy()
        row = 0
        for i in range(n_windows):
            start = i * shift
            last_val = v[start + h - 1]
            for step in range(n):
                real_rows[row, s_idx] = v[start + h + step]
                pred_rows[row, s_idx] = last_val
                row += 1

    pred_df = pd.DataFrame(pred_rows, columns=stations)
    real_df = pd.DataFrame(real_rows, columns=stations)
    return pred_df, real_df


# ------------------------------------------------------------------ #
# --- Decile stratification, matching your per-station method ------ #
# ------------------------------------------------------------------ #
def decile_errors(realy, yhat):
    """
    realy, yhat: DataFrames, columns = stations, same shape.
    Returns a DataFrame indexed by decile bin label, one column of MAE
    (mean over stations of the per-station MAE within that bin).
    """
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
            (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]

    stations = realy.shape[1]
    out = pd.Series(index=[f"{lo}-{hi}" for lo, hi in bins], dtype=float)

    for lo, hi in bins:
        station_err = []
        for station in range(stations):
            real_station = realy.iloc[:, station]
            yhat_station = yhat.iloc[:, station]
            lo_thr = np.percentile(real_station, lo)
            hi_thr = np.percentile(real_station, hi)
            if hi == 100:
                mask = (real_station >= lo_thr) & (real_station <= hi_thr)
            else:
                mask = (real_station >= lo_thr) & (real_station < hi_thr)
            station_err.append(np.mean(np.abs(real_station[mask] - yhat_station[mask])))
        out[f"{lo}-{hi}"] = np.mean(station_err)

    return out


# ------------------------------------------------------------------ #
# --- CONFIG: adjust paths/params for your setup -------------------- #
# ------------------------------------------------------------------ #
BASE_GWN = r"../trained_data/gwn/"
BASE_GRU = r"../trained_data/gru/"   # ADJUST if different

# (label, dataset folder name, freq code, forecast horizon, h, n, shift)
CONFIGS = [
    ("Milandre-4h",  "milandre_data", "4H", "12", 12, 12, 12),  # ADJUST dataset/freq strings to match your folders
    ("Milandre-1h",  "milandre_data", "H",  "12", 12, 12, 12),
    ("Yamaska",      "yamaska_data",  "D",  "12", 12, 12, 12),
]


def build_paths_gwn(base, dataset, freq, forecast):
    data_path = join(base, dataset, f"data_{freq}")
    time_path = join(data_path, f"GWN_{forecast}")
    output_dir = join(data_path, f"GWN_{forecast}", f"experiment_{forecast}_mae")
    return data_path, time_path, output_dir


def build_paths_gru(base, dataset, freq, forecast):
    # GRU folder structure has no model-name prefix on the forecast subfolder,
    # unlike GWN's "GWN_{forecast}" — it's just the raw forecast number.
    data_path = join(base, dataset, f"data_{freq}")
    time_path = join(data_path, f"GRU{forecast}{forecast}{forecast}")
    output_dir = join(data_path, f"GRU{forecast}{forecast}{forecast}", f"experiment_GRU{forecast}{forecast}{forecast}_mae")
    return data_path, time_path, output_dir


# ------------------------------------------------------------------ #
# --- Main ------------------------------------------------------------ #
# ------------------------------------------------------------------ #
def main():
    all_results = {}

    for label, dataset, freq, forecast, h, n, shift in CONFIGS:
        print(f"\n=== {label} ===")

        # --- GWN, MAE loss ---
        gwn_data_path, time_path, out_dir = build_paths_gwn(BASE_GWN, dataset, freq, forecast)
        yhat_gwn, realy_gwn = destandardize_pred(gwn_data_path, out_dir, time_path, loader="test")
        all_results[f"{label}_GWN_MAE"] = decile_errors(realy_gwn, yhat_gwn)

        # --- GRU, MAE loss ---
        gru_data_path, time_path, out_dir = build_paths_gru(BASE_GRU, dataset, freq, forecast)
        yhat_gru, realy_gru = destandardize_pred(gru_data_path, out_dir, time_path, loader="test")
        all_results[f"{label}_GRU_MAE"] = decile_errors(realy_gru, yhat_gru)

        # --- Persistence --- (built directly from test_data.csv)
        # Uses the GWN data_path, since that structure is confirmed; if your
        # gru/ workspace's train_val_data.csv / test_data.csv differ from the
        # gwn/ workspace's copy, switch this to gru_data_path instead.
        pred_p, real_p = persistence_predictions(gwn_data_path, h, n, shift)
        all_results[f"{label}_Persistence"] = decile_errors(real_p, pred_p)

    result_df = pd.DataFrame(all_results)
    print("\n=== Full decile comparison ===")
    print(result_df.round(4))
    result_df.to_csv("decile_comparison_all.csv")
    print("\nSaved to decile_comparison_all.csv")


if __name__ == "__main__":
    main()
    
"""
Decile comparison: persistence baseline vs. MAE-loss GRU-AE and GWN models,
for Milandre 4-hour, Milandre 1-hour, and Yamaska (12-step forecast, all three).

Uses your existing destandardize_pred pipeline for the DNN predictions, and
computes a matching persistence baseline directly from test_data.csv.
Stratifies error into deciles (0-10, 10-20, ..., 90-100) using the SAME
per-station/per-well method as your Panel 4 quantile code: percentile
thresholds are computed independently within each station's own
distribution, error is computed within that per-station bin, and only the
final per-station errors are averaged across stations.

ADJUST THE CONFIG SECTION BELOW before running — in particular the GRU path
pattern, which I inferred from your GWN example but have not seen directly.
"""

import os
from os.path import join
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# --- Your existing helper functions (unchanged) -------------------- #
# ------------------------------------------------------------------ #
def _get_stats(df_path):
    df = pd.read_csv(join(df_path, "train_val_data.csv"), parse_dates=True, index_col=0)
    std = df.std()
    mean = df.mean()
    return std, mean


def _load_loader_data(output_dir, loader):
    yhat = pd.read_csv(join(output_dir, loader + "_predy.csv"))
    realy = pd.read_csv(join(output_dir, loader + "_realy.csv"))
    return yhat, realy


def _is_already_raw(realy, std):
    col_std = realy.std()
    dist_standardized = (col_std - 1).abs()
    dist_raw = (col_std - std.values).abs()
    return (dist_raw < dist_standardized).mean() > 0.5


def destandardize_pred(df_path, output_dir, time_dir, loader):
    yhat, realy = _load_loader_data(output_dir, loader)
    std, mean = _get_stats(df_path)

    if _is_already_raw(realy, std):
        yhat = yhat.copy()
        realy = realy.copy()
    else:
        yhat = (yhat * std.values) + mean.values
        realy = (realy * std.values) + mean.values

    time = np.load(join(time_dir, loader + "_time.npy"), allow_pickle=True)
    time = pd.to_datetime(time.flatten())

    yhat.index = time
    realy.index = time

    return yhat, realy


# ------------------------------------------------------------------ #
# --- Persistence baseline, computed directly from test_data.csv --- #
# ------------------------------------------------------------------ #
def persistence_predictions(df_path, h, n, shift):
    """
    Builds persistence obs/pred in the SAME (n_windows, n) per-station
    structure as the DNN predy/realy, so the decile stratification code
    below can treat it identically.
    Returns (pred_df, real_df), each shaped (n_windows*n, n_stations),
    matching the flattened layout of your predy/realy CSVs.
    """
    test = pd.read_csv(join(df_path, "test_data.csv"), parse_dates=True, index_col=0).sort_index()
    stations = test.columns

    n_windows = (len(test) - h - n) // shift + 1
    pred_rows = np.zeros((n_windows * n, len(stations)))
    real_rows = np.zeros((n_windows * n, len(stations)))

    for s_idx, col in enumerate(stations):
        v = test[col].to_numpy()
        row = 0
        for i in range(n_windows):
            start = i * shift
            last_val = v[start + h - 1]
            for step in range(n):
                real_rows[row, s_idx] = v[start + h + step]
                pred_rows[row, s_idx] = last_val
                row += 1

    pred_df = pd.DataFrame(pred_rows, columns=stations)
    real_df = pd.DataFrame(real_rows, columns=stations)
    return pred_df, real_df


# ------------------------------------------------------------------ #
# --- Decile stratification, matching your per-station method ------ #
# ------------------------------------------------------------------ #
def decile_errors(realy, yhat):
    """
    realy, yhat: DataFrames, columns = stations, same shape.
    Returns a DataFrame indexed by decile bin label, one column of MAE
    (mean over stations of the per-station MAE within that bin).
    """
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
            (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]

    stations = realy.shape[1]
    out = pd.Series(index=[f"{lo}-{hi}" for lo, hi in bins], dtype=float)

    for lo, hi in bins:
        station_err = []
        for station in range(stations):
            real_station = realy.iloc[:, station]
            yhat_station = yhat.iloc[:, station]
            lo_thr = np.percentile(real_station, lo)
            hi_thr = np.percentile(real_station, hi)
            if hi == 100:
                mask = (real_station >= lo_thr) & (real_station <= hi_thr)
            else:
                mask = (real_station >= lo_thr) & (real_station < hi_thr)
            station_err.append(np.mean(np.abs(real_station[mask] - yhat_station[mask])))
        out[f"{lo}-{hi}"] = np.mean(station_err)

    return out


# ------------------------------------------------------------------ #
# --- Overall metrics (RMSE, NSE, KGE), per-station then averaged --- #
# ------------------------------------------------------------------ #
def overall_metrics(realy, yhat):
    """
    realy, yhat: DataFrames, columns = stations, same shape.
    Computes RMSE, NSE, KGE per station, then averages across stations —
    same per-station-then-average convention used elsewhere in this script.
    Returns a pd.Series with index ['RMSE', 'NSE', 'KGE'].
    """
    stations = realy.shape[1]
    rmse_vals, nse_vals, kge_vals = [], [], []

    for station in range(stations):
        o = realy.iloc[:, station].to_numpy()
        p = yhat.iloc[:, station].to_numpy()

        rmse_vals.append(np.sqrt(np.mean((o - p) ** 2)))
        nse_vals.append(1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))

        r = np.corrcoef(o, p)[0, 1]
        beta = p.mean() / o.mean()
        gamma = (p.std() / p.mean()) / (o.std() / o.mean())
        kge_vals.append(1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))

    return pd.Series({
        "RMSE": np.mean(rmse_vals),
        "NSE": np.mean(nse_vals),
        "KGE": np.mean(kge_vals),
    })


# ------------------------------------------------------------------ #
# --- CONFIG: adjust paths/params for your setup -------------------- #
# ------------------------------------------------------------------ #

#################### PATH YOU DATA HERE ##############################
BASE_GWN = r"../trained_data/gwn/"
BASE_GRU = r"../trained_data/gru/"   

#################### PICK A LOSS FUNCTION ############################

loss = "mae" #["mae", "focal", "extreme", "dense", "pp", "gumbel"] 

# (label, dataset folder name, freq code, forecast horizon, h, n, shift)
CONFIGS = [
    ("Milandre-4h",  "milandre_data", "4H", "12", 12, 12, 12),  # ADJUST dataset/freq strings to match your folders
    ("Milandre-1h",  "milandre_data", "H",  "12", 12, 12, 12),
    ("Yamaska",      "yamaska_data",  "D",  "12", 12, 12, 12),
]


def build_paths_gwn(base, dataset, freq, forecast):
    data_path = join(base, dataset, f"data_{freq}")
    time_path = join(data_path, f"GWN_{forecast}")
    output_dir = join(data_path, f"GWN_{forecast}", f"experiment_{forecast}_{loss}")
    return data_path, time_path, output_dir


def build_paths_gru(base, dataset, freq, forecast):
    # GRU folder structure has no model-name prefix on the forecast subfolder,
    # unlike GWN's "GWN_{forecast}" — it's just the raw forecast number.
    data_path = join(base, dataset, f"data_{freq}")
    time_path = join(data_path, f"GRU{forecast}{forecast}{forecast}")
    output_dir = join(data_path, f"GRU{forecast}{forecast}{forecast}", f"experiment_GRU{forecast}{forecast}{forecast}_{loss}")
    return data_path, time_path, output_dir

# ------------------------------------------------------------------ #
# --- Main ------------------------------------------------------------ #
# ------------------------------------------------------------------ #
def main():
    all_results = {}
    all_overall = {}

    for label, dataset, freq, forecast, h, n, shift in CONFIGS:
        print(f"\n=== {label} ===")

        # --- GWN, MAE loss ---
        gwn_data_path, time_path, out_dir = build_paths_gwn(BASE_GWN, dataset, freq, forecast)
        yhat_gwn, realy_gwn = destandardize_pred(gwn_data_path, out_dir, time_path, loader="test")
        all_results[f"{label}_GWN_{loss}"] = decile_errors(realy_gwn, yhat_gwn)
        all_overall[f"{label}_GWN_{loss}"] = overall_metrics(realy_gwn, yhat_gwn)

        # --- GRU, MAE loss ---
        gru_data_path, time_path, out_dir = build_paths_gru(BASE_GRU, dataset, freq, forecast)
        yhat_gru, realy_gru = destandardize_pred(gru_data_path, out_dir, time_path, loader="test")
        all_results[f"{label}_GRU_{loss}"] = decile_errors(realy_gru, yhat_gru)
        all_overall[f"{label}_GRU_{loss}"] = overall_metrics(realy_gru, yhat_gru)

        # --- Persistence --- (built directly from test_data.csv)
        # Uses the GWN data_path, since that structure is confirmed; if your
        # gru/ workspace's train_val_data.csv / test_data.csv differ from the
        # gwn/ workspace's copy, switch this to gru_data_path instead.
        pred_p, real_p = persistence_predictions(gwn_data_path, h, n, shift)
        all_results[f"{label}_Persistence"] = decile_errors(real_p, pred_p)
        all_overall[f"{label}_Persistence"] = overall_metrics(real_p, pred_p)

    result_df = pd.DataFrame(all_results)
    print("\n=== Full decile comparison ===")
    print(result_df.round(4))
    result_df.to_csv(f"decile_comparison_all_{loss}_.csv")
    print(f"\nSaved to decile_comparison_all_{loss}.csv")

    overall_df = pd.DataFrame(all_overall)
    print("\n=== Overall metrics (RMSE, NSE, KGE) — per-station mean ===")
    print(overall_df.round(4))
    overall_df.to_csv(f"overall_metrics_all_{loss}.csv")
    print(f"\nSaved to overall_metrics_all_{loss}.csv")


if __name__ == "__main__":
    main()