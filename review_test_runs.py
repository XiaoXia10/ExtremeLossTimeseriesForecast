#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run-to-run variability check for the GRU auto-encoder (reviewer comment 2).

Retrains one paper configuration N times with different random seeds, using the
paper's hyperparameters (gru/configs/paper_runs.csv), then summarises:

    <data>/review_test_runs/GRU_Yamaska_<i>/     one folder per run: model, test
                                                 predictions, metrics, loss curve,
                                                 run_config.yaml, run_report.txt
    <data>/review_test_runs/metrics_table.csv    metrics of every run + mean, std,
                                                 the averaged prediction, and the paper run
    <data>/review_test_runs/plot_test_runs.png   measured vs the N runs and their mean
    <data>/review_test_runs/plot_metrics.png     spread of RMSE / NSE / KGE over the runs

Default: Yamaska, daily, forecast 6, extreme loss, 10 runs (seeds 1..10).

    python review_test_runs.py
    python review_test_runs.py --n-runs 3          # quicker check
Finished runs are skipped, so the script can be stopped and restarted.
"""

import argparse
import shutil
import subprocess
import sys
from os.path import join, dirname, abspath, isdir, isfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_dir = dirname(abspath(__file__))
GRU_RUN = join(base_dir, "gru", "src", "run.py")
sys.path.insert(0, join(base_dir, "gru", "src"))
from data_scaler import destandardize_pred                    # noqa: E402
from permetrics.regression import RegressionMetric            # noqa: E402

METRICS = ["RMSE", "MAE", "MSE", "MAPE", "NSE", "KGE"]
SHORT = {"yamaska_data": "Yamaska", "milandre_data": "Milandre"}
YLABEL = {"yamaska_data": "Groundwater level (masl)", "milandre_data": "Discharge (l/s)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="yamaska_data", choices=list(SHORT))
    p.add_argument("--freq", default="D", choices=["D", "H", "4H"])
    p.add_argument("--forecast", type=int, default=6)
    p.add_argument("--loss", default="extreme")
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--data-root", default=join(base_dir, "gru"))
    p.add_argument("--epochs", type=int, default=None, help="override the paper epochs (quick checks only)")
    p.add_argument("--out-name", default="review_test_runs", help="output folder name")
    return p.parse_args()


def run_model(argv):
    cmd = [sys.executable, GRU_RUN] + argv
    print("  $ python gru/src/run.py " + " ".join(argv), flush=True)
    return subprocess.call(cmd)


def metrics(ytrue, ypred):
    r = RegressionMetric(ytrue.to_numpy(), ypred.to_numpy()).get_metrics_by_list_names(METRICS)
    return {m: float(np.mean(r[m])) for m in METRICS}            # mean over stations, as in the paper


def main():
    a = parse_args()
    fc, tag = a.forecast, f"GRU{a.forecast}{a.forecast}{a.forecast}"
    data_path = join(a.data_root, a.dataset, f"data_{a.freq}")
    tensor_dir = join(data_path, tag)
    out = join(data_path, a.out_name)
    name = f"GRU_{SHORT[a.dataset]}"
    common = ["--data-root", a.data_root, "--dataset", a.dataset, "--freq", a.freq,
              "--forecast", str(fc), "--loss", a.loss, "--paper-settings", "--loaders", "test"]
    if a.epochs:
        common += ["--epochs", str(a.epochs)]

    print(f"Review test runs: GRU {a.dataset} {a.freq} forecast {fc} {a.loss}, {a.n_runs} runs -> {out}")
    if run_model(common + ["--steps", "prepare"]) != 0:
        sys.exit("prepare step failed")

    seeds = list(range(a.first_seed, a.first_seed + a.n_runs))
    for i, seed in enumerate(seeds, 1):
        run_dir = join(out, f"{name}_{i}")
        if isfile(join(run_dir, "test_predy.csv")):
            print(f"\n[{i}/{a.n_runs}] {name}_{i} already done - skipped")
            continue
        print(f"\n[{i}/{a.n_runs}] {name}_{i}  (seed {seed})", flush=True)
        sfx = f"_seed{seed}"
        if run_model(common + ["--seed", str(seed), "--exp-suffix", sfx,
                               "--steps", "train,predict,metrics"]) != 0:
            sys.exit(f"run {i} failed")
        # move the experiment folder and its metric files into review_test_runs/<name>_<i>
        exp_dir = join(data_path, tag, f"experiment_{tag}_{a.loss}{sfx}")
        if isdir(run_dir):
            shutil.rmtree(run_dir)
        shutil.move(exp_dir, run_dir)
        for ext in ("csv", "json"):
            f = join(tensor_dir, f"GRU_metrics_{fc}_{a.freq}_{a.loss}{sfx}.{ext}")
            if isfile(f):
                shutil.move(f, join(run_dir, f"GRU_metrics_{fc}_{a.freq}_{a.loss}.{ext}"))

    # ---- summary ----------------------------------------------------------------
    preds, rows, realy = [], [], None
    for i in range(1, a.n_runs + 1):
        run_dir = join(out, f"{name}_{i}")
        yhat, ytrue = destandardize_pred(data_path, run_dir, tensor_dir, "test")
        preds.append(yhat)
        realy = ytrue
        rows.append({"run": f"{name}_{i}", "seed": seeds[i - 1], **metrics(ytrue, yhat)})
    table = pd.DataFrame(rows).set_index("run")
    ens = sum(preds) / len(preds)
    summary = pd.DataFrame([table[METRICS].mean(), table[METRICS].std(ddof=1),
                            pd.Series(metrics(realy, ens))],
                           index=["mean of runs", "std of runs", "averaged prediction"])
    paper = pd.read_csv(join(base_dir, "gru", "configs", "paper_runs.csv"))
    paper = paper[(paper.dataset == a.dataset) & (paper.freq == a.freq)
                  & (paper.forecast == fc) & (paper.loss == a.loss)]
    if not paper.empty:
        r = paper.iloc[0]
        summary.loc["paper run"] = {"RMSE": r.saved_rmse, "NSE": r.saved_nse, "KGE": r.saved_kge}
    full = pd.concat([table, summary])
    full.to_csv(join(out, "metrics_table.csv"), float_format="%.6g")
    ens.to_csv(join(out, "test_predy_mean.csv"))

    pd.set_option("display.width", 200)
    print("\n" + "=" * 70 + "\nTest-set metrics (mean over stations)\n" + "=" * 70)
    print(full[["seed", "RMSE", "MAE", "NSE", "KGE"]].to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- plot 1: measured vs every run and their mean -----------------------------
    names = list(pd.read_csv(join(data_path, "train_val_data.csv"), index_col=0, nrows=1).columns)
    n = realy.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=True, squeeze=False)
    for k, ax in enumerate(axes[:, 0]):
        for j, yh in enumerate(preds):
            ax.plot(yh.index, yh.iloc[:, k], color="tab:blue", alpha=0.25, linewidth=1,
                    label=f"{a.n_runs} runs" if j == 0 else None)
        ax.plot(ens.index, ens.iloc[:, k], color="tab:orange", linewidth=2, label="mean of runs")
        ax.plot(realy.index, realy.iloc[:, k], color="black", linewidth=1.5, label="measured")
        ax.set_title(str(names[k]) if k < len(names) else f"station {k}", fontsize=12)
        ax.set_ylabel(YLABEL[a.dataset], fontsize=10)
        ax.legend(fontsize=9, loc="best")
    axes[-1, 0].set_xlabel("Date")
    fig.suptitle(f"GRU - {a.dataset} {a.freq} - forecast {fc} - {a.loss} loss - test set, {a.n_runs} runs",
                 fontsize=16, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(join(out, "plot_test_runs.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- plot 2: spread of the metrics -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m in zip(axes, ["RMSE", "NSE", "KGE"]):
        v = table[m].to_numpy()
        ax.scatter(np.random.default_rng(0).uniform(-0.08, 0.08, len(v)), v, color="tab:blue", zorder=3,
                   label="runs")
        ax.errorbar(0.25, v.mean(), yerr=v.std(ddof=1), fmt="o", color="tab:orange", capsize=6,
                    label="mean ± std")
        if not paper.empty:
            ax.axhline({"RMSE": r.saved_rmse, "NSE": r.saved_nse, "KGE": r.saved_kge}[m],
                       color="black", linestyle="--", label="paper run")
        ax.set_xlim(-0.3, 0.5)
        ax.set_xticks([])
        ax.set_title(m)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"GRU {a.dataset} {a.freq} f{fc} {a.loss}: {a.n_runs} runs")
    fig.tight_layout()
    fig.savefig(join(out, "plot_metrics.png"), dpi=130)
    plt.close(fig)
    print(f"\nwritten: {out}/metrics_table.csv, plot_test_runs.png, plot_metrics.png")


if __name__ == "__main__":
    main()
