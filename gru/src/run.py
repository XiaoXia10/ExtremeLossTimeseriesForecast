#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single entry point for the GRU auto-encoder + loss-function experiments.

All settings are given once, on the command line, and passed to every step,
so the steps cannot disagree. Each step re-uses the project's own modules:

    prepare   raw CSVs   -> windowed tensors            (generate_data.py)
    train     tensors    -> GRU_best_model.h5           (train.py)
    predict   best model -> {loader}_{realy,predy}.csv  (compile_model_pred_data.py)
    metrics   preds      -> RMSE/MAE/MSE/MAPE/NSE/KGE   (get_metrics.py)
    plot      preds      -> measured-vs-predicted PNG   (plot_results.py)

--------------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------------
Full run (prepare, train, predict, metrics):
    python gru/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme \\
        --latent-dim 150 --batch-size 16 --patience 10

Score a trained model without retraining (e.g. a released paper model placed in
the experiment folder):
    python gru/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme \\
        --latent-dim 150 --steps predict,metrics

Re-train with the exact hyperparameters of the paper run (configs/paper_runs.csv):
    python gru/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme --paper-settings

Only make the plots:
    python gru/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme --steps plot

--------------------------------------------------------------------------------
FOLDER LAYOUT (relative to --data-root, default <repo>/gru)
--------------------------------------------------------------------------------
    <dataset>/data_<freq>/train_val_data.csv      input (required)
    <dataset>/data_<freq>/test_data.csv           input (required)
    <dataset>/data_<freq>/df.csv                  input (only for --loss pp)
    <dataset>/data_<freq>/GRU<f><f><f>/           tensors + metric files
    <dataset>/data_<freq>/GRU<f><f><f>/experiment_GRU<f><f><f>_<loss>/
                                                  model, predictions, plots, run_report.txt
--------------------------------------------------------------------------------
"""

import argparse
import os
import sys
import random
import traceback
from os.path import join, isfile, dirname, abspath

# Repo-relative default paths: <repo>/gru/src/run.py -> base_dir = <repo>/gru
current_dir = dirname(abspath(__file__))
base_dir = dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

# train.py and plot_results.py use matplotlib - stay headless
import matplotlib
matplotlib.use("Agg")


VALID_FREQ = {
    "milandre_data": {"H", "4H"},
    "yamaska_data": {"D"},
}
YLABEL = {
    "milandre_data": "Discharge (l/s)",
    "yamaska_data": "Groundwater level (masl)",
}
METRIC_NAMES = ["RMSE", "MAE", "MSE", "MAPE", "NSE", "KGE"]
ALL_STEPS = ["prepare", "train", "predict", "metrics", "plot"]
DEFAULT_STEPS = "prepare,train,predict,metrics"

# Hyperparameters of every paper run (one row per dataset/freq/forecast/loss),
# taken from the args.pkl saved with each original run.
PAPER_RUNS = join(base_dir, "configs", "paper_runs.csv")
PAPER_HPARAMS = ['alpha', 'epochs', 'batch_size', 'learning_rate', 'latent_dim', 'dropout', 'recurrent_dropout', 'patience', 'train_percent']


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def build_parser():
    p = argparse.ArgumentParser(
        description="Run the GRU auto-encoder pipeline (prepare, train, predict, metrics, plot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("experiment")
    g.add_argument("--data-root", default=base_dir,
                   help="folder containing <dataset>/data_<freq>/ (default: <repo>/gru)")
    g.add_argument("--dataset", default="milandre_data",
                   choices=["milandre_data", "yamaska_data"])
    g.add_argument("--freq", default="4H", choices=["H", "4H", "D"])
    g.add_argument("--forecast", type=int, default=12,
                   help="sequence length = prediction horizon = window shift")
    g.add_argument("--loss", default="extreme",
                   choices=["mae", "extreme", "gumbel", "dense", "pp", "focal"])
    g.add_argument("--alpha", type=float, default=2.0,
                   help="extreme loss threshold, in standard deviations from the mean")
    g.add_argument("--steps", default=DEFAULT_STEPS,
                   help=f"comma list, run in this order, among {','.join(ALL_STEPS)}")
    g.add_argument("--loaders", default="train,val,test",
                   help="comma list among train,val,test for the metrics and plot steps")
    g.add_argument("--paper-settings", action="store_true",
                   help="load the hyperparameters used in the paper for this dataset/freq/"
                        "forecast/loss from configs/paper_runs.csv (flags you also give win)")
    g.add_argument("--exp-suffix", default="",
                   help="appended to the experiment folder name, e.g. _seed1")

    h = p.add_argument_group("model / training (defaults = gru/src/get_parser.py)")
    h.add_argument("--epochs", type=int, default=1000)
    h.add_argument("--batch-size", type=int, default=16)
    h.add_argument("--learning-rate", type=float, default=1e-4)
    h.add_argument("--latent-dim", type=int, default=120)
    h.add_argument("--dropout", type=float, default=0.0)
    h.add_argument("--recurrent-dropout", type=float, default=0.7)
    h.add_argument("--patience", type=int, default=10,
                   help="early stopping: stop after this many epochs without improvement")
    h.add_argument("--train-percent", type=float, default=0.8,
                   help="fraction of train_val_data.csv used for training (rest = validation)")
    h.add_argument("--seed", type=int, default=0)

    o = p.add_argument_group("other")
    o.add_argument("--force", action="store_true",
                   help="skip the dataset/frequency compatibility check")
    o.add_argument("--keep-going", action="store_true",
                   help="run every step even if an earlier one failed")
    return p


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.paper_row = None
    if args.paper_settings:
        # Two-pass parse: the paper values become the defaults, so any flag
        # given explicitly on the command line still takes priority.
        import pandas as pd
        df = pd.read_csv(PAPER_RUNS)
        sel = df[(df.dataset == args.dataset) & (df.freq == args.freq)
                 & (df.forecast == args.forecast) & (df.loss == args.loss)]
        if sel.empty:
            sys.exit(f"[paper-settings] no paper run for {args.dataset} {args.freq} "
                     f"forecast={args.forecast} loss={args.loss} in {PAPER_RUNS}")
        row = sel.iloc[0]
        types = {a.dest: a.type for a in p._actions if a.dest in PAPER_HPARAMS}
        p.set_defaults(**{k: types[k](row[k]) for k in PAPER_HPARAMS})
        args = p.parse_args(argv)
        args.paper_row = row
    return args


def main(argv=None):
    args = parse_args(argv)

    if not args.force and args.freq not in VALID_FREQ[args.dataset]:
        sys.exit(f"[config] {args.dataset} is only available at freq "
                 f"{sorted(VALID_FREQ[args.dataset])}; got --freq {args.freq}. "
                 f"Use --force to override.")

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        sys.exit(f"[config] unknown step(s): {unknown}; choose from {ALL_STEPS}")

    loaders = [x.strip() for x in args.loaders.split(",") if x.strip()]
    unknown = set(loaders) - {"train", "val", "test"}
    if unknown:
        sys.exit(f"[config] unknown loader(s): {sorted(unknown)}")

    fc = args.forecast
    tag = f"GRU{fc}{fc}{fc}"
    dataset_root = join(args.data_root, args.dataset)        # .../<dataset>
    data_path = join(dataset_root, f"data_{args.freq}")      # .../<dataset>/data_<freq>
    save_dir = f"{tag}/experiment_{tag}_{args.loss}{args.exp_suffix}"   # relative to data_path
    exp_dir = join(data_path, save_dir)
    tensor_dir = join(data_path, tag)

    print("=" * 70)
    print("GRU auto-encoder pipeline")
    print("=" * 70)
    shown = ("dataset", "freq", "forecast", "loss", "alpha", "steps", "epochs", "batch_size",
             "learning_rate", "latent_dim", "dropout", "recurrent_dropout", "patience", "seed")
    for k in shown:
        print(f"  {k:18s}: {getattr(args, k)}")
    print(f"  {'data_path':18s}: {data_path}")
    print(f"  {'experiment dir':18s}: {exp_dir}")
    if args.paper_row is not None:
        r = args.paper_row
        print(f"  {'settings':18s}: paper run {r['source_run']}")
        print(f"  {'paper test metrics':18s}: RMSE {r['saved_rmse']:.4g}  NSE {r['saved_nse']:.4g}  KGE {r['saved_kge']:.4g}")
    print("=" * 70)

    # ---- pre-flight: required raw inputs -------------------------------------
    required = [join(data_path, "train_val_data.csv"), join(data_path, "test_data.csv")]
    if args.loss == "pp" and ("train" in steps or "predict" in steps):
        required.append(join(data_path, "df.csv"))
    missing = [p for p in required if not isfile(p)]
    if missing:
        sys.exit("[pre-flight] missing required input file(s):\n  " + "\n  ".join(missing)
                 + "\nDownload the data (see README) and place it under --data-root.")

    seed_everything(args.seed)
    os.makedirs(exp_dir, exist_ok=True)

    # Record every resolved setting next to the results
    try:
        import yaml
        resolved = {k: v for k, v in vars(args).items() if k != "paper_row"}
        if args.paper_row is not None:
            resolved["paper_source_run"] = str(args.paper_row["source_run"])
        with open(join(exp_dir, "run_config.yaml"), "w") as fh:
            yaml.safe_dump(resolved, fh, sort_keys=False)
    except ImportError:
        pass

    # Arguments shared by train / predict / metrics / plot (same names as get_parser.py)
    run_args = argparse.Namespace(
        loss_function=args.loss, alpha=args.alpha,
        epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
        latent_dim=args.latent_dim, dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout, patience=args.patience,
        seq_length_x=fc, seq_length_y=fc, shift=fc,
        data_dir=tag, save_dir=save_dir, save_model_name="GRU_best_model.h5",
        data_path=data_path,
    )

    # ---- steps -----------------------------------------------------------------
    def do_prepare():
        import generate_data
        gen = argparse.Namespace(
            seq_length_x=fc, seq_length_y=fc, shift=fc,
            scale_data=True, train_percent=args.train_percent,
            data_dir=dataset_root,
            df_path=f"data_{args.freq}",
            output_dir=f"data_{args.freq}/{tag}",
            df_train=f"data_{args.freq}/train_val_data.csv",
            df_test=f"data_{args.freq}/test_data.csv",
        )
        os.makedirs(tensor_dir, exist_ok=True)
        generate_data.main(gen)
        for f in ("x_train.npy", "y_train.npy", "x_val.npy", "x_test.npy"):
            arr = np.load(join(tensor_dir, f))
            print(f"    {f:14s} shape={arr.shape}")
            if arr.size == 0:
                raise AssertionError(f"{f} is empty - not enough rows for forecast={fc}.")

    def need_tensors():
        if not isfile(join(tensor_dir, "x_train.npy")):
            raise FileNotFoundError(f"no tensors in {tensor_dir} - run the 'prepare' step first")

    def do_train():
        need_tensors()
        import pickle
        import train as gru_train
        import matplotlib.pyplot as plt
        with open(join(exp_dir, "args.pkl"), "wb") as fh:
            pickle.dump(run_args, fh)
        gru_train.train(run_args)
        plt.close("all")
        model_file = join(exp_dir, run_args.save_model_name)
        if not isfile(model_file):
            raise AssertionError(f"training did not save {model_file}")
        print(f"    saved model: {model_file}")

    def do_predict():
        need_tensors()
        model_file = join(exp_dir, run_args.save_model_name)
        if not isfile(model_file):
            raise FileNotFoundError(f"no trained model at {model_file} - run 'train' first "
                                    "or place a trained model there")
        if args.loss == "pp":   # pp loss reads a look-up table built from df.csv
            from model import _get_pp_ranking
            _get_pp_ranking(join(data_path, "df.csv"))
        import compile_model_pred_data as cmp
        cmp.main(run_args)
        print(f"    prediction CSVs written to {exp_dir}")

    metric_rows = {}

    def do_metrics():
        import get_metrics as gm
        for ld in loaders:
            # test-set file keeps the paper's naming; other loaders get a suffix
            suffix = "" if ld == "test" else f"_{ld}"
            m_args = argparse.Namespace(**vars(run_args))
            m_args.loader = ld
            m_args.metric_csv = f"GRU_metrics_{fc}_{args.freq}_{args.loss}{args.exp_suffix}{suffix}.csv"
            m_args.metric_json = f"GRU_metrics_{fc}_{args.freq}_{args.loss}{args.exp_suffix}{suffix}.json"
            gm.main(m_args)
            with open(join(tensor_dir, m_args.metric_csv)) as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            metric_rows[ld] = dict(zip(METRIC_NAMES, (float(x) for x in lines[1].split(","))))
        print()
        print("    " + "loader".ljust(8) + "".join(m.rjust(12) for m in METRIC_NAMES))
        for ld, row in metric_rows.items():
            print("    " + ld.ljust(8) + "".join(f"{row[m]:12.4f}" for m in METRIC_NAMES))
        print(f"    metric files written to {tensor_dir}")

    def do_plot():
        import plot_results
        for ld in loaders:
            p_args = argparse.Namespace(**vars(run_args))
            p_args.loader, p_args.loss, p_args.forecast = ld, args.loss, fc
            p_args.ylabel = YLABEL[args.dataset]
            p_args.save_fig = join(exp_dir, f"plot_{ld}.png")
            plot_results.plot_one_loader_segment(p_args)
            print(f"    saved {p_args.save_fig}")

    actions = {"prepare": do_prepare, "train": do_train, "predict": do_predict,
               "metrics": do_metrics, "plot": do_plot}

    results = []

    def run_step(name):
        print(f"\n{'-' * 70}\n[STEP] {name}\n{'-' * 70}")
        try:
            actions[name]()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            results.append((name, "FAIL", repr(exc)))
            print(f"[FAIL] {name}")
            return False
        results.append((name, "PASS", ""))
        print(f"[PASS] {name}")
        return True

    for name in steps:
        if not run_step(name) and not args.keep_going:
            break

    # ---- summary + report ---------------------------------------------------------
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, status, note in results:
        print(f"  {name.ljust(8)}  {status}" + (f"  ({note})" if note else ""))
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")

    report = join(exp_dir, "run_report.txt")
    with open(report, "w") as fh:
        fh.write("GRU auto-encoder pipeline\n" + "=" * 60 + "\n")
        for k in shown:
            fh.write(f"{k:18s}: {getattr(args, k)}\n")
        if args.paper_row is not None:
            r = args.paper_row
            fh.write(f"{'settings':18s}: paper run {r['source_run']}\n")
            fh.write(f"{'paper test metrics':18s}: RMSE {r['saved_rmse']:.6g}  NSE {r['saved_nse']:.6g}  KGE {r['saved_kge']:.6g}\n")
        fh.write(f"{'data_path':18s}: {data_path}\n\n")
        for name, status, note in results:
            fh.write(f"[{status}] {name}" + (f"  ({note})" if note else "") + "\n")
        if metric_rows:
            fh.write("\nmetrics (mean over stations)\n")
            fh.write("loader".ljust(8) + "".join(m.rjust(12) for m in METRIC_NAMES) + "\n")
            for ld, row in metric_rows.items():
                fh.write(ld.ljust(8) + "".join(f"{row[m]:12.4f}" for m in METRIC_NAMES) + "\n")
    print(f"\nreport written: {report}")
    print(f"\n{'DONE' if n_fail == 0 else f'FAILED ({n_fail} step(s))'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
