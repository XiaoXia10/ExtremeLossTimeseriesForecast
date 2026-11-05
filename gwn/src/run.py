#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single entry point for the Graph WaveNet (GWN) + loss-function experiments.

All settings are given once, on the command line, and passed to every step,
so the steps cannot disagree. Each step re-uses the project's own modules:

    prepare   raw CSVs   -> train/val/test.npz            (generate_training_data.py)
    train     .npz       -> best_model.pth                (train.py)
    predict   best model -> {loader}_{realy,predy}.csv    (compile_model_pred.py)
    metrics   preds      -> RMSE/MAE/MSE/MAPE/NSE/KGE     (get_metrics.py)
    plot      preds      -> measured-vs-predicted PNG     (plot_results.py)

Graph convolution and the adaptive adjacency matrix are switched off, as in the
paper's runs, so the adjacency matrix is loaded but does not affect the result.
Data are z-scored with the train/validation mean and standard deviation.

--------------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------------
Full run (prepare, train, predict, metrics):
    python gwn/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme

Re-train with the exact hyperparameters of the paper run (configs/paper_runs.csv):
    python gwn/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme --paper-settings

Score a trained model without retraining (e.g. a released paper model placed in
the experiment folder):
    python gwn/src/run.py --dataset yamaska_data --freq D --forecast 6 --loss extreme \\
        --steps predict,metrics

--------------------------------------------------------------------------------
FOLDER LAYOUT (relative to --data-root, default <repo>/gwn)
--------------------------------------------------------------------------------
    <dataset>/adj_mat/adj_mx.npy                  adjacency (identity written if absent)
    <dataset>/data_<freq>/train_val_data.csv      input (required)
    <dataset>/data_<freq>/test_data.csv           input (required)
    <dataset>/data_<freq>/df.csv                  input (only for --loss pp)
    <dataset>/data_<freq>/GWN_<f>/                tensors
    <dataset>/data_<freq>/GWN_<f>/experiment_<f>_<loss>/
                                                  model, predictions, metrics, plots, run_report.txt
--------------------------------------------------------------------------------
"""

import argparse
import os
import sys
import types
import random
import pickle
import traceback
from os.path import join, isfile, dirname, abspath

# Repo-relative default paths: <repo>/gwn/src/run.py -> base_dir = <repo>/gwn
current_dir = dirname(abspath(__file__))
base_dir = dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd

# train.py / exp_results.py / plot_results.py use matplotlib - stay headless
import matplotlib
matplotlib.use("Agg")

# fastprogress only draws a progress bar in train.py; fall back to a plain loop
try:
    import fastprogress  # noqa: F401
except Exception:
    _fp = types.ModuleType("fastprogress")

    class _MB(list):
        comment = ""

    _fp.progress_bar = lambda it, *a, **k: _MB(it)
    sys.modules["fastprogress"] = _fp


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
PAPER_HPARAMS = ['alpha', 'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'lr_decay_rate', 'clip', 'dropout', 'nhid', 'es_patience', 'train_split']


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass
    except Exception:
        pass


def build_parser():
    p = argparse.ArgumentParser(
        description="Run the Graph WaveNet pipeline (prepare, train, predict, metrics, plot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("experiment")
    g.add_argument("--data-root", default=base_dir,
                   help="folder containing <dataset>/data_<freq>/ (default: <repo>/gwn)")
    g.add_argument("--dataset", default="milandre_data",
                   choices=["milandre_data", "yamaska_data"])
    g.add_argument("--freq", default="4H", choices=["H", "4H", "D"])
    g.add_argument("--forecast", type=int, default=12,
                   help="sequence length = prediction horizon = window shift")
    g.add_argument("--loss", default="extreme",
                   choices=["mae", "mse", "extreme", "gumbel", "dense", "pp", "focal"])
    g.add_argument("--alpha", type=float, default=1.5,
                   help="extreme loss threshold, in standard deviations from the mean")
    g.add_argument("--steps", default=DEFAULT_STEPS,
                   help=f"comma list, run in this order, among {','.join(ALL_STEPS)}")
    g.add_argument("--loaders", default="train,val,test",
                   help="comma list among train,val,test for the predict, metrics and plot steps")
    g.add_argument("--paper-settings", action="store_true",
                   help="load the hyperparameters used in the paper for this dataset/freq/"
                        "forecast/loss from configs/paper_runs.csv (flags you also give win)")
    g.add_argument("--exp-suffix", default="",
                   help="appended to the experiment folder name, e.g. _seed1")

    h = p.add_argument_group("model / training (defaults = gwn/src/train.py and util.py)")
    h.add_argument("--epochs", type=int, default=500)
    h.add_argument("--batch-size", type=int, default=16)
    h.add_argument("--learning-rate", type=float, default=1e-4)
    h.add_argument("--weight-decay", type=float, default=1e-5)
    h.add_argument("--lr-decay-rate", type=float, default=0.98)
    h.add_argument("--clip", type=int, default=1, help="gradient clipping")
    h.add_argument("--dropout", type=float, default=0.7)
    h.add_argument("--nhid", type=int, default=None,
                   help="internal conv channels (default: = --forecast)")
    h.add_argument("--es-patience", type=int, default=10,
                   help="early stopping: stop after this many epochs without improvement")
    h.add_argument("--train-split", type=float, default=0.8,
                   help="fraction of train_val_data.csv used for training (rest = validation)")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--device", default="auto", choices=["auto", "cpu"],
                   help="auto = mps/cuda if available, else cpu")

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
    nhid = args.nhid if args.nhid is not None else fc
    dataset_root = join(args.data_root, args.dataset)             # .../<dataset>
    freq_dir = f"data_{args.freq}"
    csv_dir = join(dataset_root, freq_dir)                        # raw CSVs
    data_sub = f"{freq_dir}/GWN_{fc}"                             # relative to dataset_root
    exp_name = f"experiment_{fc}_{args.loss}{args.exp_suffix}"
    save_sub = f"{data_sub}/{exp_name}"                           # relative to dataset_root
    tensor_dir = join(dataset_root, data_sub)
    exp_dir = join(dataset_root, save_sub)

    print("=" * 70)
    print("Graph WaveNet pipeline")
    print("=" * 70)
    shown = ("dataset", "freq", "forecast", "loss", "alpha", "steps", "epochs", "batch_size",
             "learning_rate", "weight_decay", "lr_decay_rate", "clip", "dropout", "nhid",
             "es_patience", "device", "seed")
    for k in shown:
        print(f"  {k:18s}: {nhid if k == 'nhid' else getattr(args, k)}")
    print(f"  {'dataset root':18s}: {dataset_root}")
    print(f"  {'experiment dir':18s}: {exp_dir}")
    if args.paper_row is not None:
        r = args.paper_row
        print(f"  {'settings':18s}: paper run {r['source_run']}")
        print(f"  {'paper test metrics':18s}: RMSE {r['saved_rmse']:.4g}  NSE {r['saved_nse']:.4g}  KGE {r['saved_kge']:.4g}")
    print("=" * 70)

    # ---- pre-flight: required raw inputs -------------------------------------
    required = [join(csv_dir, "train_val_data.csv"), join(csv_dir, "test_data.csv")]
    if args.loss == "pp" and ("train" in steps or "predict" in steps):
        required.append(join(csv_dir, "df.csv"))
    missing = [p for p in required if not isfile(p)]
    if missing:
        sys.exit("[pre-flight] missing required input file(s):\n  " + "\n  ".join(missing)
                 + "\nDownload the data (see README) and place it under --data-root.")

    num_nodes = pd.read_csv(required[0], index_col=0, nrows=5).shape[1]
    print(f"[pre-flight] num_nodes = {num_nodes} (columns of train_val_data.csv)")

    # The adjacency matrix is loaded by the model code but not used (graph
    # convolution is off), so an identity placeholder is fine when it is absent.
    adj_rel = "adj_mat/adj_mx.npy"
    adj_abs = join(dataset_root, adj_rel)
    if not isfile(adj_abs):
        os.makedirs(dirname(adj_abs), exist_ok=True)
        np.save(adj_abs, np.eye(num_nodes, dtype=np.float32))
        print(f"[pre-flight] {adj_abs} not found - wrote identity placeholder")

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

    if args.device == "cpu":
        import util
        import torch
        util.default_device = lambda: torch.device("cpu")
        print("[config] forcing device = cpu")

    # Arguments shared by train / predict (same names as util.get_shared_arg_parser + train.py)
    run_args = argparse.Namespace(
        loss_function=args.loss, alpha=args.alpha,
        data_path=dataset_root, data=data_sub, save=save_sub,
        adjdata=adj_rel, df_path=join(csv_dir, "df.csv"),
        do_graph_conv=False, addaptadj=False, aptonly=False, randomadj=False,
        seq_length=fc, shift=fc, nhid=nhid, in_dim=2, num_nodes=num_nodes,
        batch_size=args.batch_size, dropout=args.dropout, n_obs=None, apt_size=10,
        cat_feat_gc=False, fill_zeroes=False, checkpoint=None,
        epochs=args.epochs, clip=args.clip, weight_decay=args.weight_decay,
        learning_rate=args.learning_rate, lr_decay_rate=args.lr_decay_rate,
        n_iters=None, es_patience=args.es_patience, plot_history=True,
    )

    # ---- steps -----------------------------------------------------------------
    def do_prepare():
        import generate_training_data as gtd
        gen = argparse.Namespace(
            seq_length_x=fc, seq_length_y=fc, shift=fc,
            scale_data=True, train_split=args.train_split,
            data_path=dataset_root, df_path=freq_dir, output_dir=data_sub,
            df_train=f"{freq_dir}/train_val_data.csv",
            df_test=f"{freq_dir}/test_data.csv",
        )
        os.makedirs(tensor_dir, exist_ok=True)
        gtd.generate_train_val_test(gen)
        for cat in ("train", "val", "test"):
            d = np.load(join(tensor_dir, f"{cat}.npz"))
            print(f"    {cat}.npz  x={d['x'].shape}  y={d['y'].shape}")
            if d["x"].size == 0:
                raise AssertionError(f"{cat}.npz is empty - not enough rows for forecast={fc}.")

    def need_tensors():
        if not isfile(join(tensor_dir, "train.npz")):
            raise FileNotFoundError(f"no tensors in {tensor_dir} - run the 'prepare' step first")

    def do_train():
        need_tensors()
        import train as gwn_train
        import matplotlib.pyplot as plt
        with open(join(exp_dir, "args.pkl"), "wb") as fh:
            pickle.dump(run_args, fh)
        gwn_train.main(run_args)
        plt.close("all")
        model_file = join(exp_dir, "best_model.pth")
        if not isfile(model_file):
            raise AssertionError(f"training did not save {model_file}")
        print(f"    saved model: {model_file}")

    def do_predict():
        need_tensors()
        model_file = join(exp_dir, "best_model.pth")
        if not isfile(model_file):
            raise FileNotFoundError(f"no trained model at {model_file} - run 'train' first "
                                    "or place a trained model there")
        if args.loss == "pp":   # pp loss reads a look-up table built from df.csv
            import util
            util._get_pp_ranking(run_args.df_path)
        import compile_model_pred as cmp
        for ld in loaders:
            cmp.main(run_args, ld)
        print(f"    prediction CSVs written to {exp_dir} for loaders: {loaders}")

    metric_rows = {}

    def do_metrics():
        import get_metrics as gm
        for ld in loaders:
            # test-set file keeps the paper's naming; other loaders get a suffix
            suffix = "" if ld == "test" else f"_{ld}"
            m_args = argparse.Namespace(
                loader=ld, forecast=fc, alpha=args.alpha,
                data_path=csv_dir, data_dir=f"GWN_{fc}/{exp_name}", time_dir=f"GWN_{fc}",
                metric_csv=f"GWN_metrics_{fc}_{args.freq}_{args.loss}{args.exp_suffix}{suffix}.csv",
                metric_json=f"GWN_metrics_{fc}_{args.freq}_{args.loss}{args.exp_suffix}{suffix}.json",
            )
            results = gm.get_metrics(m_args)
            metric_rows[ld] = {m: float(np.mean(results[m])) for m in METRIC_NAMES}
        print()
        print("    " + "loader".ljust(8) + "".join(m.rjust(12) for m in METRIC_NAMES))
        for ld, row in metric_rows.items():
            print("    " + ld.ljust(8) + "".join(f"{row[m]:12.4f}" for m in METRIC_NAMES))
        print(f"    metric files written to {exp_dir}")

    def do_plot():
        import plot_results
        for ld in loaders:
            p_args = argparse.Namespace(
                loader=ld, loss=args.loss, forecast=fc, ylabel=YLABEL[args.dataset],
                data_path=csv_dir, data_dir=f"GWN_{fc}/{exp_name}", time_dir=f"GWN_{fc}",
                save_fig=join(exp_dir, f"plot_{ld}.png"),
            )
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
        fh.write("Graph WaveNet pipeline\n" + "=" * 60 + "\n")
        for k in shown:
            fh.write(f"{k:18s}: {nhid if k == 'nhid' else getattr(args, k)}\n")
        if args.paper_row is not None:
            r = args.paper_row
            fh.write(f"{'settings':18s}: paper run {r['source_run']}\n")
            fh.write(f"{'paper test metrics':18s}: RMSE {r['saved_rmse']:.6g}  NSE {r['saved_nse']:.6g}  KGE {r['saved_kge']:.6g}\n")
        fh.write(f"{'dataset root':18s}: {dataset_root}\n")
        fh.write("graph conv        : off (do_graph_conv=False, addaptadj=False)\n\n")
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
