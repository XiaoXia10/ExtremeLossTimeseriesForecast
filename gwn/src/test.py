#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end reviewer test for the Graph WaveNet (GWN) + Extreme Loss pipeline.

For ONE configuration (dataset, frequency, forecast length, loss function) this
script runs the whole chain, re-using the project's own modules:

    1. compile   raw CSVs   -> train/val/test.npz            (generate_training_data.py)
    2. train     .npz       -> best_model.pth                (train.py)
    3. predict   best model -> {loader}_{realy,predy}.csv    (compile_model_pred.py)
    4. metrics   preds      -> RMSE/MAE/MSE/MAPE/NSE/KGE     (get_metrics.py)

Plotting is intentionally NOT run here - run plot_results.py separately.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python gwn/src/test.py [options]

--data-root defaults to the repo's own gwn/ data folder, so the script runs
with no flags at all. Pass --data-root /path/to/data to point at a different
copy (e.g. a reviewer's downloaded dataset - see README). The expected layout
underneath it is:

    <data-root>/<dataset>/data_<freq>/train_val_data.csv
    <data-root>/<dataset>/data_<freq>/test_data.csv
    <data-root>/<dataset>/data_<freq>/df.csv        # only needed for --loss pp
    <data-root>/<dataset>/adj_mat/adj_mx.npy        # graph adjacency (see note)

If adj_mx.npy is absent an identity placeholder is written: this test runs GWN
with graph-convolution and the adaptive-adjacency branch DISABLED, so the
adjacency matrix does not influence the result.

Everything the script produces is written under
    <data-root>/<dataset>/data_<freq>/GWN_<forecast>/ ...
so the raw CSVs are never modified.
--------------------------------------------------------------------------------
"""

import argparse
import os
import sys
import types
import pickle
import random
import traceback
from os.path import join, isfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

# exp_results.py / train.py import matplotlib.pyplot - stay headless
import matplotlib
matplotlib.use("Agg")

# ---- tiny stand-ins for optional deps that are imported at module load -----
try:
    import durbango  # noqa: F401
except Exception:
    _d = types.ModuleType("durbango")

    def _pickle_save(obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)

    _d.pickle_save = _pickle_save
    sys.modules["durbango"] = _d

try:
    import fastprogress  # noqa: F401
except Exception:
    _fp = types.ModuleType("fastprogress")

    class _MB(list):
        comment = ""

    def _progress_bar(it, *a, **k):
        return _MB(it)

    _fp.progress_bar = _progress_bar
    sys.modules["fastprogress"] = _fp


VALID_FREQ = {
    "milandre_data": {"H", "4H"},
    "yamaska_data": {"D"},
}
METRIC_NAMES = ["RMSE", "MAE", "MSE", "MAPE", "NSE", "KGE"]


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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="End-to-end Graph WaveNet test for reviewers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="../gwn/",
                   help="path to your gwn model folder")
    p.add_argument("--dataset", default="milandre_data",
                   choices=["milandre_data", "yamaska_data"])
    p.add_argument("--freq", default="4H", choices=["H", "4H", "D"])
    p.add_argument("--forecast", type=int, default=12,
                   help="sequence length = prediction horizon = window shift")
    p.add_argument("--loss", default="extreme",
                   choices=["mae", "mse", "extreme", "gumbel", "dense", "pp", "focal"])
    p.add_argument("--alpha", type=float, default=2.0,
                   help="threshold (in std-devs) for the extreme loss")
    p.add_argument("--scaler", default="log", choices=["log", "zscore"],
                   help="log = shipped behaviour (log10 forward transform); this also "
                        "swaps get_metrics' linear de-scaling for the matching exp10 "
                        "inverse. zscore = standardise instead, get_metrics unchanged.")
    p.add_argument("--epochs", type=int, default=25,
                   help="keep small for a smoke test; raise to reproduce a paper run")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clip", type=int, default=1)
    p.add_argument("--lr-decay-rate", type=float, default=0.98)
    p.add_argument("--nhid", type=int, default=None,
                   help="internal conv channels (default: = --forecast, as shipped)")
    p.add_argument("--dropout", type=float, default=0.7)
    p.add_argument("--es-patience", type=int, default=10)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu"])
    p.add_argument("--loaders", default="test",
                   help="comma list among train,val,test to score in the metrics stage")
    p.add_argument("--force", action="store_true",
                   help="skip the dataset/frequency compatibility check")
    p.add_argument("--keep-going", action="store_true",
                   help="run every stage even if an earlier one failed")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.force and args.freq not in VALID_FREQ[args.dataset]:
        sys.exit(f"[config] {args.dataset} is only published at freq "
                 f"{sorted(VALID_FREQ[args.dataset])}; got --freq {args.freq}. "
                 f"Use --force to override.")

    loaders = [x.strip() for x in args.loaders.split(",") if x.strip()]
    unknown = set(loaders) - {"train", "val", "test"}
    if unknown:
        sys.exit(f"[config] unknown loader(s): {sorted(unknown)}")

    fc = args.forecast
    nhid = args.nhid if args.nhid is not None else fc
    dataset_root = join(args.data_root, args.dataset)             # .../<dataset>
    freq_dir = f"data_{args.freq}"
    data_sub = f"{freq_dir}/GWN_{fc}"                             # rel. to dataset_root
    save_sub = f"{freq_dir}/GWN_{fc}/experiment_{fc}_{args.loss}_testscript"  # rel. to dataset_root
    csv_dir = join(dataset_root, freq_dir)                        # holds the raw CSVs
    compiled_dir = join(dataset_root, data_sub)
    exp_dir = join(dataset_root, save_sub)

    print("=" * 70)
    print("Graph WaveNet end-to-end test")
    print("=" * 70)
    for k in ("dataset", "freq", "forecast", "loss", "alpha", "scaler", "epochs",
              "batch_size", "learning_rate", "device", "seed"):
        print(f"  {k:16s}: {getattr(args, k)}")
    print(f"  dataset root    : {dataset_root}")
    print(f"  experiment dir  : {exp_dir}")
    print("=" * 70)

    # ---- pre-flight: required raw inputs ---------------------------------
    required = [join(csv_dir, "train_val_data.csv"),
                join(csv_dir, "test_data.csv")]
    if args.loss == "pp":
        required.append(join(csv_dir, "df.csv"))
    missing = [p for p in required if not isfile(p)]
    if missing:
        sys.exit("[pre-flight] missing required input file(s):\n  " + "\n  ".join(missing))

    num_nodes = pd.read_csv(required[0], index_col=0, nrows=5).shape[1]
    print(f"[pre-flight] num_nodes = {num_nodes} (columns of train_val_data.csv)")

    # adjacency: use the shipped file, else drop a harmless identity placeholder
    # (graph conv + adaptive adj are disabled below, so it does not affect output)
    adj_rel = "adj_mat/adj_mx.npy"
    adj_abs = join(dataset_root, adj_rel)
    if not isfile(adj_abs):
        os.makedirs(join(dataset_root, "adj_mat"), exist_ok=True)
        np.save(adj_abs, np.stack([np.eye(num_nodes, dtype=np.float32)] * 2))
        print(f"[pre-flight] adj_mx.npy not found - wrote identity placeholder to {adj_abs}")

    seed_everything(args.seed)
    os.makedirs(exp_dir, exist_ok=True)

    if args.device == "cpu":
        import util
        import torch
        util.default_device = lambda: torch.device("cpu")
        print("[config] forcing device = cpu")

    stages = []

    def run_stage(name, fn):
        print(f"\n{'-' * 70}\n[STAGE] {name}\n{'-' * 70}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            stages.append((name, "FAIL", repr(exc)))
            print(f"[FAIL] {name}")
            return False
        stages.append((name, "PASS", ""))
        print(f"[PASS] {name}")
        return True

    # ---- stage 1: compile ---------------------------------------------
    def do_compile():
        import generate_training_data as gtd
        if args.scaler == "zscore":
            gtd.log_df = gtd.standardize_df
            print("    scaler = zscore  (patched generate_training_data.log_df)")
        else:
            print("    scaler = log10   (shipped behaviour)")
        gen = argparse.Namespace(
            seq_length_x=fc, seq_length_y=fc, shift=fc,
            scale_data=True, train_split=args.train_split,
            data_path=dataset_root,
            df_path=freq_dir,
            output_dir=data_sub,
            df_train=f"{freq_dir}/train_val_data.csv",
            df_test=f"{freq_dir}/test_data.csv",
        )
        os.makedirs(join(gen.data_path, gen.output_dir), exist_ok=True)
        gtd.generate_train_val_test(gen)

        for f in ("train.npz", "val.npz", "test.npz",
                  "train_time.npy", "val_time.npy", "test_time.npy"):
            if not isfile(join(compiled_dir, f)):
                raise AssertionError(f"compile did not produce {f}")
        for cat in ("train", "val", "test"):
            d = np.load(join(compiled_dir, f"{cat}.npz"))
            print(f"    {cat}.npz  x={d['x'].shape}  y={d['y'].shape}")
            if d["x"].size == 0:
                raise AssertionError(
                    f"{cat}.npz is empty - the CSV does not have enough rows for "
                    f"forecast={fc} (try a smaller --forecast or more data).")

    # ---- args shared by stages 2-3 ---------------------------------
    run_args = argparse.Namespace(
        loss_function=args.loss,
        data_path=dataset_root, data=data_sub, save=save_sub,
        batch_size=args.batch_size, n_obs=None, fill_zeroes=False,
        adjdata=adj_rel,
        randomadj=False, aptonly=False, addaptadj=False, do_graph_conv=False,
        dropout=args.dropout, in_dim=2, apt_size=10,
        seq_length=fc, shift=fc, nhid=nhid, num_nodes=num_nodes, cat_feat_gc=False,
        checkpoint=None,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        clip=args.clip, lr_decay_rate=args.lr_decay_rate,
        epochs=args.epochs, n_iters=None, es_patience=args.es_patience,
        plot_history=False,
        df_path=join(csv_dir, "df.csv"),
    )

    def patch_alpha():
        if args.loss != "extreme":
            return
        import util
        base = util.extreme_value_loss
        if getattr(base, "_alpha_patched", False):
            return

        def extreme_value_loss(y_true, y_pred):
            return base(y_true, y_pred, alpha=args.alpha)

        extreme_value_loss._alpha_patched = True
        util.extreme_value_loss = extreme_value_loss
        print(f"    patched util.extreme_value_loss with alpha={args.alpha}")

    # ---- stage 2: train ---------------------------------------------
    def do_train():
        patch_alpha()
        import train as gwn_train
        import matplotlib.pyplot as plt
        gwn_train.args = run_args          # train.eval_() reads a module-global `args`
        gwn_train.main(run_args)
        plt.close("all")
        mfile = join(exp_dir, "best_model.pth")
        if not isfile(mfile):
            raise AssertionError(f"training did not save {mfile}")
        if not isfile(join(exp_dir, "metrics.csv")):
            raise AssertionError("training did not write metrics.csv")
        print(f"    saved model: {mfile} ({os.path.getsize(mfile) / 1e6:.2f} MB)")

    # ---- stage 3: compile predictions --------------------------
    def do_predict():
        patch_alpha()
        import compile_model_pred as cmp
        for ld in loaders:
            cmp.main(run_args, ld)
            for kind in ("realy", "predy"):
                f = join(exp_dir, f"{ld}_{kind}.csv")
                if not isfile(f):
                    raise AssertionError(f"missing prediction file {f}")
        print(f"    prediction CSVs written for loaders: {loaders}")

    # ---- stage 4: metrics -------------------------------------
    metric_rows = {}

    def do_metrics():
        import get_metrics as gm
        import data_scaler
        if args.scaler == "log":
            gm.destandardize_pred = data_scaler.exp10_pred
            print("    metrics de-scaling: exp10_pred (matches log10 compile)")
        else:
            print("    metrics de-scaling: destandardize_pred (matches zscore compile)")
        for ld in loaders:
            m_args = argparse.Namespace(
                forecast=fc, alpha=args.alpha,
                data_dir=f"GWN_{fc}/experiment_{fc}_{args.loss}",
                time_dir=f"GWN_{fc}",
                loader=ld,
                metric_csv=f"GWN_metrics_{fc}_{args.freq}_{args.loss}_{ld}.csv",
                metric_json=f"GWN_metrics_{fc}_{args.freq}_{args.loss}_{ld}.json",
                data_path=csv_dir,   # NB: get_metrics wants the freq-suffixed path
            )
            results = gm.get_metrics(m_args)
            row = {m: float(np.mean(results[m])) for m in METRIC_NAMES}
            metric_rows[ld] = row
            nonfinite = [k for k, v in row.items() if not np.isfinite(v)]
            if nonfinite:
                raise AssertionError(f"{ld}: non-finite metric(s) {nonfinite}")
            if row["NSE"] > 1.0001:
                raise AssertionError(f"{ld}: NSE={row['NSE']:.4f} > 1 (impossible)")
        print()
        print("    " + "loader".ljust(8) + "".join(m.rjust(12) for m in METRIC_NAMES))
        for ld, row in metric_rows.items():
            print("    " + ld.ljust(8) + "".join(f"{row[m]:12.4f}" for m in METRIC_NAMES))

    ok = run_stage("1/4  compile data", do_compile)
    if ok or args.keep_going:
        ok = run_stage("2/4  train", do_train)
    if ok or args.keep_going:
        ok = run_stage("3/4  compile predictions", do_predict)
    if ok or args.keep_going:
        run_stage("4/4  metrics", do_metrics)

    # ---- summary + report -------------------------------------
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    width = max(len(n) for n, _, _ in stages)
    for name, status, note in stages:
        print(f"  {name.ljust(width)}  {status}" + (f"  ({note})" if note else ""))
    n_fail = sum(1 for _, s, _ in stages if s == "FAIL")

    report = join(exp_dir, "reviewer_test_report.txt")
    try:
        with open(report, "w") as fh:
            fh.write("Graph WaveNet end-to-end reviewer test\n")
            fh.write("=" * 60 + "\n")
            for k in ("dataset", "freq", "forecast", "loss", "alpha", "scaler",
                      "epochs", "batch_size", "learning_rate", "device", "seed"):
                fh.write(f"{k:16s}: {getattr(args, k)}\n")
            fh.write(f"dataset root    : {dataset_root}\n\n")
            for name, status, note in stages:
                fh.write(f"[{status}] {name}" + (f"  ({note})" if note else "") + "\n")
            if metric_rows:
                fh.write("\nmetrics\n")
                fh.write("loader".ljust(8) + "".join(m.rjust(12) for m in METRIC_NAMES) + "\n")
                for ld, row in metric_rows.items():
                    fh.write(ld.ljust(8) + "".join(f"{row[m]:12.4f}" for m in METRIC_NAMES) + "\n")
            fh.write("\nNOTES\n")
            fh.write("- Shipped generate_training_data scales with log10 but shipped\n")
            fh.write("  get_metrics de-scales linearly; --scaler keeps the pair consistent.\n")
            fh.write("- GWN ran with do_graph_conv=False and addaptadj=False, so the\n")
            fh.write("  adjacency matrix does not affect these numbers.\n")
            fh.write("- extreme_value_loss default alpha is 1.5 in gwn/src/util.py vs\n")
            fh.write("  2.0 in gru/src/model.py; this test pins it to --alpha.\n")
            fh.write("- The upstream code sets no RNG seed; --seed makes runs only\n")
            fh.write("  roughly comparable, not bit-identical.\n")
        print(f"\nreport written: {report}")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not write report: {exc})")

    print(f"\n{'PASSED' if n_fail == 0 else f'FAILED ({n_fail} stage(s))'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
