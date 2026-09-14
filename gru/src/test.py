#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end reviewer test for the GRU auto-encoder + Extreme Loss pipeline.

For ONE configuration (dataset, frequency, forecast length, loss function) this
script runs the whole chain, re-using the project's own modules:

    1. compile   raw CSVs   -> windowed tensors            (generate_data.py)
    2. train     tensors    -> GRU_best_model.h5           (train.py)
    3. predict   best model -> {loader}_{realy,predy}.csv  (compile_model_pred_data.py)
    4. metrics   preds      -> RMSE/MAE/MSE/MAPE/NSE/KGE   (get_metrics.py)

Plotting is intentionally NOT run here - run plot_results.py separately.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python gru/src/test.py --data-root /path/to/data [options]

The reviewer downloads the datasets (see README) and passes their location with
--data-root. The expected layout underneath it is:

    <data-root>/<dataset>/data_<freq>/train_val_data.csv
    <data-root>/<dataset>/data_<freq>/test_data.csv
    <data-root>/<dataset>/data_<freq>/df.csv       # only needed for --loss pp

Everything the script produces (tensors, model, prediction CSVs, metric files
and a reviewer_test_report.txt) is written under
    <data-root>/<dataset>/data_<freq>/GRU<f><f><f>/ ...
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

# train.py imports matplotlib.pyplot and draws a loss-history figure - stay headless
import matplotlib
matplotlib.use("Agg")

# train.py does `from durbango import pickle_save` at import time (it is only used
# in train.py's own __main__, which we never call). Provide a tiny stand-in so the
# reviewer does not have to install that package.
try:
    import durbango  # noqa: F401
except Exception:
    _stub = types.ModuleType("durbango")

    def _pickle_save(obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)

    _stub.pickle_save = _pickle_save
    sys.modules["durbango"] = _stub


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
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="End-to-end GRU auto-encoder test for reviewers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True,
                   help="folder that contains <dataset>/data_<freq>/*.csv")
    p.add_argument("--dataset", default="milandre_data",
                   choices=["milandre_data", "yamaska_data"])
    p.add_argument("--freq", default="4H", choices=["H", "4H", "D"])
    p.add_argument("--forecast", type=int, default=12,
                   help="sequence length = prediction horizon = window shift")
    p.add_argument("--loss", default="extreme",
                   choices=["mae", "extreme", "gumbel", "dense", "pp", "focal"])
    p.add_argument("--alpha", type=float, default=2.0,
                   help="threshold (in std-devs) for the extreme loss")
    p.add_argument("--epochs", type=int, default=5,
                   help="keep small for a smoke test; raise to reproduce a paper run")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--latent-dim", type=int, default=120)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--recurrent-dropout", type=float, default=0.7)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train-percent", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
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
    tag = f"GRU{fc}{fc}{fc}"
    dataset_root = join(args.data_root, args.dataset)        # .../<dataset>
    data_path = join(dataset_root, f"data_{args.freq}")      # .../<dataset>/data_<freq>
    save_dir = f"{tag}/experiment_{tag}_{args.loss}"         # relative to data_path
    exp_dir = join(data_path, save_dir)

    print("=" * 70)
    print("GRU auto-encoder end-to-end test")
    print("=" * 70)
    for k in ("dataset", "freq", "forecast", "loss", "alpha", "epochs",
              "batch_size", "learning_rate", "seed"):
        print(f"  {k:16s}: {getattr(args, k)}")
    print(f"  data_path       : {data_path}")
    print(f"  experiment dir  : {exp_dir}")
    print("=" * 70)

    # ---- pre-flight: required raw inputs -----------------------------------
    required = [join(data_path, "train_val_data.csv"),
                join(data_path, "test_data.csv")]
    if args.loss == "pp":
        required.append(join(data_path, "df.csv"))
    missing = [p for p in required if not isfile(p)]
    if missing:
        sys.exit("[pre-flight] missing required input file(s):\n  " + "\n  ".join(missing))

    seed_everything(args.seed)
    os.makedirs(exp_dir, exist_ok=True)

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

    # ---- stage 1: compile raw CSVs into windowed tensors -----------------
    def do_compile():
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
        os.makedirs(join(gen.data_dir, gen.output_dir), exist_ok=True)
        generate_data.main(gen)

        produced = ["x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy",
                    "x_test.npy", "y_test.npy",
                    "train_time.npy", "val_time.npy", "test_time.npy"]
        out = join(data_path, tag)
        miss = [f for f in produced if not isfile(join(out, f))]
        if miss:
            raise AssertionError(f"compile did not produce: {miss}")
        for f in ("x_train.npy", "y_train.npy", "x_val.npy", "x_test.npy"):
            arr = np.load(join(out, f))
            print(f"    {f:14s} shape={arr.shape} dtype={arr.dtype}")
            if arr.size == 0:
                raise AssertionError(
                    f"{f} is empty - the CSV does not have enough rows for "
                    f"forecast={fc} (try a smaller --forecast or more data).")

    # ---- args shared by stages 2-4 -------------------------------------
    run_args = argparse.Namespace(
        loss_function=args.loss,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
        patience=args.patience,
        seq_length_x=fc, seq_length_y=fc, shift=fc,
        data_dir=tag,
        save_dir=save_dir,
        save_model_name="GRU_best_model.h5",
        data_path=data_path,
        alpha=args.alpha,
        loader="test",
        metric_csv=f"GRU_metrics_{fc}_{args.freq}_{args.loss}.csv",
        metric_json=f"GRU_metrics_{fc}_{args.freq}_{args.loss}.json",
    )

    def patch_alpha():
        """Route the chosen --alpha into model.extreme_value_loss (its signature
        hard-codes alpha=2.0 and it is handed to model.compile by name)."""
        if args.loss != "extreme":
            return
        import model as gru_model
        base = gru_model.extreme_value_loss
        if getattr(base, "_alpha_patched", False):
            return

        def extreme_value_loss(y_true, y_pred):
            return base(y_true, y_pred, alpha=args.alpha)

        extreme_value_loss._alpha_patched = True
        gru_model.extreme_value_loss = extreme_value_loss
        print(f"    patched model.extreme_value_loss with alpha={args.alpha}")

    # ---- stage 2: train --------------------------------------------------
    def do_train():
        patch_alpha()
        import train as gru_train
        import matplotlib.pyplot as plt
        os.makedirs(exp_dir, exist_ok=True)
        gru_train.train(run_args)
        plt.close("all")
        model_file = join(exp_dir, run_args.save_model_name)
        if not isfile(model_file):
            raise AssertionError(f"training did not save {model_file}")
        print(f"    saved model: {model_file} "
              f"({os.path.getsize(model_file) / 1e6:.2f} MB)")

    # ---- stage 3: compile predictions --------------------------------
    def do_predict():
        patch_alpha()
        import compile_model_pred_data as cmp
        cmp.main(run_args)
        for ld in loaders:
            for kind in ("realy", "predy"):
                f = join(exp_dir, f"{ld}_{kind}.csv")
                if not isfile(f):
                    raise AssertionError(f"missing prediction file {f}")
        print(f"    prediction CSVs written for loaders: {loaders}")

    # ---- stage 4: metrics -----------------------------------------
    metric_rows = {}

    def do_metrics():
        import get_metrics as gm
        for ld in loaders:
            run_args.loader = ld
            run_args.metric_csv = f"GRU_metrics_{fc}_{args.freq}_{args.loss}_{ld}.csv"
            run_args.metric_json = f"GRU_metrics_{fc}_{args.freq}_{args.loss}_{ld}.json"
            gm.main(run_args)
            csv_path = join(data_path, tag, run_args.metric_csv)
            with open(csv_path) as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            vals = [float(x) for x in lines[1].split(",")]
            row = dict(zip(METRIC_NAMES, vals))
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

    # ---- summary + report ----------------------------------------
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    width = max(len(n) for n, _, _ in stages)
    for name, status, note in stages:
        print(f"  {name.ljust(width)}  {status}" + (f"  ({note})" if note else ""))
    n_fail = sum(1 for _, s, _ in stages if s == "FAIL")

    report = join(exp_dir, "reviewer_test_report.txt")
    try:
        with open(report, "w") as fh:
            fh.write("GRU auto-encoder end-to-end reviewer test\n")
            fh.write("=" * 60 + "\n")
            for k in ("dataset", "freq", "forecast", "loss", "alpha", "epochs",
                      "batch_size", "learning_rate", "seed"):
                fh.write(f"{k:16s}: {getattr(args, k)}\n")
            fh.write(f"data_path       : {data_path}\n\n")
            for name, status, note in stages:
                fh.write(f"[{status}] {name}" + (f"  ({note})" if note else "") + "\n")
            if metric_rows:
                fh.write("\nmetrics\n")
                fh.write("loader".ljust(8) + "".join(m.rjust(12) for m in METRIC_NAMES) + "\n")
                for ld, row in metric_rows.items():
                    fh.write(ld.ljust(8) + "".join(f"{row[m]:12.4f}" for m in METRIC_NAMES) + "\n")
            fh.write("\nNOTES\n")
            fh.write("- The upstream code sets no RNG seed; --seed makes runs only\n")
            fh.write("  roughly comparable, not bit-identical.\n")
            fh.write("- extreme_value_loss default alpha is 2.0 here (gru/src/model.py)\n")
            fh.write("  vs 1.5 in gwn/src/util.py; a third unused copy sits in\n")
            fh.write("  gru/src/loss_functions.py. This test pins alpha to --alpha.\n")
        print(f"\nreport written: {report}")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not write report: {exc})")

    print(f"\n{'PASSED' if n_fail == 0 else f'FAILED ({n_fail} stage(s))'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
