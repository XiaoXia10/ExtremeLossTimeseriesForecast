#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single entry point for the extreme-loss experiments (GRU and Graph WaveNet).

Choose what to run in a YAML file - model, dataset, resolution, forecast length
and loss function(s) - and this script fills in the hyperparameters used in the
paper (gru/configs/paper_runs.csv, gwn/configs/paper_runs.csv), then runs the
pipeline of the chosen model once per loss function.

    python run.py --config configs/yamaska_D.yaml
    python run.py --config configs/yamaska_D.yaml --dry-run    # show what would run

See configs/*.yaml for commented examples.
"""

import argparse
import importlib.util
import subprocess
import sys
from os.path import join, dirname, abspath, isfile

import pandas as pd

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

base_dir = dirname(abspath(__file__))

MODELS = {
    "gru": join(base_dir, "gru", "src", "run.py"),
    "gwn": join(base_dir, "gwn", "src", "run.py"),
}
VALID_FREQ = {"yamaska_data": ["D"], "milandre_data": ["H", "4H"]}
LOSSES = {
    "gru": ["mae", "extreme", "focal", "dense", "gumbel", "pp"],
    "gwn": ["mae", "extreme", "focal", "dense", "gumbel", "pp", "mse"],
}
STEPS = ["prepare", "train", "predict", "metrics", "plot"]
TOP_KEYS = {"model", "dataset", "freq", "forecast", "loss", "steps", "loaders",
            "hyperparameters", "overrides", "data_root", "exp_suffix", "keep_going"}
# Overrides handled by the YAML structure itself, not allowed under `overrides:`
RESERVED = {"dataset", "freq", "forecast", "loss", "steps", "loaders", "paper_settings",
            "data_root", "exp_suffix", "force", "keep_going"}


def fail(msg):
    sys.exit(f"[config] {msg}")


def load_model_parser(model):
    """Return the argparse parser of <model>/src/run.py (used to check overrides)."""
    spec = importlib.util.spec_from_file_location(f"{model}_run", MODELS[model])
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, dirname(MODELS[model]))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod.build_parser()


def as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def read_config(path):
    if not isfile(path):
        fail(f"config file not found: {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    unknown = set(cfg) - TOP_KEYS
    if unknown:
        fail(f"unknown key(s) {sorted(unknown)} in {path}; allowed: {sorted(TOP_KEYS)}")
    for key in ("model", "dataset", "freq", "forecast", "loss"):
        if key not in cfg:
            fail(f"'{key}' is missing in {path}")
    return cfg


def validate(cfg):
    model = str(cfg["model"]).lower()
    if model not in MODELS:
        fail(f"model must be one of {list(MODELS)}, got '{cfg['model']}'")
    cfg["model"] = model

    dataset, freq = cfg["dataset"], str(cfg["freq"])
    if dataset not in VALID_FREQ:
        fail(f"dataset must be one of {list(VALID_FREQ)}, got '{dataset}'")
    if freq not in VALID_FREQ[dataset]:
        fail(f"{dataset} is only available at freq {VALID_FREQ[dataset]}, got '{freq}'")
    cfg["freq"] = freq

    try:
        cfg["forecast"] = int(cfg["forecast"])
    except (TypeError, ValueError):
        fail(f"forecast must be an integer, got '{cfg['forecast']}'")

    losses = as_list(cfg["loss"])
    if not losses:
        fail("choose at least one loss function")
    bad = [l for l in losses if l not in LOSSES[model]]
    if bad:
        fail(f"unknown loss(es) {bad} for {model}; choose from {LOSSES[model]}")
    cfg["loss"] = losses

    steps = as_list(cfg.get("steps", ["prepare", "train", "predict", "metrics"]))
    bad = [s for s in steps if s not in STEPS]
    if bad:
        fail(f"unknown step(s) {bad}; choose from {STEPS}")
    cfg["steps"] = steps

    hp = str(cfg.get("hyperparameters", "paper")).lower()
    if hp not in ("paper", "default"):
        fail(f"hyperparameters must be 'paper' or 'default', got '{hp}'")
    cfg["hyperparameters"] = hp

    table = pd.read_csv(join(base_dir, model, "configs", "paper_runs.csv"))
    cfg["_table"] = table
    if hp == "paper":
        here = table[(table.dataset == dataset) & (table.freq == freq)]
        if cfg["forecast"] not in set(here.forecast):
            fail(f"forecast {cfg['forecast']} was not used in the paper for {dataset} {freq} "
                 f"(paper: {sorted(set(here.forecast))}). Pick one of those, or set "
                 f"'hyperparameters: default' to use the built-in defaults.")
        missing = [l for l in losses
                   if here[(here.forecast == cfg["forecast"]) & (here.loss == l)].empty]
        if missing:
            fail(f"no paper hyperparameters for loss(es) {missing}; set "
                 f"'hyperparameters: default' to run them with the built-in defaults.")

    overrides = cfg.get("overrides") or {}
    if not isinstance(overrides, dict):
        fail("'overrides' must be a mapping, e.g.  overrides: {epochs: 5}")
    parser = load_model_parser(model)
    options = {a.dest: a for a in parser._actions if a.option_strings}
    for key in overrides:
        dest = key.replace("-", "_")
        if dest in RESERVED or dest not in options:
            allowed = sorted(d for d in options if d not in RESERVED | {"help"})
            fail(f"'{key}' cannot be overridden for {model}; allowed: {allowed}")
    cfg["_options"] = options
    return cfg


def build_argv(cfg, loss):
    argv = ["--dataset", cfg["dataset"], "--freq", cfg["freq"],
            "--forecast", str(cfg["forecast"]), "--loss", loss,
            "--steps", ",".join(cfg["steps"])]
    if cfg.get("loaders"):
        argv += ["--loaders", ",".join(as_list(cfg["loaders"]))]
    if cfg.get("data_root"):
        argv += ["--data-root", str(cfg["data_root"])]
    if cfg.get("exp_suffix"):
        argv += ["--exp-suffix", str(cfg["exp_suffix"])]
    if cfg["hyperparameters"] == "paper":
        argv.append("--paper-settings")
    for key, value in (cfg.get("overrides") or {}).items():
        action = cfg["_options"][key.replace("-", "_")]
        flag = action.option_strings[0]
        if action.nargs == 0:              # on/off switch
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def metric_file(cfg, loss):
    model, ds, freq, fc = cfg["model"], cfg["dataset"], cfg["freq"], cfg["forecast"]
    root = cfg.get("data_root") or join(base_dir, model)
    sfx = cfg.get("exp_suffix") or ""
    data_path = join(root, ds, f"data_{freq}")
    if model == "gru":
        return join(data_path, f"GRU{fc}{fc}{fc}", f"GRU_metrics_{fc}_{freq}_{loss}{sfx}.csv")
    return join(data_path, f"GWN_{fc}", f"experiment_{fc}_{loss}{sfx}", f"GWN_metrics_{fc}_{freq}_{loss}{sfx}.csv")


def read_test_metrics(path):
    try:
        with open(path) as fh:
            v = [float(x) for x in fh.read().splitlines()[1].split(",")]
        return {"RMSE": v[0], "NSE": v[4], "KGE": v[5]}
    except Exception:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML file, see configs/*.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = p.parse_args(argv)
    # keep our messages in order with the model's output, also when piped to a file
    sys.stdout.reconfigure(line_buffering=True)

    cfg = validate(read_config(a.config))
    model, losses = cfg["model"], cfg["loss"]

    print("=" * 70)
    print(f"model {model.upper()} | {cfg['dataset']} {cfg['freq']} | forecast {cfg['forecast']} | "
          f"hyperparameters: {cfg['hyperparameters']}")
    print(f"losses: {', '.join(losses)} | steps: {', '.join(cfg['steps'])}")
    if cfg.get("overrides"):
        print(f"overrides: {cfg['overrides']}")
    print("=" * 70)

    status = {}
    for i, loss in enumerate(losses, 1):
        cmd = [sys.executable, MODELS[model]] + build_argv(cfg, loss)
        print(f"\n##### [{i}/{len(losses)}] {model.upper()} - {loss} loss")
        print("      " + " ".join(cmd[1:]))
        if a.dry_run:
            continue
        rc = subprocess.call(cmd)
        status[loss] = rc
        if rc != 0 and not cfg.get("keep_going", False):
            print(f"\n[stop] {loss} failed (exit {rc}); set 'keep_going: true' to continue with the next loss")
            break

    if a.dry_run:
        return 0

    # ---- summary: test-set metrics next to the paper's ----------------------
    table = cfg["_table"]
    rows = []
    for loss, rc in status.items():
        got = read_test_metrics(metric_file(cfg, loss)) if "metrics" in cfg["steps"] else None
        ref = table[(table.dataset == cfg["dataset"]) & (table.freq == cfg["freq"])
                    & (table.forecast == cfg["forecast"]) & (table.loss == loss)]
        row = {"loss": loss, "status": "ok" if rc == 0 else f"failed ({rc})"}
        if got:
            row.update({k: round(v, 4) for k, v in got.items()})
        if not ref.empty:
            r = ref.iloc[0]
            row.update({"paper_RMSE": round(r.saved_rmse, 4), "paper_NSE": round(r.saved_nse, 4),
                        "paper_KGE": round(r.saved_kge, 4)})
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(f"\n{'=' * 70}\nSUMMARY (test set, mean over stations)\n{'=' * 70}")
    print(summary.to_string(index=False))
    return 0 if all(rc == 0 for rc in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
