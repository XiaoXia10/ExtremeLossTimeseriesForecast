# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:30:35 2024

@author: Xiao Xia Liang
"""

import argparse
import pandas as pd
from os.path import join, dirname, abspath
import matplotlib.pyplot as plt
import numpy as np
from data_scaler import destandardize_pred

# Repo-relative default paths: <repo>/gru/src/this_file.py -> base_dir = <repo>/gru
current_dir = dirname(abspath(__file__))
base_dir = dirname(current_dir)

def plot_one_loader_segment(args):
    """Measured vs predicted for every station of one loader (train/val/test).

    Uses args.loss, args.forecast and (optionally) args.ylabel / args.save_fig.
    """
    yhat, realy = destandardize_pred(args.data_path, f'{args.data_path}/{args.save_dir}', f'{args.data_path}/{args.data_dir}', args.loader)
    names = list(pd.read_csv(join(args.data_path, "train_val_data.csv"), index_col=0, nrows=1).columns)
    n = realy.shape[1]
    ylabel = getattr(args, "ylabel", None) or "Value"

    fig, axes = plt.subplots(n, 1, figsize=(20, max(4, 3.5 * n)), sharex=True, squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(realy.index, realy.iloc[:, i], label="Measured", linewidth=2)
        ax.plot(yhat.index, yhat.iloc[:, i], label="Predicted - " + args.loss + " loss", linewidth=2)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(str(names[i]) if i < len(names) else f"station {i}", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
    axes[-1, 0].set_xlabel("Date", fontsize=14)
    fig.suptitle("GRU - " + args.loss + " loss - forecast length " + str(args.forecast) + " - " + args.loader,
                 fontsize=18, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    save_fig = getattr(args, "save_fig", None)
    if save_fig:
        fig.savefig(save_fig, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig

def main(args):

    plot_one_loader_segment(args)
    
if __name__ == "__main__":
    
    loss ="extreme" #[mae, extreme, gumbel, dense, pp, focal]
    dataset = "milandre_data" #[milandre_data, yamaska_data]

    freq = "4H"  #[H, 4H, D]
    
    seq_length_x, seq_length_y, shift  = 12, 12, 12 #[6,9,12]
    forecast = 12

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--loss", type=str, default=loss, help="Loss function name (used for labels).")
    parser.add_argument("--forecast", type=int, default=forecast, help="Forecast length (used for labels).")
    parser.add_argument("--ylabel", type=str, default="Discharge (l/s)" if dataset == "milandre_data" else "Groundwater level (masl)", help="y-axis label")
    parser.add_argument("--save_fig", type=str, default=None, help="If set, save the figure to this path.")
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    
    parser.add_argument("--data_dir", type=str, default="GRU"+str(seq_length_x)+str(seq_length_y)+str(shift), help="df for testing.",)
    parser.add_argument("--save_dir", type=str, default="GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"/experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_"+loss, help="Save Path")
    parser.add_argument("--save_model_name", type=str, default="GRU_best_model.h5", help="Best saved model name")
   
    parser.add_argument("--data_dir_ext", type=str, default="GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"/experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_extreme", help="Model predicted data directory.")
    parser.add_argument("--data_dir_mae", type=str, default="GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"/experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_mae", help="Model predicted data directory.")

    parser.add_argument("--one_loader", default=True, type=str, help="If true, will only plot the specified loader type.",)
    parser.add_argument("--loader", type=str, default="test", help="Type of loaders - train, val, test.",)
    
    parser.add_argument("--data_path", type=str, default=join(base_dir, dataset, "data_"+freq), help="Data path: <base_dir>/<dataset>/data_<freq>")
    
    args = parser.parse_args()


    main(args)



