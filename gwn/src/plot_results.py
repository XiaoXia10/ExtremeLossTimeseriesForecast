# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:05:41 2024

@author: Xiao Xia Liang
"""
import argparse
import pandas as pd
from os.path import join, dirname, abspath
import matplotlib.pyplot as plt
import numpy as np
from data_scaler import destandardize_pred, exp10_pred

# Repo-relative default paths: <repo>/gwn/src/this_file.py -> base_dir = <repo>/gwn
current_dir = dirname(abspath(__file__))
base_dir = dirname(current_dir)

def plot_one_loader_segment(args):
    """Measured vs predicted for every station of one loader (train/val/test).

    Uses args.loss, args.forecast and (optionally) args.ylabel / args.save_fig.
    """
    yhat, realy = destandardize_pred(args.data_path, f"{args.data_path}/{args.data_dir}", f"{args.data_path}/{args.time_dir}", args.loader)
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
    fig.suptitle("GWN - " + args.loss + " loss - forecast length " + str(args.forecast) + " - " + args.loader,
                 fontsize=18, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    save_fig = getattr(args, "save_fig", None)
    if save_fig:
        fig.savefig(save_fig, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_entire_timerseries(args):
    
    station = 1 # 0 to 3 total of 4 stations, "Milandrine","Bame","Saivu","Font"  
    names = args.list_names
    
    loaders = ["train","val", "test"]
    df_all_yhat = pd.DataFrame()
    df_all_realy = pd.DataFrame()
    
    plt.figure(figsize=(25,5))
    for loader in loaders:
        yhat, realy = destandardize_pred(args.data_path, f"{args.data_path}/{args.data_dir}", f"{args.data_path}/{args.time_dir}", loader)
        
        df_all_yhat = pd.concat([df_all_yhat, yhat])
        df_all_realy = pd.concat([df_all_realy, realy])
        
        plt.plot(realy.index, realy.iloc[:,station], color="r", linewidth=2)
        plt.plot(yhat.index, yhat.iloc[:,station], label="Pred_"+loader, linewidth=2)
    
    plt.title(str(names[station]), fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Discharge (l/s)', fontsize=20) # Milandrine is measured at l/s, the other springs are measured in m^3/s
    plt.legend(fontsize=20)
    plt.axis("equal")  


def cross_plot(args):
    yhat_ext, realy = destandardize_pred(f"{args.data_path}/{args.df_path}", f"{args.data_path}/{args.data_dir_ext}", f"{args.data_path}/{args.time_dir}", args.loader)
    yhat_mae, _ = destandardize_pred(f"{args.data_path}/{args.df_path}", f"{args.data_path}/{args.data_dir_mae}", f"{args.data_path}/{args.time_dir}", args.loader)
   
    
    fig, axes = plt.subplots(1,2, figsize=(20, 10))

    axes[0].plot(realy.iloc[:,2], yhat_mae.iloc[:,2], "o",  color="blue", label="Saivu - MAE Loss", ms=6)
    axes[0].plot(realy.iloc[:,2], yhat_ext.iloc[:,2],  "*", color="orange",label="Saivu - EXT Loss", ms=6)
    axes[0].plot([0,realy.iloc[:,2].max()], [0,realy.iloc[:,2].max()], "k--", linewidth = 3)
    axes[0].set_xlabel("Measured (l/s)", fontsize = 30)
    axes[0].set_ylabel("Predicted (l/s)", fontsize = 30)
    axes[0].set(xlim=(0,realy.iloc[:,2].max()), ylim=(0,realy.iloc[:,2].max()))
    
    axes[1].plot(realy.iloc[:,3], yhat_mae.iloc[:,3], "o",  color="blue", label="Font - MAE Loss", ms=6)
    axes[1].plot(realy.iloc[:,3], yhat_ext.iloc[:,3],  "*", color="orange", label="Font - EXT Loss", ms=6)
    axes[1].plot([0,realy.iloc[:,3].max()], [0,realy.iloc[:,3].max()], "k--", linewidth = 3)
    axes[1].set_xlabel("Measured (l/s)", fontsize = 30)
    axes[1].set_ylabel("Predicted (l/s)", fontsize = 30)
    axes[1].set(xlim=(0,realy.iloc[:,3].max()), ylim=(0,realy.iloc[:,3].max()))
    
    axes[0].tick_params(labelsize=30)
    axes[0].legend(fontsize =30)
    axes[1].tick_params(labelsize=30)
    axes[1].legend(fontsize =30)

    # plt.suptitle("GWN - "+ " Forecast Length " + str(forecast) +"Time Steps", fontsize = 35)
    # plt.tight_layout(pad=2.0)
    # plt.savefig(join(path, "GWN_forecast"+str(forecast)+"_CP.png"))

         
def plot_compare(args):

    fig, (ax3, ax4) = plt.subplots(2,figsize=(20, 20), sharex=True,)
    
    ax3.plot(realy.iloc[:,2], "--", color = "k", label="Saivu - Meas", linewidth = 3)
    ax3.plot(yhat_ext.iloc[:,2], "orange", label="EXT Loss - Pred", linewidth = 3)
    ax3.plot(yhat_mae.iloc[:,2], "blue", label="MAE Loss - Pred", linewidth = 3)
    ax3.set_ylabel("Flow Rate (l/s)", fontsize = 30)
    
    ax4.plot(realy.iloc[:,3], "--", color = "k", label="Font - Meas", linewidth = 3)
    ax4.plot(yhat_ext.iloc[:,3], "orange", label="EXT Loss - Pred", linewidth = 3)
    ax4.plot(yhat_mae.iloc[:,3], "blue", label="MAE Loss - Pred", linewidth = 3)
    ax4.set_ylabel("Flow Rate (l/s)", fontsize = 30)
    ax4.set_xlabel("Date", fontsize = 30)

    ax3.tick_params(labelsize=30)
    ax3.legend(fontsize = 30)
    ax4.tick_params(labelsize=30)
    ax4.legend(fontsize = 30)
    # plt.savefig(join(path, "GWN_forecast"+str(forecast)+".png"))
    
def generate_plot(args):
    
    plot_one_loader_segment(args)
    
    
if __name__ == "__main__":

    forecast = 12

    loss ="mae" #[mse, mae, extreme, focal, pp, dense, gumbel]

    freq = "4H" # [H, 4H, D]
    dataset = "milandre_data"
    # dataset = "yamaska_data"
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss", type=str, default=loss, help="Loss function name (used for labels).")
    parser.add_argument("--forecast", type=int, default=forecast, help="Forecast length (used for labels).")
    parser.add_argument("--ylabel", type=str, default="Discharge (l/s)" if dataset == "milandre_data" else "Groundwater level (masl)", help="y-axis label")
    parser.add_argument("--save_fig", type=str, default=None, help="If set, save the figure to this path.")
    
    parser.add_argument("--data_dir", type=str, default="GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_"+loss, help="Model predicted data directory.")
    parser.add_argument("--time_dir", type=str, default="GWN_"+str(forecast), help="Directory for time.",)

    parser.add_argument("--data_dir_ext", type=str, default="GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_extreme", help="Model predicted data directory.")
    parser.add_argument("--data_dir_mae", type=str, default="GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_mae", help="Model predicted data directory.")

    parser.add_argument("--loader", type=str, default="test", help="Type of loaders - train, val, test.",)
    
    parser.add_argument("--data_path", type=str, default=join(base_dir, dataset, "data_"+str(freq)), help="Data path: <base_dir>/<dataset>/data_<freq>")
    
    args = parser.parse_args()

    generate_plot(args)
    
    
    
    
    
    