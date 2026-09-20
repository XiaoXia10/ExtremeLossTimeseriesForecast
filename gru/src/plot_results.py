# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:30:35 2024

@author: Xiao Xia Liang
"""

import argparse
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from data_scaler import destandardize_pred

def plot_one_loader_segment(args):
    
    yhat, realy = destandardize_pred(args.data_path, f'{args.data_path}/{args.save_dir}', f'{args.data_path}/{args.data_dir}', args.loader)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,figsize=(20, 20), sharex=True,)

    ax1.plot(realy.index, realy.iloc[:,0], label="Measured - ", linewidth=3)
    ax1.plot(yhat.index, yhat.iloc[:,0], label= "Predicted - "+loss+ " loss", linewidth=3)
    ax1.set_title("GWN - "+ " Forecast Length " + str(forecast), fontsize = 35)
    ax1.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax2.plot(realy.index, realy.iloc[:,1], label="Measured - ", linewidth=3)
    ax2.plot(yhat.index, yhat.iloc[:,1], label="Predicted - "+loss+ " loss", linewidth=3)
    ax2.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax3.plot(realy.index, realy.iloc[:,2], label="Measured - ", linewidth=3)
    ax3.plot(yhat.index, yhat.iloc[:,2], label="Predicted - "+loss+ " loss", linewidth=3)
    ax3.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax4.plot(realy.index, realy.iloc[:,3], label="Measured - ", linewidth=3)
    ax4.plot(yhat.index, yhat.iloc[:,3], label="Predicted - "+loss+ " loss", linewidth=3)
    ax4.set_xlabel("Date", fontsize = 25)
    ax4.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax1.tick_params(labelsize=25)
    ax1.legend(fontsize = 25)
    ax2.tick_params(labelsize=25)
    ax2.legend(fontsize = 25)
    ax3.tick_params(labelsize=25)
    ax3.legend(fontsize = 25)
    ax4.tick_params(labelsize=25)
    ax4.legend(fontsize = 25)
    
def main(args):

    plot_one_loader_segment(args)
    
if __name__ == "__main__":
    
    loss ="extreme" #[mae, extreme, gumbel, dense, pp, focal]
    dataset = "milandre_data" #[milandre_data, yamaska_data]

    freq = "4H"  #[H, 4H, D]
    
    seq_length_x, seq_length_y, shift  = 12, 12, 12 #[6,9,12]
    forecast = 12

    
    parser = argparse.ArgumentParser()
    
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
    
    parser.add_argument("--data_path", type=str, default="../gru/"+dataset+"/data_"+freq, help="Data path")
    
    args = parser.parse_args()


    main(args)



