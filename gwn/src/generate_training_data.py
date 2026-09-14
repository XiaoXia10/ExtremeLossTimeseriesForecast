# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:18:06 2024

@author: Xiao Xia Liang
"""
import argparse
import numpy as np
import os
import pandas as pd
from os.path import join
from scipy import signal
from data_scaler import standardize_df, log_df

def _get_leap_years(year_list):
    leaplist=[]
    for year in year_list:
        if ((year%4==0 and year%100!=0) or (year%400==0)):
            leaplist.append(year)
            
    return leaplist


def _generate_graph_seq2seq_io_data(df, seq_length_x, seq_length_y, shift, scaler=None):    
    """
    Generate samples from
    :param df:
    :param x_offsets: the x offsets are the x indices of the sequence.
    e.g. if seq_length_x is 3 then x_offsets = [-2,-1,0]
    
    :param y_offsets: the y offsets are the y indices of the sequence.
    e.g. if seq_length_y is 3 then x_offsets = [1, 2, 3]
   
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df, axis=-1)
    feature_list = [data]
    
    df_time = df.index

       
    years = df.index.values.astype('datetime64[Y]').astype(int) + 1970
    days = df.index.values.astype("datetime64[D]") - df.index.values.astype("datetime64[Y]")
    days = days.astype("int32")+1

    leap_years = _get_leap_years(np.unique(years))

    scaled_days = [] 
    
    for day, year in zip(days, years):
        if year in leap_years:
            scaled_days.append(day/366)
        else:
            scaled_days.append(day/365)
            
    time_in_year = np.tile(scaled_days, [1, num_nodes, 1]).transpose((2, 1, 0))
    feature_list.append(time_in_year)
    
    data = np.concatenate(feature_list, axis=-1)
    x, y, time_list = [], [], []
    #shift = seq_length_x
    max_t = num_samples - (seq_length_y+shift)
   
    for t in range(0, max_t, shift):  # t is the index of the last observation.
        
        total_window_len = data[t:seq_length_x+seq_length_y+t]
        x.append(total_window_len[:seq_length_x, ...])
        y.append(total_window_len[seq_length_x:, ...]) 
        
        time_window_len = df_time[t:seq_length_x+seq_length_y+t]
        time_list.append(time_window_len[seq_length_x:]) # Only collecting the time for y dataset
        
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    time = np.stack(time_list, axis=0)
    
    
    return x, y, time

def generate_train_val_test(args):
    """

    Parameters
    ----------
    args : From parser
       
    Returns
    -------
    x_train : nummpy
        Used for model training.
    y_train : numpy
        Used for model training.
    x_val : numpy
        Used for model validation.
    y_val : numpy
        Used for model validation.
    x_test : numpy
        Used for model testing.
    y_test : numpy
        Used for model testing.
    """
    
    seq_length_x, seq_length_y, shift = args.seq_length_x, args.seq_length_y, args.shift
    # df = pd.read_csv(args.df_train, index_col=0, parse_dates=True)
    # df_test = pd.read_csv(args.df_test, index_col=0, parse_dates=True)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # print('x offset ' + str(x_offsets))
    # Predict the time steps
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    # print('y offset ' + str(y_offsets))

    print(x_offsets)
    print(y_offsets)
    
    
    if args.scale_data == True:
        df = standardize_df(f"{args.data_path}/{args.df_path}", loader="train")
        df_test = standardize_df(f"{args.data_path}/{args.df_path}", loader="test")
        
        #df = log_df(f"{args.data_path}/{args.df_path}", loader="train")
        #df_test = log_df(f"{args.data_path}/{args.df_path}", loader="test")
        
    else:
        df = pd.read_csv(f"{args.data_path}/{args.df_train}", index_col=0, parse_dates=True)
        df_test = pd.read_csv(f"{args.data_path}/{args.df_test}", index_col=0, parse_dates=True)
    
    x, y, time = _generate_graph_seq2seq_io_data(
        df,
        seq_length_x,
        seq_length_y,
        shift
    )

    x_test, y_test, time_test = _generate_graph_seq2seq_io_data(
        df_test,
        seq_length_x,
        seq_length_y,
        shift
    )    
    
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    print("x_test shape: ", x_test.shape, ", y_test shape: ", y_test.shape)
    
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_split)
    num_val = num_samples - num_train
    
    x_train, y_train, time_train = x[:num_train], y[:num_train], time[:num_train]
   
    x_val, y_val,time_val = x[num_train:], y[num_train:], time[num_train:]
    
    # This is the data format for training and testing the Graph WaveNet model
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(f"{args.data_path}/{args.output_dir}", f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    
    # Write data for reconstruction of data after model training 
    # These files are for plotting and debugging
    np.save(join( f"{args.data_path}/{args.output_dir}", "x_train.npy"), x_train)
    np.save(join( f"{args.data_path}/{args.output_dir}", "y_train.npy"), y_train)
    np.save(join( f"{args.data_path}/{args.output_dir}", "x_val.npy"), x_val)
    np.save(join( f"{args.data_path}/{args.output_dir}", "y_val.npy"), y_val)
    np.save(join( f"{args.data_path}/{args.output_dir}", "x_test.npy"), x_test)
    np.save(join( f"{args.data_path}/{args.output_dir}", "y_test.npy"), y_test)
    
    np.save(join(f"{args.data_path}/{args.output_dir}", "train_time.npy"), time_train)
    np.save(join(f"{args.data_path}/{args.output_dir}", "val_time.npy"), time_val)
    np.save(join(f"{args.data_path}/{args.output_dir}", "test_time.npy"), time_test)
    
    print("Data files are saved in the output directory")
    
    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    
    forecast = 12 # Forecasting time step
    freq = "4H"     # Frequency [H, 4H, D]
    dataset = "milandre_data" ### [milandre_data, yamaska_data]
    # dataset = "yamaska_data"
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--output_dir", type=str, default="data_"+freq+"/GWN_"+str(forecast), help="Output directory.")
    
    parser.add_argument("--df_path", type=str, default="data_"+freq, help="df path",)
    # parser.add_argument("--time_dir", type=str, default="data_H/GWN_12", help="Time directory")
    
    parser.add_argument("--df_train", type=str, default="data_"+freq+"/train_val_data.csv", help="df for train and validation.",)
    parser.add_argument("--df_test", type=str, default="data_"+freq+"/test_data.csv", help="df for testing.",)
    parser.add_argument("--seq_length_x", type=int, default=forecast, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=forecast, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=forecast, help="Default is seq_length_x", ) # this is a sequence window shift
    parser.add_argument("--train_split", type=float, default=0.8, help="The percentage split for training and validation",)
    parser.add_argument("--scale_data", type=bool, default=True, help="Data scaling",)
    parser.add_argument("--data_path", type=str, default="/Users/xl3138/workspaces/extreme_loss/gwn/"+dataset, help="Data dir")
    
    args = parser.parse_args()
    if not os.path.exists(f"{args.data_path}/{args.output_dir}"):
        os.mkdir(f"{args.data_path}/{args.output_dir}")
        
    x_train, y_train, x_val, y_val, x_test, y_test = generate_train_val_test(args)
    # generate_train_val_test(args)
    
    # np.save(r"/Users/xl3138/workspaces/extreme_loss/gwn/milandre_data/data_4H/x_train_4h.npy", x_train)
