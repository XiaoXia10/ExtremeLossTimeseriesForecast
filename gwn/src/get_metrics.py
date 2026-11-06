#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 17:47:12 2026

@author: xl3138
"""

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from os.path import join, dirname, abspath
import argparse
from data_scaler import destandardize_pred
import numpy as np
import pickle
from permetrics.regression import RegressionMetric

# Repo-relative default paths: <repo>/gwn/src/this_file.py -> base_dir = <repo>/gwn
current_dir = dirname(abspath(__file__))
base_dir = dirname(current_dir)

def get_metrics(args):

    loader = args.loader

    
    # yhat, realy = destandardize_pred(args.df_dir, f'{args.df_dir}/{args.pred_dir}', f'{args.df_dir}/{args.data_dir}', args.loader)
    # yhat, realy = destandardize_pred(args.df_path, args.save_dir, args.data_dir, args.loader)
    # ypred, ytrue = destandardize_pred(args.data_path, f'{args.data_path}/{args.save_dir}', f'{args.data_path}/{args.data_dir}', args.loader)
    
    
    ypred, ytrue = destandardize_pred(args.data_path, f"{args.data_path}/{args.data_dir}", f"{args.data_path}/{args.time_dir}", args.loader)
    ####### Evaluate each well seperately #########
    list_metrics = ["RMSE", "MAE", "MSE", "MAPE", "NSE", "KGE"]
    evaluator = RegressionMetric(ytrue.to_numpy(), ypred.to_numpy())
    results = evaluator.get_metrics_by_list_names(list_metrics)
    
    ######## Save results into a file ########
    ######## Beware!!!! This is a pickle file, can only be open with the same environment
    
    with open(join(f'{args.data_path}/{args.data_dir}', args.metric_json), 'wb') as file:
        pickle.dump(results, file)
        
    f = open(join(f'{args.data_path}/{args.data_dir}', args.metric_csv), 'w')
    # units of the data: Milandre = spring discharge (l/s), Yamaska = groundwater level (m)
    unit = "m" if "yamaska_data" in str(args.data_path) else "l/s"
    f.write(f"RMSE ({unit}), MAE ({unit}), MSE ({unit})^2, MAPE, NSE, KGE \n")
    
    ###### metric mean of all wells
    f.write(",".join(str(np.mean(results[metric])) for metric in list_metrics) + "\n")
    f.close()
        
    return results

if __name__ == "__main__":
    
    loss = "extreme" #[mae, extreme, focal, pp, dense, gumbel]
    # loss = "extreme"
    forecast = 12
    freq = "4H"
    dataset = "milandre_data"
    # dataset = "yamaska_data"
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--forecast", type=int, default=forecast, help="metric file name",)
    parser.add_argument("--alpha", type=float, default=2.0, help="The alpha value used for the extreme loss function",)
    
    # parser.add_argument("--df_path", type=str, default="data_"+freq, help="df path",)
    parser.add_argument("--data_dir", type=str, default="GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_"+loss, help="Data directory")
    parser.add_argument("--time_dir", type=str, default="GWN_"+str(forecast), help="Time directory")

    # parser.add_argument("--metric_name", type=str, default="GWN_model_metrics.csv", help="metric file name",)
    parser.add_argument("--metric_csv", type=str, default="GWN_metrics_"+str(forecast)+"_"+str(freq)+"_"+loss+".csv", help="metric file name",)
    parser.add_argument("--metric_json", type=str, default="GWN_metrics_"+str(forecast)+"_"+str(freq)+"_"+loss+".json", help="metric file name",)
    # parser.add_argument("--metric_json_out", type=str, default="GWN_metrics_"+str(forecast)+"_"+str(freq)+"_"+loss+"_out.json", help="metric file name",)
    
    parser.add_argument("--loader", type=str, default="test", help="Which dataset to loader, loader = train, val, test",)
    
    parser.add_argument("--data_path", type=str, default=join(base_dir, dataset, "data_"+freq), help="Data path: <base_dir>/<dataset>/data_<freq>")
    
    
    args = parser.parse_args()
    results = get_metrics(args)