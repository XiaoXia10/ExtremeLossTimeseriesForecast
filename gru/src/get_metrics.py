# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:55:42 2025

@author: Xiao Xia Liang
"""

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from os.path import join, dirname, abspath
import argparse
from data_scaler import destandardize_pred
import numpy as np
import pickle
from permetrics.regression import RegressionMetric

# Repo-relative default paths: <repo>/gru/src/this_file.py -> base_dir = <repo>/gru
current_dir = dirname(abspath(__file__))
base_dir = dirname(current_dir)


def main(args):

    loader = args.loader

    ypred, ytrue = destandardize_pred(args.data_path, f'{args.data_path}/{args.save_dir}', f'{args.data_path}/{args.data_dir}', args.loader)
    ####### Evaluate each well seperately #########
    list_metrics = ["RMSE", "MAE", "MSE", "MAPE", "NSE", "KGE"]
    evaluator = RegressionMetric(ytrue.to_numpy(), ypred.to_numpy())
    results = evaluator.get_metrics_by_list_names(list_metrics)
    
    ######## Save results into a file ########
    
    with open(join(f'{args.data_path}/{args.data_dir}', args.metric_json), 'wb') as file:
        pickle.dump(results, file)
        
    f = open(join(f'{args.data_path}/{args.data_dir}', args.metric_csv), 'w')
    # units of the data: Milandre = spring discharge (l/s), Yamaska = groundwater level (m)
    unit = "m" if "yamaska_data" in str(args.data_path) else "l/s"
    f.write(f"RMSE ({unit}), MAE ({unit}), MSE ({unit})^2, MAPE, NSE, KGE \n")
    
    ###### metric mean of all wells
    f.write(",".join(str(np.mean(results[metric])) for metric in list_metrics) + "\n")
    f.close()
        
    # return results

if __name__ == "__main__":
    
    loss ="extreme" #[mae, extreme, gumbel, dense, pp, focal]
    
    dataset = "milandre_data" #[milandre_data, yamaska_data]
    freq = "4H" 
    seq_length_x, seq_length_y, shift  = 12, 12, 12
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--alpha", type=float, default=2.0, help="the alpha value used for extreme loss function",) 
    
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    
    parser.add_argument("--data_dir", type=str, default="GRU"+str(seq_length_x)+str(seq_length_y)+str(shift), help="df for testing.",)
    parser.add_argument("--save_dir", type=str, default="GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"/experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_"+loss, help="Save Path")
 
    parser.add_argument("--loader", type=str, default="test", help="Which dataset to loader, loader = train, val, test",)
    
    parser.add_argument("--metric_csv", type=str, default="GRU_metrics_"+str(seq_length_x)+"_"+str(freq)+"_"+loss+".csv", help="metric file name",)
    parser.add_argument("--metric_json", type=str, default="GRU_metrics_"+str(seq_length_x)+"_"+str(freq)+"_"+loss+".json", help="metric file name",)
    
    parser.add_argument("--data_path", type=str, default=join(base_dir, dataset, "data_"+freq), help="Data path: <base_dir>/<dataset>/data_<freq>")
    args = parser.parse_args()

        
    main(args)


