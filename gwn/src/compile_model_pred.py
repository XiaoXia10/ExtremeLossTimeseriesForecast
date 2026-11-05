# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:37:18 2024

@author: Xiao Xia Liang
"""

import util
from model import *
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
from os.path import join
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

def main(args, 
         loader='test', 
         save_pred='predy.csv', 
         save_real='realy.csv',
         **model_kwargs):
         
    # device = torch.device(args.device)
    device = util.default_device()
    adjinit, supports = util.make_graph_inputs(args, device)
    
    # Create model 
    model = GWNet.from_args(args, device, supports, adjinit, **model_kwargs)
    model.to(device)
    
    # Load best trained model
    model.load_state_dict(torch.load(join(f"{args.data_path}/{args.save}", 'best_model.pth')))
    model.eval()
    print('model loaded successfully')
    
    data = util.load_dataset(f"{args.data_path}/{args.data}", 
                             args.batch_size, 
                             args.batch_size, 
                             args.batch_size, 
                             n_obs=args.n_obs, 
                             fill_zeroes=args.fill_zeroes
                             )
    
    scaler = data['scaler']
    realy = torch.Tensor(data[f'y_{loader}']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    met_df, yhat = util.calc_tstep_metrics(model, device, data[f'{loader}_loader'], scaler, realy, args.seq_length, args.loss_function, alpha=args.alpha)
    df_pred, df_real = util.make_pred_df_wells(realy, yhat, scaler, args.seq_length, args.shift) # Compile and save the GWN prediction data into a dataframe
   
    # met_df.to_csv(join(save_path, "last_test_metrics.csv"))
    df_pred.to_csv(join(f"{args.data_path}/{args.save}", loader+"_predy.csv"), index=False)
    df_real.to_csv(join(f"{args.data_path}/{args.save}", loader+"_realy.csv"), index=False)

    # return df_pred, df_real 


if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    args = parser.parse_args()
    # save_path = args.save
    
    list_loader = ["train", "val", "test"] 
    for loader in list_loader:
        # df_pred, df_real = main(args, loader)
        main(args, loader)

