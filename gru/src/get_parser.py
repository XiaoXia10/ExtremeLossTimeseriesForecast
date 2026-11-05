# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:52:34 2024

@author: Xiao Xia Liang
"""
import argparse
from os.path import join, dirname, abspath

# Repo-relative default paths: <repo>/gru/src/this_file.py -> base_dir = <repo>/gru
current_dir = dirname(abspath(__file__))
base_dir = dirname(current_dir)

def get_shared_arg_parser():
    
    freq = "4H" #[H, 4H, D]
    dataset = "milandre_data" #[milandre_data, yamaska_data]
    
    seq_length_x = 12
    seq_length_y = 12
    shift = 12
    
    loss = 'extreme' #[mae, extreme, gumbel, dense, pp, focal]
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--list_names", type=list, default=["Milandrine","Bâme","Saivu","Font" ], help="List of names for measuring stations",) #Keep double quotes or s**t
    
    parser.add_argument('--loss_function', type=str, default=loss, help='Which loss function to call')
        
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') 
    parser.add_argument('--latent_dim', type=int, default=120, help='Latent dimension') 
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout') 
    parser.add_argument('--recurrent_dropout', type=float, default=0.7, help='Recurrent dropout') 
        
    
    # parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--patience', type=int, default=10, help='quit if no improvement after this many iterations')

    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    
    parser.add_argument("--data_dir", type=str, default="GRU"+str(seq_length_x)+str(seq_length_y)+str(shift), help="df for testing.",)
    parser.add_argument("--save_dir", type=str, default= "GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"/experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_"+loss, help="Save Path")
    parser.add_argument("--save_model_name", type=str, default="GRU_best_model.h5", help="Best saved model name")
   
    parser.add_argument("--data_path", type=str, default=join(base_dir, dataset, "data_"+freq), help="Data path: <base_dir>/<dataset>/data_<freq>")
    
    return parser
