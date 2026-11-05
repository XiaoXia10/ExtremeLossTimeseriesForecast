# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:39:16 2024

@author: Xiao Xia Liang
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from model import auto_encoder_gru, _get_pp_ranking 
import pickle
import argparse
from get_parser import get_shared_arg_parser
import compile_model_pred_data
import tensorflow as tf

def train(args, **model_kwargs):
    
    train_x = np.load(join(f"{args.data_path}/{args.data_dir}", "x_train.npy"))
    train_y = np.load(join(f"{args.data_path}/{args.data_dir}", "y_train.npy"))
    val_x = np.load(join(f"{args.data_path}/{args.data_dir}", "x_val.npy"))
    val_y = np.load(join(f"{args.data_path}/{args.data_dir}", "y_val.npy"))
    
    if args.loss_function == "pp":
        _get_pp_ranking(join(args.data_path, 'df.csv'))
        

    model = auto_encoder_gru(train_x, train_y, args)
    
    model.summary()
        
    # Save the best trained model with the min val_error
    save_path = join(f"{args.data_path}/{args.save_dir}", args.save_model_name)
    

    # Regulate the model
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                             min_delta=0, 
                                             patience=args.patience, 
                                             verbose=1, 
                                             mode='min')

    model_checkpoint =  callbacks.ModelCheckpoint(filepath=save_path,   
                                                  monitor='val_loss', 
                                                  save_best_only=True, 
                                                  verbose=1)

    list_callback = [early_stopping, model_checkpoint]

    # Train the model
    history = model.fit(x=train_x, 
                        y=train_y, 
                        epochs=args.epochs, 
                        batch_size= args.batch_size, 
                        validation_data=(val_x, val_y),
                        callbacks=list_callback,
                        verbose=2, 
                        shuffle=False)

    # plot history
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss', fontsize= 15)
    plt.ylabel('Loss', fontsize= 15)
    plt.xlabel('Epoch', fontsize= 15)
    plt.legend()
    plt.savefig(join(f"{args.data_path}/{args.save_dir}", "loss_history.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return history

if __name__ == "__main__":
    
    parser = get_shared_arg_parser()
    args = parser.parse_args()
        
    os.makedirs(f"{args.data_path}/{args.save_dir}", exist_ok=True)
    with open(f"{args.data_path}/{args.save_dir}/args.pkl", "wb") as fh:
        pickle.dump(args, fh)
    
    train(args) # Train model

    compile_model_pred_data.main(args)  # Compile the predicted data from trained model


    