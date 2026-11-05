# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:56:57 2024

@author: Xiao Xia Liang
"""
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed, GRU
from tensorflow.keras import initializers, callbacks
import numpy as np
import torch
import torch.nn.functional as F
from denseweight import DenseWeight
import pandas as pd
import os


tf.config.run_functions_eagerly(True)


def extreme_value_loss(y_true, y_pred, alpha=2.0):
    """
    Extreme Value Loss Function, focusing on extreme/outlier events in predictions.
    
    Parameters:
    y_true (tensor): True values.
    y_pred (tensor): Predicted values.
    alpha (float): Coefficient for standard deviation to define outliers.
    beta (float): Weight multiplier for high extreme values.
    gamma (float): Weight multiplier for low extreme values.
    
    Returns:
    loss (tensor): Computed loss value focusing on outlier events.
    """
    
    # Standard deviation and mean based on true values
    mean, variance = tf.nn.moments(y_true, axes=[0,1,2])
    std_dev = tf.sqrt(variance)
    max_value = tf.reduce_max(tf.abs(y_true))
    
    # Define extremes: values beyond 'alpha' standard deviations from the mean
    extreme_mask_pos = tf.abs(y_true - mean) > alpha * std_dev
    extreme_mask_neg = tf.abs(y_true - mean) < alpha * std_dev

    # Calculate absolute errors
    errors = tf.abs(y_true - y_pred)
    # errors = tf.sqrt(tf.square(y_true - y_pred))
    
    
    dat_min = tf.reduce_min(y_true)
    dat_max = tf.reduce_max(y_true)
    
    beta = tf.abs(dat_max-mean)/std_dev
    gamma = tf.abs(mean-dat_min)/std_dev
    
    # Apply weights to errors based on whether the data is an outlier or not
    weights= tf.where(extreme_mask_pos, beta, 1.0) * tf.where(extreme_mask_neg, gamma, 1.0)
    # weights = tf.where(extreme_mask_pos, beta, tf.where(extreme_mask_neg, gamma, 1.0))
    
    weighted_errors = weights * errors

    # Mean error to form the loss
    loss = tf.reduce_mean(weighted_errors)
    
    return loss

def gumbel_loss(y_true, y_pred, gamma=1.0):
    # Small values of gamma can lead to instability, while larger values make the loss behave more like MSE. 
    mse = tf.square(y_pred - y_true) # check the mse function 
   
    weights = ((1-tf.exp(-mse))**gamma)
    
    gumbel = tf.reduce_mean(weights*mse)
    
    return gumbel


def focal_r_loss(y_true, y_pred, activate='sigmoid', beta=.2, gamma=1):
    
    loss = tf.abs(y_pred-y_true)
    
    loss *= (tf.tanh(beta * tf.abs(y_pred - y_true))) ** gamma if activate == 'tanh' else \
        (2 * tf.sigmoid(beta * tf.abs(y_pred - y_true)) - 1) ** gamma
        
    focal = tf.reduce_mean(loss)
    
    return focal


############# Dense Loss Function #################

def dense_loss(y_true, y_pred, gamma=1.0):
    ### gamma = 0.0 -> uniform weighting; larger alpha -> more emphasis on rare samples.
    
    dw = DenseWeight(gamma)
    
    orig_shape = tf.shape(y_true)               
    y_flat = tf.reshape(y_true, [-1]) 
    
    weights = dw.fit(y_flat.numpy())
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    
    weights = tf.reshape(weights, orig_shape)
    
    mse = tf.square(y_pred - y_true) 

    dense = tf.reduce_mean(weights*mse)
    
 
    return dense

############# Plotting Position Function #################
def _get_pp_ranking(df_path):
    
    df = pd.read_csv(df_path, index_col=0, parse_dates=True)
    df_std = (df-df.mean())/df.std()
    
    all_values = pd.concat([df_std[col] for col in df_std])
    
    all_values_floored = np.floor(all_values)
    
    series_df = pd.DataFrame({'values': all_values, 'bins': all_values_floored})
    
    bins = series_df['bins'].value_counts()
    
    if not os.path.exists(r"pp_ranks"):
        os.mkdir(r"pp_ranks")
        
    bins.to_csv(r"pp_ranks/bins.csv")
    
    # return bins

    
def pp_loss(y_true, y_pred):
    # 1. get unique values in the TS
    # 2. get ranking using the ranking r function with the unique values 
    # 3. calculate the PP weights
    # 4. calculate the PP loss
    
    bins_df = pd.read_csv(r"pp_ranks/bins.csv", index_col=0)

    ind = tf.cast(bins_df.index.values, tf.int32)          # (11,)
    bins = tf.cast(bins_df.values.squeeze(), tf.float32)  # (11,)

    n = tf.reduce_sum(bins)

    y_true_floored = tf.cast(tf.floor(y_true), tf.int32)  # (B,X,Y)
    orig_shape = tf.shape(y_true_floored)

    y_flat = tf.reshape(y_true_floored, [-1])             # (N,)

    matches = tf.equal(
        tf.expand_dims(y_flat, axis=-1),                   # (N,1)
        tf.expand_dims(ind, axis=0)                         # (1,11)
    )                                                       # (N,11)

    exists = tf.reduce_any(matches, axis=-1)
    indices = tf.where(exists, tf.argmax(matches, axis=-1), -1)

    # indices = tf.argmax(matches, axis=-1)                  # (N,)

    # --- LOOKUP ---
    ranks_flat = tf.gather(bins, indices)                  # (N,)
    ranks = tf.reshape(ranks_flat, orig_shape)             # (B,X,Y)


    # --- PP LOSS ---
    pp = ranks / (n + 1.0)
    pp_hazen = tf.square(pp - 0.5)
    pp_weights = pp_hazen / tf.reduce_mean(pp_hazen)

    mse = tf.square(y_pred - y_true)
    loss = tf.reduce_mean(pp_weights * mse)

    return loss
 
#%%

def auto_encoder_gru(train_x, train_y, args):

    # Define the encoder
    encoder_inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    encoder = GRU(args.latent_dim,
                    dropout=args.dropout, 
                    recurrent_dropout=args.recurrent_dropout, 
                    return_state=True)

    encoder_outputs, state_h = encoder(encoder_inputs)

    # Define the decoder
    decoder_inputs = RepeatVector(train_y.shape[1])(encoder_outputs)
    decoder_lstm = GRU(args.latent_dim, 
                        dropout=args.dropout, 
                        recurrent_dropout=args.recurrent_dropout, 
                        return_sequences=True, 
                        return_state=False)

    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=state_h)
    decoder_dense = TimeDistributed(Dense(train_y.shape[2], activation='linear'))
    
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the encoder-decoder model
    model = Model(encoder_inputs, decoder_outputs)
    
    adam_optimizer = Adam(learning_rate=args.learning_rate)
    

    if args.loss_function == "mae":
        print("MAE Loss function is used")
        model.compile(optimizer=adam_optimizer, loss="mae")
        
    elif args.loss_function == "extreme":
        print("Extreme Loss function is used")
        model.compile(optimizer=adam_optimizer, loss=extreme_value_loss)
    
    elif args.loss_function == "pp":
        print("pp Loss function is used")
        model.compile(optimizer=adam_optimizer, loss=pp_loss)
    
    elif args.loss_function == "dense":
        print("Dense Loss function is used")
        model.compile(optimizer=adam_optimizer, loss=dense_loss)
    
    elif args.loss_function == "focal":
        print("Focal R Loss function is used")
        model.compile(optimizer=adam_optimizer, loss=focal_r_loss)
        
    elif args.loss_function == "gumbel":
        print("Gumbel Loss function is used")
        model.compile(optimizer=adam_optimizer, loss=gumbel_loss)
        
    return model
