import argparse
import pickle
import numpy as np
import os
from os.path import join, dirname, abspath

import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from scipy.sparse import linalg
from denseweight import DenseWeight

# Repo-relative default paths: <repo>/gwn/src/this_file.py -> base_dir = <repo>/gwn
current_dir = dirname(abspath(__file__))
base_dir = dirname(current_dir)

# DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():

    def __init__(self, mean, std, fill_zeroes=True):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def default_device():
    
    # processor_info = platform.processor()
    # print(f"Processor Name: {processor_info}")
    
    # list_device = ["cpu", "gpu", "mps", "xpu", "xla", "meta", 'cuda']

    if torch.backends.mps.is_available():
        #print("mps")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        #print("cuda")
        device = torch.device("cuda")
    else:
        #print("cpu")
        device = torch.device("cpu")

    return device
    

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(file_path):
    adj_mx = np.load(file_path)
    return adj_mx


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, n_obs=None, fill_zeroes=True):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        if n_obs is not None:
            data['x_' + category] = data['x_' + category][:n_obs]
            data['y_' + category] = data['y_' + category][:n_obs]
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std(), fill_zeroes=fill_zeroes)
    # Data format
    for category in ['train', 'val', 'test']:
        
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = scaler
    return data

############# Extreme Loss Function ###############
def extreme_value_loss(y_true, y_pred, alpha=1.5):
    """
    Extreme Value Loss Function, focusing on extreme events in predictions in PyTorch.
    
    Parameters:
    y_true (tensor): True values.
    y_pred (tensor): Predicted values.
    alpha (float): Coefficient for standard deviation to define extremes. This is a parameters that needs to be tuned
    recommend to be 2 or greater. (> than 2 standard deviation from the mean)
    beta (float): Weight multiplier for extreme values.
    
    Returns:
    loss (tensor): Computed loss value focusing on extreme events.
    """
    # print("Using Extreme loss")
    
    # Calculate mean and standard deviation based on true values
    mean = torch.mean(y_true) 
    std_dev = torch.std(y_true)
    dat_min = torch.min(y_true)
    dat_max = torch.max(y_true)
    

    beta = (dat_max-mean)/std_dev
    gamma = torch.abs(mean-dat_min)/std_dev
    
    # Define extremes: values beyond 'alpha' standard deviations from the mean
    extreme_mask_pos = torch.abs(y_true - mean) > alpha * std_dev
    extreme_mask_neg = torch.abs(y_true - mean) < alpha * std_dev

    
    # Calculate absolute errors
    errors = torch.abs(y_true - y_pred)

    # Apply different weights to the errors based on whether they are extremes
    weights= torch.where(extreme_mask_pos, beta * torch.ones_like(errors), torch.ones_like(errors))*torch.where(extreme_mask_neg, gamma * torch.ones_like(errors), torch.ones_like(errors))
    
    weighted_errors = weights * errors

    # Mean error to form the losss
    extreme = torch.mean(weighted_errors)
    
    return extreme

############# Gumbel Loss Function ###############

def gumbel_loss(y_true, y_pred, gamma=1.0):
    # Small values of gamma can lead to instability, while larger values make the loss behave more like MSE. 
    
    # print("Using Gumbel loss")
    mse = (y_pred - y_true).pow(2)

    weights = (1.0 - torch.exp(-mse)).pow(gamma)

    gumbel = (weights * mse).mean()
    
    return gumbel

############# Focal-R Loss Function ###############

# from focal_loss import SparseCategoricalFocalLoss

# def focal_r_loss(y_true, y_pred, beta, gamma):

#     focal_loss_fn = SparseCategoricalFocalLoss(gamma, beta)
#     focal_r = focal_loss_fn(y_true, y_pred) 
    
#     return focal_r


def focal_r_loss(y_pred, y_true, weights=None, activate='sigmoid', beta=.2, gamma=1):
    
    # print("Using Focal R loss")
    loss = F.l1_loss(y_pred, y_true, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(y_pred - y_true))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(y_pred - y_true)) - 1) ** gamma
    
    if weights is not None:
        loss *= weights.expand_as(loss)
        
    focal = torch.mean(loss)
    
    return focal


############# Dense Loss Function #################

def dense_loss(y_true, y_pred, gamma=1.0):
    ### gamma = 0.0 -> uniform weighting; larger alpha -> more emphasis on rare samples.

    # print("Using Dense loss")

    
    device = y_true.device
    dtype = y_true.dtype

    dw = DenseWeight(gamma)

    orig_shape = y_true.shape

    # Flatten
    y_flat = y_true.reshape(-1)

    # NumPy-based weight fitting (non-differentiable by design)
    weights_np = dw.fit(y_flat.detach().cpu().numpy())

    # Back to torch
    weights = torch.tensor(weights_np, device=device, dtype=dtype)
    weights = weights.reshape(orig_shape)

    # MSE
    mse = (y_pred - y_true).pow(2)

    dense = (weights * mse).mean()
    
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
    
    # print("Using PP loss")
    
    device = y_true.device
    dtype = y_true.dtype

    # load bins
    bins_df = pd.read_csv("pp_ranks/bins.csv", index_col=0)

    ind = torch.tensor(bins_df.index.values, device=device, dtype=torch.int32)   
    bins = torch.tensor(
        bins_df.values.squeeze(),
        device=device,
        dtype=dtype,
    )                                                                           

    n = bins.sum()

    # floor and flatten tensor
    y_true_floored = torch.floor(y_true).to(torch.int32)  
    orig_shape = y_true_floored.shape

    y_flat = y_true_floored.reshape(-1)                   

    # match the floor values to true values
    matches = y_flat.unsqueeze(-1) == ind.unsqueeze(0)    

    exists = matches.any(dim=-1)
    indices = torch.where(
        exists,
        matches.argmax(dim=-1),
        torch.full_like(y_flat, -1),
    )                                                      

    # lookup
    ranks_flat = torch.where(
        indices >= 0,
        bins[indices.clamp(min=0)],
        torch.zeros_like(y_flat, dtype=dtype),
    )                                                    

    ranks = ranks_flat.reshape(orig_shape)                 

    # --- PP LOSS ---
    pp = ranks / (n + 1.0)
    pp_hazen = (pp - 0.5).pow(2)
    pp_weights = pp_hazen / pp_hazen.mean()

    mse = (y_pred - y_true).pow(2)
    pp_loss = (pp_weights * mse).mean()

    return pp_loss

# def calc_metrics_extreme(preds, labels, null_val=0.):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     # print(preds.shape)
#     # print(labels.shape)
#     mse = (preds - labels) ** 2
#     mae = torch.abs(preds-labels)
#     mape = mae / labels
#     extreme_loss = extreme_value_loss(labels, preds)
    
#     extreme_loss, mape, mse= [mask_and_fillna(l, mask) for l in [extreme_loss, mape, mse]]
#     rmse = torch.sqrt(mse)
#     return extreme_loss, mape, rmse

    
# def calc_metrics(preds, labels, null_val=0.0):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     # print(preds.shape)
#     # print(labels.shape)
#     mse = (preds - labels) ** 2
#     mae = torch.abs(preds-labels)
#     mape = mae / labels
    
#     mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
#     rmse = torch.sqrt(mse)
#     return mae, mape, rmse


def calc_metrics(y_pred, y_true, loss_fun, null_val=np.nan, alpha=1.5):
    
# def calc_metrics(y_pred, y_true, loss_fun, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(y_true)
    else:
        mask = (y_true != null_val)
    
    mask = (y_true != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    mse = (y_pred - y_true) ** 2

    mae = torch.abs(y_pred-y_true)
    mape = torch.abs(mae/y_true)
    
    if loss_fun == "mse":
        loss, mape, mse = [mask_and_fillna(l, mask) for l in [mse, mape, mse]]
        rmse = torch.sqrt(mse)
        
    if loss_fun == "mae":
        
        mae_loss = torch.abs(y_pred-y_true)
        loss, mape, mse = [mask_and_fillna(l, mask) for l in [mae_loss, mape, mse]]
        rmse = torch.sqrt(mse)
        
    if loss_fun =="extreme":
        extreme = extreme_value_loss(y_true, y_pred, alpha=alpha)
        loss, mape, mse= [mask_and_fillna(l, mask) for l in [extreme, mape, mse]]
        rmse = torch.sqrt(mse)
        
    if loss_fun =="dense":
        dense = dense_loss(y_true, y_pred)
        loss, mape, mse= [mask_and_fillna(l, mask) for l in [dense, mape, mse]]
        rmse = torch.sqrt(mse)
        
    if loss_fun =="gumbel":
        gumbel = gumbel_loss(y_true, y_pred)
        loss, mape, mse= [mask_and_fillna(l, mask) for l in [gumbel, mape, mse]]
        rmse = torch.sqrt(mse)
        
    if loss_fun =="focal":
        focal = focal_r_loss(y_true, y_pred)
        loss, mape, mse= [mask_and_fillna(l, mask) for l in [focal, mape, mse]]
        rmse = torch.sqrt(mse)
        
    if loss_fun =="pp":
        pp = pp_loss(y_true, y_pred)
        loss, mape, mse= [mask_and_fillna(l, mask) for l in [pp, mape, mse]]
        rmse = torch.sqrt(mse)
    
    return loss, mape, rmse


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_tstep_metrics(model, device, test_loader, scaler, realy, seq_length, loss_function, alpha=1.5) -> pd.DataFrame:
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    for i in range(seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        pred = torch.clamp(pred, min=0., max=70.)
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real, loss_function, alpha=alpha)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    
    
    return test_met_df, yhat


def _to_ser(arr):
    return pd.DataFrame(arr.cpu().detach().numpy()).stack().rename_axis(['obs', 'well num'])


def make_pred_df_GWN(realy, yhat, scaler, seq_length):
    df = pd.DataFrame(dict(y_last=_to_ser(realy[:, :, seq_length - 1]),
                           yhat_last=_to_ser(scaler.inverse_transform(yhat[:, :, seq_length - 1])),
                           y_3=_to_ser(realy[:, :, 2]),
                           yhat_3=_to_ser(scaler.inverse_transform(yhat[:, :, 2]))))
    return df

def make_pred_df_wells(realy, yhat, scaler, seq_length, shift):
    
    if seq_length==shift:
        print('shift length matches seq_length')
        temp = torch.transpose(yhat, 1, 2 ).cpu().detach().numpy()
        temp = temp.reshape(temp.shape[0]*temp.shape[1], temp.shape[2])
        df_yhat = pd.DataFrame(scaler.inverse_transform(temp))
    
        temp = torch.transpose(realy, 1, 2 ).cpu().detach().numpy()
        temp = temp.reshape(temp.shape[0]*temp.shape[1], temp.shape[2])
        df_realy = pd.DataFrame(temp)
        
        return df_yhat, df_realy
    
    else:
        print('shift length does not match seq_length')
        realy = torch.transpose(realy, 1, 2 ).cpu().detach().numpy()
        yhat = torch.transpose(yhat, 1, 2 ).cpu().detach().numpy()
        yhat = scaler.inverse_transform(yhat)
        
        return yhat, realy

def make_graph_inputs(args, device):

    adj_mx = load_adj(f"{args.data_path}/{args.adjdata}")
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    aptinit = None if args.randomadj else supports[0]  # ignored without do_graph_conv and add_apt_adj
    if args.aptonly:
        if not args.addaptadj and args.do_graph_conv: raise ValueError(
            'WARNING: not using adjacency matrix')
        supports = None
    return aptinit, supports


def get_shared_arg_parser():
    
    freq = "4H" #(D for yamaska_data, [H, 4H] for milandre_data)
    forecast = 12
    
    loss = "extreme" # [mse, mae, extreme, focal, pp, dense, gumbel]
    
    dataset = "milandre_data"
    # dataset = "yamaska_data"
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--loss_function', type=str, default=loss, help=' Use strings [mae, extreme, focal, pp, dense, gumbel]')
    parser.add_argument('--alpha', type=float, default=1.5, help='Extreme loss threshold, in standard deviations from the mean')
    # parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--adjdata', type=str, default='adj_mat/adj_mx.npy', help='adj matrix path')
    # parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
    parser.add_argument('--do_graph_conv', action='store_true', help='whether to add graph convolution layer')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
    
    parser.add_argument('--data', type=str, default="data_"+freq+"/GWN_"+str(forecast), help='data path')
    parser.add_argument('--save', type=str, default="data_"+freq+"/GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_"+loss, help='save path')
    parser.add_argument('--seq_length', type=int, default=forecast, help='') # need to change this
    parser.add_argument("--shift", type=int, default=forecast, help="Default is seq_length_x", ) # this is the window shift
    parser.add_argument('--nhid', type=int, default=forecast, help='Number of channels for internal conv') #need to be optimized
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension') # DONT touch this, this is the last dimension of the tensor which is fixed to be two.
    parser.add_argument('--num_nodes', type=int, default=4, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') 
    parser.add_argument('--dropout', type=float, default=0.7, help='dropout rate')
    parser.add_argument('--n_obs', default=None, help='Only use this many observations. For unit testing.')
    parser.add_argument('--apt_size', default=10, type=int)
    parser.add_argument('--cat_feat_gc', action='store_true')
    parser.add_argument('--fill_zeroes', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='Create checkpoints for model training')
    
    parser.add_argument("--df_path", type=str, default=join(base_dir, dataset, "data_"+freq, "df.csv"), help="df path: <base_dir>/<dataset>/data_<freq>/df.csv")
    
    parser.add_argument("--data_path", type=str, default=join(base_dir, dataset), help="Data path: <base_dir>/<dataset>")
    
    return parser





