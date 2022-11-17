import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
import wandb
import scipy.misc
import time as tm
from tqdm import tqdm
from utils import *
from model import *
from data import *
from args import Args
import create_graphs
from rdkit import Chem
from torchviz import make_dot, make_dot_from_trace
import time
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/grnn')
NUM_POS_ATOMS = 118
NUM_POS_EDGES = len(Chem.rdchem.BondType.names.keys())
MODELS = {0: "Atomic Number", 1: "Formal Charge", 2:"Chirality", 3:"Hybridization", 4: "Explicit H", 5: "Aromatic"}
def train_vae_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred,z_mu,z_lsgms = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = cross_entropy_weight(y_pred, y)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
        loss = loss_bce + loss_kl
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss_bce.data[0], loss_kl.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        # logging
        log_value('bce_loss_'+args.fname, loss_bce.data[0], epoch*args.batch_ratio+batch_idx)
        log_value('kl_loss_' +args.fname, loss_kl.data[0], epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_mean_'+args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_min_'+args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_max_'+args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_mean_'+args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_min_'+args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_max_'+args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)

        loss_sum += loss.data[0]
    return loss_sum/(batch_idx+1)

def test_vae_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time = 1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to(args.device)
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(args.device)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)


    return G_pred_list


def test_vae_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to(args.device)
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to(args.device), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(args.device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list



def train_mlp_epoch(epoch, args, rnn, output, node_models, feature_extractor, data_loader,
                    optimizer_rnn, optimizer_output,optimizer_node_models,
                    scheduler_rnn, scheduler_output, scheduler_node_models, val_loader):
    rnn.train()
    output.train()
    for node_model in node_models:
        node_model.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        optimizer_output.zero_grad()
        for optimizer_node_model in optimizer_node_models:
            optimizer_node_model.zero_grad()
        x_unsorted = data['x_bonds'].float()
        y_unsorted = data['y_bonds'].long()
        x_nodes_unsorted = data['x_nodes'].float()
        y_nodes_unsorted = data['y_nodes'].long()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]

        x_nodes_unsorted = x_nodes_unsorted[:, 0:y_len_max, :]
        y_nodes_unsorted = y_nodes_unsorted[:, 0:y_len_max, :]

        y_len, sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        x_nodes = torch.index_select(x_nodes_unsorted,0,sort_index)
        y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index)
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)

        x_nodes = Variable(x_nodes).to(args.device)
        y_nodes = Variable(y_nodes).to(args.device)

        h = rnn(x, nodes=x_nodes, pack=True, input_len=y_len)
        y_pred = output(h)
        # y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # print(y_pred.shape)
        # print(y.shape)
        # print(y_node_pred.shape)
        # print(y_nodes.shape)
        # print("-----")
        # use cross entropy loss
        y_pred = torch.reshape(y_pred, (-1, y_pred.size()[-1]))
        y = torch.flatten(y)
        
        loss = cross_entropy_weight(y_pred, y)
        feature_out = feature_extractor(h)
        acc_dict = {}
        for n_model_idx in range(len(node_models))[:6]:
            curr_y_node = y_nodes[:,:,n_model_idx]
            n_model = node_models[n_model_idx]
            y_node_pred = n_model(feature_out)
            y_node_pred = pack_padded_sequence(y_node_pred, y_len, batch_first=True)
            y_node_pred = pad_packed_sequence(y_node_pred, batch_first=True)[0]
            if n_model_idx == 1:
                y_node_pred = y_node_pred.squeeze(-1)
                if n_model_idx == 0:
                    loss += args.atomic_loss_factor * torch.nn.functional.mse_loss(y_node_pred, curr_y_node.float())
                else:
                    loss += torch.nn.functional.mse_loss(y_node_pred, curr_y_node.float())
                acc_node = y_node_pred.long()
                acc_dict[MODELS[n_model_idx] + " Accuracy"] = (acc_node == curr_y_node).sum()/(curr_y_node.flatten().size()[0])
            else:
                y_node_pred = torch.reshape(y_node_pred, (-1, y_node_pred.size()[-1]))
                curr_y_node = torch.flatten(curr_y_node)
                loss += cross_entropy_weight(y_node_pred, curr_y_node)
                disc_node_preds = torch.argmax(torch.softmax(y_node_pred,-1),-1)
                acc_dict[MODELS[n_model_idx] + " Accuracy"] = (disc_node_preds == curr_y_node).sum()/(curr_y_node.size()[0])
        #     print("---------")
        # y = torch.flatten(y)
        # loss = cross_entropy_weight(y_pred, y)
        # loss = None
        # for i in range(y_pred.size()[2]):
        #     curr_y_pred = y_pred[:,:,i,:]
        #     curr_y = y[:,:,i]

        loss.backward()

        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        # for optimizer_node_model in optimizer_node_models:
        #     optimizer_node_model.step()
        scheduler_output.step()
        scheduler_rnn.step()
        # for scheduler_node_model in scheduler_node_models:
        #     scheduler_node_model.step()
        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))
        disc_y_preds = torch.argmax(torch.softmax(y_pred,-1),-1)
        y_acc = torch.sum(disc_y_preds == y)/y.size()[0]
        # disc_node_preds = torch.argmax(torch.softmax(y_node_pred,-1),-1)
        # y_node_acc = torch.sum(disc_node_preds == y_nodes)/y_nodes.size()[0]
        wandb.log({"loss": loss.item(), 'epoch': epoch, 'batch': batch_idx, 'Edge Accuracy': y_acc, **acc_dict})

        loss_sum += loss.item()
    rnn.eval()
    output.eval()
    for node_model in node_models:
        node_model.eval()
    val_loss_sum = 0
    for val_batch_idx, data in enumerate(val_loader):
        x_unsorted = data['x_bonds'].float()
        y_unsorted = data['y_bonds'].long()
        x_nodes_unsorted = data['x_nodes'].float()
        y_nodes_unsorted = data['y_nodes'].long()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]

        x_nodes_unsorted = x_nodes_unsorted[:, 0:y_len_max, :]
        y_nodes_unsorted = y_nodes_unsorted[:, 0:y_len_max, :]

        y_len, sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        x_nodes = torch.index_select(x_nodes_unsorted,0,sort_index)
        y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index)
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)

        x_nodes = Variable(x_nodes).to(args.device)
        y_nodes = Variable(y_nodes).to(args.device)

        h = rnn(x, nodes=x_nodes, pack=True, input_len=y_len)
        y_pred = output(h)
        # y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # print(y_pred.shape)
        # print(y.shape)
        # print(y_node_pred.shape)
        # print(y_nodes.shape)
        # print("-----")
        # use cross entropy loss
        y_pred = torch.reshape(y_pred, (-1, y_pred.size()[-1]))
        y = torch.flatten(y)
        
        val_loss = cross_entropy_weight(y_pred, y)
        feature_out = feature_extractor(h)
        val_acc_dict = {}
        for n_model_idx in range(len(node_models))[:6]:
            curr_y_node = y_nodes[:,:,n_model_idx]
            n_model = node_models[n_model_idx]
            y_node_pred = n_model(feature_out)
            y_node_pred = pack_padded_sequence(y_node_pred, y_len, batch_first=True)
            y_node_pred = pad_packed_sequence(y_node_pred, batch_first=True)[0]
            if n_model_idx == 1:
                y_node_pred = y_node_pred.squeeze(-1)
                val_loss += torch.nn.functional.mse_loss(y_node_pred, curr_y_node.float())
                acc_node = y_node_pred.long()
                val_acc_dict["Val "+ MODELS[n_model_idx] + " Accuracy"] = (acc_node == curr_y_node).sum()/(curr_y_node.flatten().size()[0])
            else:
                y_node_pred = torch.reshape(y_node_pred, (-1, y_node_pred.size()[-1]))
                curr_y_node = torch.flatten(curr_y_node)
                val_loss += cross_entropy_weight(y_node_pred, curr_y_node)
                disc_node_preds = torch.argmax(torch.softmax(y_node_pred,-1),-1)
                val_acc_dict["Val "+ MODELS[n_model_idx] + " Accuracy"] = (disc_node_preds == curr_y_node).sum()/(curr_y_node.size()[0])
        wandb.log({'epoch': epoch, 'Val batch': val_batch_idx, 'Val Edge Accuracy': y_acc, **val_acc_dict})
        val_loss_sum += val_loss.item()
    wandb.log({'epoch': epoch, "Epoch Loss": loss_sum/(batch_idx+1),'Epoch Val Loss': val_loss_sum/(val_batch_idx+1), })
    return loss_sum/(batch_idx+1)


def test_mlp_epoch(epoch, args, rnn, output, node_layer, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to(args.device)
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        n_pred_step = node_layer(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(args.device)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)


    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list



def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to(args.device)
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to(args.device), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(args.device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to(args.device)
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised_simple(y_pred_step, y[:,i:i+1,:].to(args.device), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(args.device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss

        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j+1,y.size(2))
            loss += binary_cross_entropy_weight(y_pred[:,j,0:end_idx], y[:,j,0:end_idx])*end_idx


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data[0]
    return loss_sum/(batch_idx+1)





## too complicated, deprecated
# def test_mlp_partial_bfs_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to(args.device)
#         for i in range(max_num_node):
#             # 1 back up hidden state
#             hidden_prev = Variable(rnn.hidden.data).to(args.device)
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to(args.device), current=i, y_len=y_len, sample_time=sample_time)
#             y_pred_long[:, i:i + 1, :] = x_step
#
#             rnn.hidden = Variable(rnn.hidden.data).to(args.device)
#
#             print('finish node', i)
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()
#
#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


def train_grnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output, node_layer=None):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x_bonds'].float()
        y_unsorted = data['y_bonds'].long()
        x_nodes_unsorted = data['x_nodes'].float()
        y_nodes_unsorted = data['y_nodes'].long()
        y_len_unsorted = data['len']
        y_len_node_max = max(y_len_unsorted)
        x_nodes_unsorted = x_nodes_unsorted[:, 0:y_len_node_max, :]
        y_nodes_unsorted = y_nodes_unsorted[:, 0:y_len_node_max, :]

        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
#         # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        x_nodes = torch.index_select(x_nodes_unsorted,0,sort_index)
        y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index)

        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)
        output_x = Variable(output_x).to(args.device)
        output_y = Variable(output_y).to(args.device)

        y_nodes_reshape = pack_padded_sequence(y_nodes,y_len,batch_first=True).data
        idx = [i for i in range(y_nodes_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_nodes_reshape = y_nodes_reshape.index_select(0, idx)
        y_nodes_reshape = y_nodes_reshape.view(y_nodes_reshape.size(0),y_nodes_reshape.size(1),1)
        output_x_nodes = torch.cat((torch.ones(y_nodes_reshape.size(0),1,1),y_nodes_reshape[:,0:-1,0:1]),dim=1)
        output_y_nodes = y_nodes_reshape
        output_y_node_len = []
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_node_len.extend([min(i,y_nodes.size(2))]*count_temp)

        x_nodes = Variable(x_nodes).to(args.device)
        y_nodes = Variable(y_nodes).to(args.device)
        output_x_nodes = Variable(output_x_nodes).to(args.device)
        output_y_nodes = Variable(output_y_nodes).to(args.device)

        model = CompleteGRNN(x_unsorted, rnn, output, node_layer, args.num_layers, output_y_len, output_y_node_len, y_len)
        if epoch == 0:
            res = make_dot(model(x,output_x,output_x_nodes, x_nodes), params=dict(model.named_parameters()))
            res.render(directory='doctest-output').replace('\\', '/')
        y_pred, y_nodes_pred = model.forward(x,output_x,output_x_nodes, x_nodes)
        if isinstance(output, MLP_plain):
            y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
            y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            output_y = y
        else:
            y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
            y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
            output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        loss = None
        for t in range(y_pred.size()[1]):
            inp_target = output_y[:,t,:].squeeze(-1)
            if not loss:
                loss = cross_entropy_weight(y_pred[:,t,:], inp_target)
            else:
                loss += cross_entropy_weight(y_pred[:,t,:], inp_target)

        if isinstance(node_layer, MLP_plain):
            y_nodes_pred = pack_padded_sequence(y_nodes_pred, y_nodes, batch_first=True)
            y_nodes_pred = pad_packed_sequence(y_nodes_pred, batch_first=True)[0]
            output_y_nodes = y_nodes
        else:
            y_nodes_pred = pack_padded_sequence(y_nodes_pred, output_y_node_len, batch_first=True)
            y_nodes_pred = pad_packed_sequence(y_nodes_pred, batch_first=True)[0]
            output_y_nodes = pack_padded_sequence(output_y_nodes,output_y_node_len,batch_first=True)
            output_y_nodes = pad_packed_sequence(output_y_nodes,batch_first=True)[0]

        for t in range(output_y_nodes.size()[1]):
            inp_target = output_y_nodes[:,t,:].squeeze(-1)
            if not loss:
                loss = cross_entropy_weight(y_nodes_pred[:,t,:], inp_target)
            else:
                loss += cross_entropy_weight(y_nodes_pred[:,t,:], inp_target)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))


        wandb.log({"loss": loss.item(), 'epoch': epoch, 'batch': batch_idx})
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.item()*feature_dim
    return loss_sum/(batch_idx+1)



def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output, node_layer=None):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x_bonds'].float()
        y_unsorted = data['y_bonds'].long()
        x_nodes_unsorted = data['x_nodes'].float()
        y_nodes_unsorted = data['y_nodes'].long()
        y_len_unsorted = data['len']
        y_len_node_max = max(y_len_unsorted)
        x_nodes_unsorted = x_nodes_unsorted[:, 0:y_len_node_max, :]
        y_nodes_unsorted = y_nodes_unsorted[:, 0:y_len_node_max, :]

        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
    
#         # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
#         # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

#         # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        x_nodes = torch.index_select(x_nodes_unsorted,0,sort_index)
        y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index)

        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # # sort input
        # y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        # y_len = y_len.numpy().tolist()
        # x = torch.index_select(x_unsorted,0,sort_index)
        # y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)
        output_x = Variable(output_x).to(args.device)
        output_y = Variable(output_y).to(args.device)
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, nodes=x_nodes, pack=True, input_len=y_len)
        # print("hidden")
        # print(h.shape)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(args.device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).to(args.device)
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        # print("output hidden")
        # print(output.hidden.shape)
        # print("output x")
        # print(output_x.shape)
        y_pred = output(output_x, pack=False, input_len=output_y_len)
        # print("y_pred")
        # print(y_pred.shape)
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        loss = None
        for t in range(y_pred.size()[1]):
            inp_target = output_y[:,t,:].squeeze(-1)
            if not loss:
                loss = cross_entropy_weight(y_pred[:,t,:], inp_target)
            else:
                loss += cross_entropy_weight(y_pred[:,t,:], inp_target)

        if node_layer:
            y_nodes_reshape = pack_padded_sequence(y_nodes,y_len,batch_first=True).data
            idx = [i for i in range(y_nodes_reshape.size(0)-1, -1, -1)]
            idx = torch.LongTensor(idx)
            y_nodes_reshape = y_nodes_reshape.index_select(0, idx)
            y_nodes_reshape = y_nodes_reshape.view(y_nodes_reshape.size(0),y_nodes_reshape.size(1),1)
            output_x_nodes = torch.cat((torch.ones(y_nodes_reshape.size(0),1,1),y_nodes_reshape[:,0:-1,0:1]),dim=1)
            output_y_nodes = y_nodes_reshape
            output_y_node_len = []
            for i in range(len(output_y_len_bin)-1,0,-1):
                count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
                output_y_node_len.extend([min(i,y_nodes.size(2))]*count_temp)

            x_nodes = Variable(x_nodes).to(args.device)
            y_nodes = Variable(y_nodes).to(args.device)
            output_x_nodes = Variable(output_x_nodes).to(args.device)
            output_y_nodes = Variable(output_y_nodes).to(args.device)
            node_layer.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0)
            y_nodes_pred = node_layer(output_x_nodes, pack=True, input_len=output_y_node_len)
            y_nodes_pred = pack_padded_sequence(y_nodes_pred, output_y_node_len, batch_first=True)
            y_nodes_pred = pad_packed_sequence(y_nodes_pred, batch_first=True)[0]
            output_y_nodes = pack_padded_sequence(output_y_nodes,output_y_node_len,batch_first=True)
            output_y_nodes = pad_packed_sequence(output_y_nodes,batch_first=True)[0]
            for t in range(output_y_nodes.size()[1]):
                inp_target = output_y_nodes[:,t,:].squeeze(-1)
                print(inp_target)
                if not loss:
                    loss = cross_entropy_weight(y_nodes_pred[:,t,:], inp_target)
                else:
                    loss += cross_entropy_weight(y_nodes_pred[:,t,:], inp_target)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))


        # wandb.log({"loss": loss.item(), 'epoch': epoch, 'batch': batch_idx})
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.item()*feature_dim
    return loss_sum/(batch_idx+1)


# def train_rnn_epoch(epoch, args, rnn, output, data_loader,
#                     optimizer_rnn, optimizer_output,
#                     scheduler_rnn, scheduler_output, node_layer=None):
#     rnn.train()
#     output.train()
#     loss_sum = 0
#     for batch_idx, data in enumerate(data_loader):
#         rnn.zero_grad()
#         output.zero_grad()
#         x_unsorted = data['x_bonds'].float()
#         y_unsorted = data['y_bonds'].long()
#         x_nodes_unsorted = data['x_nodes'].float()
#         y_nodes_unsorted = data['y_nodes'].long()
#         y_len_node_max = NUM_POS_ATOMS
#         x_nodes_unsorted = x_nodes_unsorted[:, 0:y_len_node_max, :]
#         y_nodes_unsorted = y_nodes_unsorted[:, 0:y_len_node_max, :]

#         y_len_unsorted = data['len']
#         y_len_max = NUM_POS_EDGES
#         x_unsorted = x_unsorted[:, 0:y_len_max, :]
#         y_unsorted = y_unsorted[:, 0:y_len_max, :]
    
#         # initialize lstm hidden state according to batch size
#         rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
#         # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

#         # sort input
#         y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
#         y_len = y_len.numpy().tolist()
#         x = torch.index_select(x_unsorted,0,sort_index)
#         y = torch.index_select(y_unsorted,0,sort_index)

#         x_nodes = torch.index_select(x_nodes_unsorted,0,sort_index)
#         y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index)
#         # input, output for output rnn module
#         # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
#         y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
#         # reverse y_reshape, so that their lengths are sorted, add dimension
#         idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
#         idx = torch.LongTensor(idx)
#         y_reshape = y_reshape.index_select(0, idx)
#         y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

#         output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
#         output_y = y_reshape

#         # batch size for output module: sum(y_len)
#         output_y_len = []
#         output_y_len_bin = np.bincount(np.array(y_len))
#         for i in range(len(output_y_len_bin)-1,0,-1):
#             count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
#             output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
#         # pack into variable
#         x = Variable(x).to(args.device)
#         y = Variable(y).to(args.device)
#         output_x = Variable(output_x).to(args.device)
#         output_y = Variable(output_y).to(args.device)

#         # if using ground truth to train
#         h = rnn(x, nodes=None, pack=True, input_len=y_len)
#         h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
#         # reverse h
#         idx = [i for i in range(h.size(0) - 1, -1, -1)]
#         idx = Variable(torch.LongTensor(idx)).to(args.device)
#         h = h.index_select(0, idx)
#         hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).to(args.device)

#         output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
#         y_pred = output(output_x, pack=True, input_len=output_y_len)
#         # y_pred = F.softmax(y_pred)
#         # clean
#         y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
#         y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
#         output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
#         output_y = pad_packed_sequence(output_y,batch_first=True)[0]
#         output_y = output_y.squeeze(-1)

#         # use cross entropy loss
#         loss = cross_entropy_weight(y_pred, output_y)
#         if node_layer:
#             y_nodes_reshape = pack_padded_sequence(y_nodes,y_len,batch_first=True).data
#             idx = [i for i in range(y_nodes_reshape.size(0)-1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             y_nodes_reshape = y_nodes_reshape.index_select(0, idx)
#             y_nodes_reshape = y_nodes_reshape.view(y_nodes_reshape.size(0),y_nodes_reshape.size(1),1)
#             output_x_nodes = torch.cat((torch.ones(y_nodes_reshape.size(0),1,1),y_nodes_reshape[:,0:-1,0:1]),dim=1)
#             output_y_nodes = y_nodes_reshape
#             output_y_node_len = []
#             for i in range(len(output_y_len_bin)-1,0,-1):
#                 count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
#                 output_y_node_len.extend([min(i,y_nodes.size(2))]*count_temp)
            



#             x_nodes = Variable(x_nodes).to(args.device)
#             y_nodes = Variable(y_nodes).to(args.device)
#             output_x_nodes = Variable(output_x_nodes).to(args.device)
#             output_y_nodes = Variable(output_y_nodes).to(args.device)
#             node_layer.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0)
#             y_nodes_pred = node_layer(output_x_nodes, pack=True, input_len=output_y_node_len)
#             y_nodes_pred = pack_padded_sequence(y_nodes_pred, output_y_node_len, batch_first=True)
#             y_nodes_pred = pad_packed_sequence(y_nodes_pred, batch_first=True)[0]
#             output_y_nodes = pack_padded_sequence(output_y_nodes,output_y_node_len,batch_first=True)
#             output_y_nodes = pad_packed_sequence(output_y_nodes,batch_first=True)[0]
#             output_y_nodes = output_y_nodes.squeeze(-1)
#             loss += cross_entropy_weight(y_nodes_pred, output_y_nodes)
#         loss.backward()
#         # update deterministic and lstm
#         optimizer_output.step()
#         optimizer_rnn.step()
#         scheduler_output.step()
#         scheduler_rnn.step()


#         if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
#             print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
#                 epoch, args.epochs,loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

#         # logging
#         wandb.log({"loss": loss.item(), 'epoch': epoch, 'batch': batch_idx})
#         feature_dim = y.size(1)*y.size(2)
#         loss_sum += loss.data.item()*feature_dim
#     return loss_sum/(batch_idx+1)



def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(args.device) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to(args.device)
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).to(args.device)
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).to(args.device)
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).to(args.device)
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).to(args.device)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(args.device)
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)
        # wandb.log({"Predicted": wandb.Molecule.from_rdkit(nx_to_mol(G_pred)), "Test Batch": i, "Epoch": epoch})
    return G_pred_list




def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)
        output_x = Variable(output_x).to(args.device)
        output_y = Variable(output_y).to(args.device)
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(args.device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).to(args.device)
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
        # print(y_pred.size())
        feature_dim = y_pred.size(0)*y_pred.size(1)
        loss_sum += loss.data[0]*feature_dim/y.size(0)
    return loss_sum/(batch_idx+1)


########### train function for LSTM + VAE
def train(args, dataset_train, dataset_val, rnn, output, node_layers, feature_extractor):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.AdamW(rnn.parameters(), lr=args.lr)
    optimizer_output = optim.AdamW(output.parameters(), lr=args.output_lr)
    optimizer_node_models = []
    scheduler_node_models = []
    lrs = {0: args.atomic_lr, 1: args.formal_lr, 2:args.chiral_lr, 3:args.hybr_lr,4:args.numh_lr,5:args.aromatic_lr}
    for i in range(len(node_layers)):
        optimizer_node_m = optim.AdamW(output.parameters(), lr=lrs[i])
        optimizer_node_models.append(optimizer_node_m)
        scheduler_node_models.append(MultiStepLR(optimizer_node_m, milestones=args.milestones, gamma=args.lr_rate))

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)
    # start main loop
    time_all = np.zeros(args.epochs)
    for epoch in tqdm(range(args.epochs)):
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, node_layers, feature_extractor, dataset_train,
                            optimizer_rnn, optimizer_output, optimizer_node_models,
                            scheduler_rnn, scheduler_output, scheduler_node_models,
                            dataset_val)
        elif 'GraphRNN_RNN' in args.note:
            train_grnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output, node_layer=node_layers)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    curr_time = time.time()
    torch.save(rnn.state_dict(), './models/model-%d-rnn.pth' % (curr_time))
    torch.save(output.state_dict(), './models/model-%d-ouput.pth' % (curr_time))
    for node_layer_idx in range(len(node_layers)):
        model_name = MODELS[node_layer_idx]
        node_layer = node_layers[node_layer_idx]
        torch.save(node_layer.state_dict(), './models/model-%d-%s-node.pth' % (curr_time, model_name))
        artifact.add_file('./models/model-%d-%s-node.pth' % (curr_time, model_name))
# Save as artifact for version control.
    artifact = wandb.Artifact('model-%d' % (curr_time), type='model')
    artifact.add_file('./models/model-%d-rnn.pth'% (curr_time))
    artifact.add_file('./models/model-%d-ouput.pth'% (curr_time))

    wandb.log_artifact(artifact)
    np.save(args.timing_save_path+args.fname,time_all)
    

########### for graph completion task
def train_graph_completion(args, dataset_test, rnn, output):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))

    for sample_time in range(1,4):
        if 'GraphRNN_MLP' in args.note:
            G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        if 'GraphRNN_VAE' in args.note:
            G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
        save_graph_list(G_pred, fname)
    print('graph completion done, graphs saved')


########### for NLL evaluation
def train_nll(args, dataset_train, dataset_test, rnn, output,graph_validate_len,graph_test_len, max_iter = 1000):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
    with open(fname_output, 'w+') as f:
        f.write(str(graph_validate_len)+','+str(graph_test_len)+'\n')
        f.write('train,test\n')
        for iter in range(max_iter):
            if 'GraphRNN_MLP' in args.note:
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
            if 'GraphRNN_RNN' in args.note:
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
            print('train',nll_train,'test',nll_test)
            f.write(str(nll_train)+','+str(nll_test)+'\n')

    print('NLL evaluation done')
