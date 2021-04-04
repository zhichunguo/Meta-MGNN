import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score

import pandas as pd
from meta_model import Meta_model

import os
import shutil

def main(dataset, input_model_file, gnn_type, add_similarity, add_selfsupervise, add_masking, add_weight, m_support):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size for training (default: 32)') 
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="graphsage")
    parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')

    parser.add_argument('--num_tasks', type=int, default=12, help = "# of tasks")
    parser.add_argument('--num_train_tasks', type=int, default=9, help = "# of training tasks")
    parser.add_argument('--num_test_tasks', type=int, default=3, help = "# of testing tasks")
    parser.add_argument('--n_way', type=int, default=2, help = "n_way of dataset")
    parser.add_argument('--m_support', type=int, default=5, help = "size of the support dataset")
    parser.add_argument('--k_query', type = int, default=128, help = "size of querry datasets")
    parser.add_argument('--meta_lr', type=float, default=0.001) 
    parser.add_argument('--update_lr', type=float, default=0.4) #0.4
    parser.add_argument('--update_step', type=int, default=5) #5
    parser.add_argument('--update_step_test', type=int, default=10) #10
    parser.add_argument('--add_similarity', type=bool, default=False)
    parser.add_argument('--add_selfsupervise', type=bool, default=False)
    parser.add_argument('--interact', type=bool, default=False)
    parser.add_argument('--add_weight', type=float, default=0.1)
    
    args = parser.parse_args()

    args.dataset = dataset
    args.input_model_file = input_model_file
    args.gnn_type = gnn_type
    args.add_similarity = add_similarity
    args.add_selfsupervise = add_selfsupervise
    args.add_masking = add_masking
    args.add_weight = add_weight
    args.m_support = m_support

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset == "tox21":
        args.num_tasks = 12
        args.num_train_tasks = 9
        args.num_test_tasks = 3

    elif args.dataset == "sider":
        args.num_tasks = 27
        args.num_train_tasks = 21
        args.num_test_tasks = 6
    else:
        raise ValueError("Invalid dataset name.")

    model = Meta_model(args).to(device)
    # model.to(device)

    print(args.dataset)

    best_accs = []
    for epoch in range(1, args.epochs+1):
        support_grads = model(epoch)

        if epoch % 1 == 0:
            accs = model.test(support_grads)

            if best_accs != []:
                for acc_num in range(len(best_accs)):
                    if best_accs[acc_num] < accs[acc_num]:
                        best_accs[acc_num] = accs[acc_num]
            else:
                best_accs = accs

            fw = open("result/" + args.dataset + "_" + args.gnn_type + "_" + str(args.m_support) + "_" + str(args.add_similarity) + "_" + str(args.add_selfsupervise) + "_" + str(args.add_masking) + "_" + str(args.add_weight) + "_" + str(args.update_step) + ".txt", "a")
            fw.write("test: " + "\t")
            for i in accs:
                fw.write(str(i) + "\t")

            fw.write("best: " + "\t")
            for i in best_accs:
                fw.write(str(i) + "\t")
            fw.write("\n")
            fw.close()
    

if __name__ == "__main__":
    # dataset, pretrained_model, graph_model, taskaware_attention, edge_pred, atom_pred, weight, #support 
    main("sider", "model_gin/supervised_contextpred.pth", "gin", True, True, True, 0.1, 5)
