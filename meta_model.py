import copy
import torch
import random
import torch.nn as nn
from samples import sample_datasets, sample_test_datasets, sample_premeta_datasets
from model import GNN, GNN_graphpred
import torch.nn.functional as F
from loader import MoleculeDataset
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x

class Interact_attention(nn.Module):
    def __init__(self, dim, num_tasks):
        super(Interact_attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_tasks * dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class Meta_model(nn.Module):
    def __init__(self, args):
        super(Meta_model,self).__init__()

        self.dataset = args.dataset
        self.num_tasks = args.num_tasks
        self.num_train_tasks = args.num_train_tasks
        self.num_test_tasks = args.num_test_tasks
        self.n_way = args.n_way
        self.m_support = args.m_support
        self.k_query = args.k_query
        self.gnn_type = args.gnn_type

        self.emb_dim = args.emb_dim

        self.device = args.device

        self.add_similarity = args.add_similarity
        self.add_selfsupervise = args.add_selfsupervise
        self.add_masking = args.add_masking
        self.add_weight = args.add_weight
        self.interact = args.interact

        self.batch_size = args.batch_size

        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.criterion = nn.BCEWithLogitsLoss()
        
        self.graph_model = GNN_graphpred(args.num_layer, args.emb_dim, 1, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        if not args.input_model_file == "":
            self.graph_model.from_pretrained(args.input_model_file)

        if self.add_selfsupervise:
            self.self_criterion = nn.BCEWithLogitsLoss()

        if self.add_masking:
            self.masking_criterion = nn.CrossEntropyLoss()
            self.masking_linear = nn.Linear(self.emb_dim, 119)

        if self.add_similarity:
            self.Attention = attention(self.emb_dim)

        if self.interact:
            self.softmax = nn.Softmax(dim=0)
            self.Interact_attention = Interact_attention(self.emb_dim, self.num_train_tasks)
            
        model_param_group = []
        model_param_group.append({"params": self.graph_model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": self.graph_model.pool.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": self.graph_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})

        if self.add_masking:
            model_param_group.append({"params": self.masking_linear.parameters()})

        if self.add_similarity:
            model_param_group.append({"params": self.Attention.parameters()})
            
        if self.interact:
            model_param_group.append({"params": self.Interact_attention.parameters()})
        
        self.optimizer = optim.Adam(model_param_group, lr=args.meta_lr, weight_decay=args.decay)

        # for name, para in self.named_parameters():
        #     if para.requires_grad:
        #         print(name, para.data.shape)
        # raise TypeError

    def update_params(self, loss, update_lr):
        grads = torch.autograd.grad(loss, self.graph_model.parameters())
        return parameters_to_vector(grads), parameters_to_vector(self.graph_model.parameters()) - parameters_to_vector(grads) * update_lr

    def build_negative_edges(self, batch):
        font_list = batch.edge_index[0, ::2].tolist()
        back_list = batch.edge_index[1, ::2].tolist()
        
        all_edge = {}
        for count, front_e in enumerate(font_list):
            if front_e not in all_edge:
                all_edge[front_e] = [back_list[count]]
            else:
                all_edge[front_e].append(back_list[count])
        
        negative_edges = []
        for num in range(batch.x.size()[0]):
            if num in all_edge:
                for num_back in range(num, batch.x.size()[0]):
                    if num_back not in all_edge[num] and num != num_back:
                        negative_edges.append((num, num_back))
            else:
                for num_back in range(num, batch.x.size()[0]):
                    if num != num_back:
                        negative_edges.append((num, num_back))

        negative_edge_index = torch.tensor(np.array(random.sample(negative_edges, len(font_list))).T, dtype=torch.long)

        return negative_edge_index

    def forward(self, epoch):
        support_loaders = []
        query_loaders = []

        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.graph_model.train()

        # tasks_list = random.sample(range(0,self.num_train_tasks), self.batch_size)

        for task in range(self.num_train_tasks):
        # for task in tasks_list:
            dataset = MoleculeDataset("Original_datasets/" + self.dataset + "/new/" + str(task+1), dataset = self.dataset)
            support_dataset, query_dataset = sample_datasets(dataset, self.dataset, task, self.n_way, self.m_support, self.k_query)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            support_loaders.append(support_loader)
            query_loaders.append(query_loader)

        for k in range(0, self.update_step):
            # print(self.fi)
            old_params = parameters_to_vector(self.graph_model.parameters())

            losses_q = torch.tensor([0.0]).to(device)

            # support_params = []
            # support_grads = torch.Tensor(self.num_train_tasks, parameters_to_vector(self.graph_model.parameters()).size()[0]).to(device)

            for task in range(self.num_train_tasks):

                losses_s = torch.tensor([0.0]).to(device)
                if self.add_similarity or self.interact:
                    one_task_emb = torch.zeros(300).to(device)

                for step, batch in enumerate(tqdm(support_loaders[task], desc="Iteration")):
                    batch = batch.to(device)

                    pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    y = batch.y.view(pred.shape).to(torch.float64)
                                
                    loss = torch.sum(self.criterion(pred.double(), y)) /pred.size()[0]

                    if self.add_selfsupervise:
                        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim = 1)

                        negative_edge_index = self.build_negative_edges(batch)
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]], dim = 1)

                        self_loss = torch.sum(self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(negative_score, torch.zeros_like(negative_score)))/negative_edge_index[0].size()[0]

                        loss += (self.add_weight * self_loss)

                    if self.add_masking:
                        mask_num = random.sample(range(0,node_emb.size()[0]), self.batch_size)
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(), batch.x[mask_num,0]))

                    if self.add_similarity or self.interact:
                        one_task_emb = torch.div((one_task_emb + torch.mean(node_emb,0)), 2.0)

                    losses_s += loss

                if self.add_similarity or self.interact:
                    if task == 0:
                        tasks_emb = []
                    tasks_emb.append(one_task_emb)
                
                new_grad, new_params = self.update_params(losses_s, update_lr = self.update_lr)

                vector_to_parameters(new_params, self.graph_model.parameters())

                this_loss_q = torch.tensor([0.0]).to(device)
                for step, batch in enumerate(tqdm(query_loaders[task], desc="Iteration")):
                    batch = batch.to(device)

                    pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    y = batch.y.view(pred.shape).to(torch.float64)

                    loss_q = torch.sum(self.criterion(pred.double(), y))/pred.size()[0]

                    if self.add_selfsupervise:
                        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim = 1)

                        negative_edge_index = self.build_negative_edges(batch)
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]], dim = 1)

                        self_loss = torch.sum(self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(negative_score, torch.zeros_like(negative_score)))/negative_edge_index[0].size()[0]

                        loss_q += (self.add_weight * self_loss)

                    if self.add_masking:
                        mask_num = random.sample(range(0,node_emb.size()[0]), self.batch_size)
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(), batch.x[mask_num,0]))

                    this_loss_q += loss_q

                if task == 0:
                    losses_q = this_loss_q
                else:
                    losses_q = torch.cat((losses_q, this_loss_q), 0)

                vector_to_parameters(old_params, self.graph_model.parameters())

            if self.add_similarity:
                for t_index, one_task_e in enumerate(tasks_emb):
                    if t_index == 0:
                        tasks_emb_new = one_task_e
                    else:
                        tasks_emb_new = torch.cat((tasks_emb_new, one_task_e), 0)
                
                tasks_emb_new = torch.reshape(tasks_emb_new, (self.num_train_tasks, self.emb_dim))
                tasks_emb_new = tasks_emb_new.detach()

                tasks_weight = self.Attention(tasks_emb_new)
                losses_q = torch.sum(tasks_weight * losses_q)

            elif self.interact:
                for t_index, one_task_e in enumerate(tasks_emb):
                    if t_index == 0:
                        tasks_emb_new = one_task_e
                    else:
                        tasks_emb_new = torch.cat((tasks_emb_new, one_task_e), 0)

                tasks_emb_new = tasks_emb_new.detach()
                represent_emb = self.Interact_attention(tasks_emb_new)
                represent_emb = F.normalize(represent_emb, p=2, dim=0)

                tasks_emb_new = torch.reshape(tasks_emb_new, (self.num_train_tasks, self.emb_dim))
                tasks_emb_new = F.normalize(tasks_emb_new, p=2, dim=1)

                tasks_weight = torch.mm(tasks_emb_new, torch.reshape(represent_emb, (self.emb_dim, 1)))
                print(tasks_weight)
                print(self.softmax(tasks_weight))
                print(losses_q)

                # tasks_emb_new = tasks_emb_new * torch.reshape(represent_emb_m, (self.batch_size, self.emb_dim))
                
                losses_q = torch.sum(losses_q * torch.transpose(self.softmax(tasks_weight), 1, 0))
                print(losses_q)

            else:
                losses_q = torch.sum(losses_q)
            
            loss_q = losses_q / self.num_train_tasks       
            self.optimizer.zero_grad()
            loss_q.backward()
            self.optimizer.step()
        
        return []

    def test(self, support_grads):
        accs = []
        old_params = parameters_to_vector(self.graph_model.parameters())
        for task in range(self.num_test_tasks):
            print(self.num_tasks-task)
            dataset = MoleculeDataset("Original_datasets/" + self.dataset + "/new/" + str(self.num_tasks-task), dataset = self.dataset)
            support_dataset, query_dataset = sample_test_datasets(dataset, self.dataset, self.num_tasks-task-1, self.n_way, self.m_support, self.k_query)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)

            device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")

            self.graph_model.eval()

            for k in range(0, self.update_step_test):
                loss = torch.tensor([0.0]).to(device)
                for step, batch in enumerate(tqdm(support_loader, desc="Iteration")):
                    batch = batch.to(device)

                    pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    y = batch.y.view(pred.shape).to(torch.float64)

                    loss += torch.sum(self.criterion(pred.double(), y))/pred.size()[0]

                    if self.add_selfsupervise:
                        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim = 1)

                        negative_edge_index = self.build_negative_edges(batch)
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]], dim = 1)

                        self_loss = torch.sum(self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(negative_score, torch.zeros_like(negative_score)))/negative_edge_index[0].size()[0]

                        loss += (self.add_weight *self_loss)

                    if self.add_masking:
                        mask_num = random.sample(range(0,node_emb.size()[0]), self.batch_size)
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(), batch.x[mask_num,0]))

                    print(loss)

                new_grad, new_params = self.update_params(loss, update_lr = self.update_lr)

                # if self.add_similarity:
                #     new_params = self.update_similarity_params(new_grad, support_grads)

                vector_to_parameters(new_params, self.graph_model.parameters())
                

            y_true = []
            y_scores = []
            for step, batch in enumerate(tqdm(query_loader, desc="Iteration")):
                batch = batch.to(device)

                pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                # print(pred)
                pred = F.sigmoid(pred)
                pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
                pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred)
                y_scores.append(pred)
                y_true.append(batch.y.view(pred.shape))
                

            y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
            y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
           
            roc_list = []
            roc_list.append(roc_auc_score(y_true, y_scores))
            acc = sum(roc_list)/len(roc_list)
            accs.append(acc)

            vector_to_parameters(old_params, self.graph_model.parameters())

        return accs
        