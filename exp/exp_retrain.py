from cgi import test
import logging
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

torch.cuda.empty_cache()
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

from exp.exp import Exp
from lib_gnn_model.gat.gat_net_batch import GATNet
from lib_gnn_model.gin.gin_net_batch import GINNet
from lib_gnn_model.gcn.gcn_net_batch import GCNNet
from lib_gnn_model.graphsage.graphsage_net import SageNet
from lib_gnn_model.sgc.sgc_net_batch import SGCNet
from lib_gnn_model.node_classifier import NodeClassifier
from lib_gnn_model.gnn_base import GNNBase
from parameter_parser import parameter_parser
from lib_utils import utils


class ExpRetraining(Exp):
    def __init__(self, args):
        super(ExpRetraining, self).__init__(args)

        self.logger = logging.getLogger('ExpRetraining')

        self.load_data()
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.gen_train_graph()

        self.target_model_name = self.args['target_model']

        self.determine_target_model()

        run_f1 = np.empty((0))
        run_f1_unlearning = np.empty((0))
        training_times = np.empty((0))
        for run in range(self.args['num_runs']):
            self.logger.info("Run %f" % run)

            run_training_time, grad_all = self._train_model(run)

            f1_score = self.evaluate(run)
            run_f1 = np.append(run_f1, f1_score)
            training_times = np.append(training_times, run_training_time)

        f1_score_avg = np.average(run_f1)
        f1_score_std = np.std(run_f1)
        self.logger.info("f1_score: avg=%s, std=%s" % (f1_score_avg, f1_score_std))
        self.logger.info("model training time: avg=%s seconds" % np.average(training_times))

    def load_data(self):
        self.data = self.data_store.load_raw_data()

    def train_test_split(self):
        if self.args['is_split']:
            self.logger.info('splitting train/test data')
            # use the dataset's default split
            if self.data.name in ['ogbn-arxiv', 'ogbn-products']:
                self.train_indices, self.test_indices = self.data.train_indices.numpy(), self.data.test_indices.numpy()
            else:
                self.train_indices, self.test_indices = train_test_split(np.arange((self.data.num_nodes)), test_size=self.args['test_ratio'], random_state=100)
                
            self.data_store.save_train_test_split(self.train_indices, self.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = self.data_store.load_train_test_split()

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))

    def gen_train_graph(self):
        self.logger.debug("Before deletion. train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))

        if self.args["unlearn_ratio"] != 0:
            if self.args["unlearn_task"] == 'feature':
                unique_nodes = np.random.choice(len(self.train_indices),
                                                int(len(self.train_indices) * self.args['unlearn_ratio']),
                                                replace=False)
                feature_mask = np.random.choice(a=[0.0, 1.0], size=(len(unique_nodes), self.num_feats), p=[1.0, 0.0]).astype(np.float32)
                # self.data.x[unique_nodes, :] = self.data.x[unique_nodes, :] * feature_mask
                self.data.x[unique_nodes] = 0.

            else:
                self.data.edge_index = self._ratio_delete()

    def determine_target_model(self):
        self.logger.info('target model: %s' % (self.args['target_model'],))
        num_classes = len(self.data.y.unique())

        self.target_model = NodeClassifier(self.num_feats, num_classes, self.args)

    def evaluate(self, run):
        self.logger.info('model evaluation')

        start_time = time.time()
        posterior = self.target_model.posterior()
        test_f1 = f1_score(
            self.data.y[self.data['test_mask']].cpu().numpy(), 
            posterior.argmax(axis=1).cpu().numpy(), 
            average="micro"
        )

        evaluate_time = time.time() - start_time
        self.logger.info("Evaluation cost %s seconds." % evaluate_time)

        self.logger.info("Final Test F1: %s" % (test_f1,))
        return test_f1

    def _train_model(self, run):
        self.logger.info('training target models, run %s' % run)

        start_time = time.time()
        self.target_model.data = self.data
        grad_all = self.target_model.train_model()
        train_time = time.time() - start_time

        self.logger.info("Model training time: %s" % (train_time))

        return train_time, grad_all

    def _ratio_delete(self):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]
        if self.args["unlearn_task"] == 'edge':
            remain_indices = np.random.choice(
                unique_indices,
                int(unique_indices.shape[0] * (1.0 - self.args['unlearn_ratio'])),
                replace=False)
        else:
            delete_nodes = np.random.choice(
                len(self.train_indices),
                int(len(self.train_indices) * self.args['unlearn_ratio']),
                replace=False)
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])
