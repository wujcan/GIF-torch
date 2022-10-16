import os
import pickle
import logging
import shutil

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

import config


class DataStore:
    def __init__(self, args):
        self.logger = logging.getLogger('data_store')
        self.args = args

        self.dataset_name = self.args['dataset_name']
        self.num_features = {
            "cora": 1433,
            "pubmed": 500,
            "citeseer": 3703,
            "CS": 6805,
            "Physics": 8415,            
            "ogbn-arxiv": 128,
            "ogbn-products": 100,
        }
        self.target_model = self.args['target_model']

        self.determine_data_path()

    def determine_data_path(self):
        embedding_name = '_'.join(('embedding', self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        target_model_name = '_'.join((self.target_model, self.args['unlearn_task'], str(self.args['unlearn_ratio'])))
        optimal_weight_name = '_'.join((self.target_model, self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        processed_data_prefix = config.PROCESSED_DATA_PATH + self.dataset_name + "/"
        self.train_test_split_file =  processed_data_prefix + "train_test_split" + str(self.args['test_ratio'])
        self.train_data_file = processed_data_prefix + "train_data"
        self.train_graph_file = processed_data_prefix + "train_graph"
        self.embedding_file = processed_data_prefix + embedding_name
        self.unlearned_file = processed_data_prefix+ '_'.join(('unlearned', self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        self.target_model_file = config.MODEL_PATH + self.dataset_name + '/' + target_model_name
        self.optimal_weight_file = config.ANALYSIS_PATH + 'optimal/' + self.dataset_name + '/' + optimal_weight_name
        self.posteriors_file = config.ANALYSIS_PATH + 'posteriors/' + self.dataset_name + '/' + target_model_name

        dir_lists = [s + self.dataset_name for s in [config.PROCESSED_DATA_PATH,
                                                     config.MODEL_PATH,
                                                     config.ANALYSIS_PATH + 'optimal/',
                                                     config.ANALYSIS_PATH + 'posteriors/']]
        for dir in dir_lists:
            self._check_and_create_dirs(dir)

    def _check_and_create_dirs(self, folder):
        if not os.path.exists(folder):
            try:
                self.logger.info("checking directory %s", folder)
                os.makedirs(folder, exist_ok=True)
                self.logger.info("new directory %s created", folder)
            except OSError as error:
                self.logger.info("deleting old and creating new empty %s", folder)
                shutil.rmtree(folder)
                os.mkdir(folder)
                self.logger.info("new empty directory %s created", folder)
        else:
            self.logger.info("folder %s exists, do not need to create again.", folder)

    def load_raw_data(self):
        self.logger.info('loading raw data')
        if not self.args['is_use_node_feature']:
            self.transform = T.Compose([
                T.OneHotDegree(-2, cat=False)  # use only node degree as node feature.
            ])
        else:
            self.transform = None

        if self.dataset_name in ["cora", "pubmed", "citeseer"]:
            dataset = Planetoid(config.RAW_DATA_PATH, self.dataset_name, transform=T.NormalizeFeatures())
            labels = np.unique(dataset.data.y.numpy())
            data = dataset[0]

        elif self.dataset_name in ["CS", "Physics"]:
            if self.dataset_name == "Physics":
                dataset = Coauthor(config.RAW_DATA_PATH, name="Physics", pre_transform=self.transform)
            else:
                dataset = Coauthor(config.RAW_DATA_PATH, name="CS", pre_transform=self.transform)
            data = dataset[0]

        elif self.dataset_name in ['ogbn-arxiv', 'ogbn-products']:
            
            dataset = PygNodePropPredDataset(name=self.dataset_name, root='../dataset')
            ogb_data = dataset[0]
            split_idx = dataset.get_idx_split()
            
            mask = torch.zeros(ogb_data.x.size(0))
            mask[split_idx['train']] = 1
            ogb_data.train_mask = mask.to(torch.bool)
            ogb_data.train_indices = split_idx['train']

            mask = torch.zeros(ogb_data.x.size(0))
            mask[split_idx['valid']] = 1
            ogb_data.val_mask = mask.to(torch.bool)
            ogb_data.val_indices = split_idx['valid']

            mask = torch.zeros(ogb_data.x.size(0))
            mask[split_idx['test']] = 1
            ogb_data.test_mask = mask.to(torch.bool)
            ogb_data.test_indices = split_idx['test']

            ogb_data.y = ogb_data.y.flatten()
            data = ogb_data
        else:
            raise Exception('unsupported dataset')

        data.name = self.dataset_name

        return data

    def save_train_data(self, train_data):
        self.logger.info('saving train data')
        pickle.dump(train_data, open(self.train_data_file, 'wb'))

    def load_train_data(self):
        self.logger.info('loading train data')
        return pickle.load(open(self.train_data_file, 'rb'))

    def save_train_graph(self, train_data):
        self.logger.info('saving train graph')
        pickle.dump(train_data, open(self.train_graph_file, 'wb'))

    def load_train_graph(self):
        self.logger.info('loading train graph')
        return pickle.load(open(self.train_graph_file, 'rb'))

    def save_train_test_split(self, train_indices, test_indices):
        self.logger.info('saving train test split data')
        pickle.dump((train_indices, test_indices), open(self.train_test_split_file, 'wb'))

    def load_train_test_split(self):
        self.logger.info('loading train test split data')
        return pickle.load(open(self.train_test_split_file, 'rb'))

    def save_embeddings(self, embeddings):
        self.logger.info('saving embedding data')
        pickle.dump(embeddings, open(self.embedding_file, 'wb'))

    def load_embeddings(self):
        self.logger.info('loading embedding data')
        return pickle.load(open(self.embedding_file, 'rb'))

    def load_unlearned_data(self, suffix):
        file_path = '_'.join((self.unlearned_file, suffix))
        self.logger.info('loading unlearned data from %s' % file_path)
        return pickle.load(open(file_path, 'rb'))

    def save_unlearned_data(self, data, suffix):
        self.logger.info('saving unlearned data %s' % suffix)
        pickle.dump(data, open('_'.join((self.unlearned_file, suffix)), 'wb'))

    def save_target_model(self, run, model, suffix=''):
        model.save_model(self.target_model_file + '_' + str(run))

    def load_target_model(self, run, model, suffix=''):
        model.load_model(self.target_model_file + '_'  + '_' + str(0))

    def save_optimal_weight(self, weight, run):
        torch.save(weight, self.optimal_weight_file + '_' + str(run))

    def load_optimal_weight(self, run):
        return torch.load(self.optimal_weight_file + '_' + str(run))

    def save_posteriors(self, posteriors, run, suffix=''):
        torch.save(posteriors, self.posteriors_file + '_' + str(run) + suffix)

    def load_posteriors(self, run):
        return torch.load(self.posteriors_file + '_' + str(run))

    def _extract_embedding_method(self, partition_method):
        return partition_method.split('_')[0]
