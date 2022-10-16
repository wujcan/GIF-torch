import torch
import torch.nn.functional as F
from lib_gnn_model.sgc.sgc_conv_batch import SGConvBatch

class SGCNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(SGCNet, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SGConvBatch(num_feats, 16, cached=False, add_self_loops=True, bias=False))
        self.convs.append(SGConvBatch(16, num_classes, cached=False, add_self_loops=True, bias=False))

    def forward(self, x, adjs, edge_weight):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index, edge_weight=edge_weight[e_id])

            if i != self.num_layers - 1:
                x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)

    def forward_once(self, data, edge_weight):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def forward_once_unlearn(self, data, edge_weight):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = self.convs[0](x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, edge_weight, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index, edge_weight=edge_weight[e_id])

                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()