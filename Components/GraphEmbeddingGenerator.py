import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool,global_mean_pool
from torch_geometric.data import Data


class GraphEmbeddingGenerator(nn.Module):
    '''
    用于提取每个 client（如一个 app）的图风格嵌入。
    图嵌入将作为全局语境参与用户推荐，体现 client 偏好（如虎扑偏体育，知乎偏知识）。
    '''
    def __init__(self, node_feat_dim, hidden_dim, style_dim, num_layers=2, dropout=0.3):
        '''
        参数说明：
        node_feat_dim : 节点特征维度（输入维度）
        hidden_dim    : 中间 GCN 层维度
        style_dim     : 输出图风格嵌入维度
        num_layers    : 图卷积层数
        dropout       : dropout 概率
        '''
        super(GraphEmbeddingGenerator, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 多层 GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 将图嵌入投影为指定维度的风格向量
        self.convs.append(GCNConv(hidden_dim, style_dim))

    def forward(self, data):
        '''
        输入：
        data.x         节点特征 (num_nodes, node_feat_dim)
        data.edge_index 边信息
        返回：
        graph_style    每个图的风格嵌入 (num_graphs, style_dim)
        '''
        x, edge_index,batch = data.x, data.edge_index,data.batch

        # 多层 GCN
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # 图级聚合
        graph_embed = global_mean_pool(x, batch)

        return graph_embed









