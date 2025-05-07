import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class UserEmbeddingGenerator(nn.Module):
    '''
    用户图嵌入生成器：用于生成中间节点表示，供个性化推荐使用。
    '''
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers=2, dropout=0.3):
        '''
        :param input_dim: 输入特征维度
        :param hidden_dim: 中间隐藏层维度
        :param embed_dim: 输出嵌入维度（推荐使用的用户表示）
        :param num_layers: 图卷积层数
        :param dropout: Dropout比例
        '''
        super(UserEmbeddingGenerator, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_dim = embed_dim

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, embed_dim))  # 最后一层输出目标维度

    def forward(self, data):
        '''
        :param data: 包含图结构的 Data 对象（需包含 x 和 edge_index）
        :return: 节点嵌入（如用户表示）
        '''
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x  # 返回节点的最终嵌入



