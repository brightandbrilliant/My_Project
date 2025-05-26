import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from Components.GraphEmbeddingGenerator import GraphEmbeddingGenerator
from Components.NodeEmbeddingGenerator import UserEmbeddingGenerator
from Components.EmbeddingAggregator import PersonalizedUserAggregator
from Components.Fusion import FusionModule, AttentionalFusion, CrossAttentionFusionModule



class MLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3):
        super(MLPPredictor, self).__init__()
        layers = []

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(num_layers):
            self.linears.append(nn.Linear(dims[i], dims[i+1]))
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(dims[i+1]))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, linear in enumerate(self.linears[:-1]):
            residual = x
            x = linear(x)
            x = self.norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            if residual.shape == x.shape:
                x = x + residual  # 加残差
        x = self.linears[-1](x)
        return x


class Client(nn.Module):
    def __init__(self,
                 node_feat_dim,          # 节点的初始特征维度
                 node_hidden_dim,        # 节点编码器的中间层维度
                 node_embed_dim,         # 节点最终嵌入维度
                 graph_hidden_dim,       # 图编码器中间层维度
                 graph_style_dim,        # 图嵌入最终输出维度（风格维度）
                 fusion_output_dim,      # 融合后输出维度
                 node_num_layers,             # GNN 层数
                 graph_num_layers,          # GNN 层数
                 dropout,                # Dropout 比例
                 n_clients,              # 除了本地端以外的客户数量
                 n_users,                # 用户总数（用于分类预测）
                 ):
        super(Client, self).__init__()

        # 节点嵌入生成器
        self.node_encoder = UserEmbeddingGenerator(
            input_dim=node_feat_dim,
            hidden_dim=node_hidden_dim,
            embed_dim=node_embed_dim,
            num_layers=node_num_layers,
            dropout=dropout
        )

        # 图嵌入生成器
        self.graph_encoder = GraphEmbeddingGenerator(
            node_feat_dim=node_feat_dim,
            hidden_dim=graph_hidden_dim,
            style_dim=graph_style_dim,
            num_layers=graph_num_layers,
            dropout=dropout
        )

        # 用户嵌入的个性化聚合器
        self.attn_aggregator = PersonalizedUserAggregator(
            embed_dim=node_embed_dim,
            n_clients=n_clients
        )

        # 节点嵌入与图嵌入的融合模块
        self.fusion = FusionModule(
            node_dim=node_embed_dim,
            graph_dim=graph_style_dim,
            out_dim=fusion_output_dim
        )

        # 用于最终关注用户预测的分类器
        self.predictor = MLPPredictor(
            input_dim=fusion_output_dim,
            hidden_dim=96,  # 可调
            output_dim=n_users,
            num_layers=4,  # 可调
            dropout=0.2  # 可调
        )

        # 交叉熵损失函数（用于多分类）
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = None

    def create_optimizer(self, lr=1e-3, weight_decay=1e-5):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, data, user_ids, target_labels, alpha, external_node_embeds_dict=None, mask=None, debug=False):
        """
        Args:
            user_ids: 1D Tensor，本 client 所负责的全局 user index（即列 index），用于从 logits 中选出对应部分
            target_labels: [N, num_local_users] 的真实标签
            mask: Optional[Tensor]，形状为 [N] 的布尔张量，表示哪些样本用于训练。
        """

        # Step 1: 本地节点嵌入生成
        local_node_embeds = self.node_encoder(data)
        if debug and (torch.isnan(local_node_embeds).any() or torch.isinf(local_node_embeds).any()):
            print("[Client Debug] local_node_embeds has nan or inf!")

        # Step 2: 聚合外部嵌入
        if external_node_embeds_dict is not None:
            aligned_external_embeds = []
            for i, uid in enumerate(user_ids):
                if uid in external_node_embeds_dict:
                    aligned_external_embeds.append(external_node_embeds_dict[uid])
                else:
                    zeros = [torch.zeros_like(local_node_embeds[i]) for _ in range(self.attn_aggregator.n_clients)]
                    aligned_external_embeds.append(zeros)
            aligned_external_embeds = [torch.stack(embed_list, dim=0) for embed_list in aligned_external_embeds]
            personalized_node_embed = self.attn_aggregator(aligned_external_embeds, local_node_embeds, alpha)
        else:
            personalized_node_embed = local_node_embeds

        if debug and (torch.isnan(personalized_node_embed).any() or torch.isinf(personalized_node_embed).any()):
            print("[Client Debug] personalized_node_embed has nan or inf!")

        # Step 3: 图嵌入生成
        graph_embed = self.graph_encoder(data)
        if debug and (torch.isnan(graph_embed).any() or torch.isinf(graph_embed).any()):
            print("[Client Debug] graph_embed has nan or inf!")

        # Step 4: 融合节点与图嵌入
        fused_embed = self.fusion(personalized_node_embed, graph_embed, batch=data.batch)
        if debug and (torch.isnan(fused_embed).any() or torch.isinf(fused_embed).any()):
            print("[Client Debug] fused_embed has nan or inf!")

        # Step 5: 使用 mask 过滤训练数据
        if mask is not None:
            fused_embed = fused_embed[mask]
            target_labels = target_labels[mask]

        # Step 6: 全量输出 -> 选择本 client 有效列
        # full_logits = self.predictor(fused_embed)  # [N, total_user_count]
        # logits = full_logits[:, user_ids]  # [N, num_local_users]

        logits = self.predictor(fused_embed)

        if debug and (torch.isnan(logits).any() or torch.isinf(logits).any()):
            print("[Client Debug] logits has nan or inf!")

        # Step 7: 计算 BCE 多标签损失
        loss = self.loss_fn(logits, target_labels)

        return loss, logits




