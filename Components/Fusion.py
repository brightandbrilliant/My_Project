import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, node_dim, graph_dim, out_dim):
        super(FusionModule, self).__init__()
        self.fc = nn.Linear(node_dim + graph_dim, out_dim)

    def forward(self, node_embed, graph_embed, batch=None):
        if len(graph_embed.shape) == 2 and node_embed.shape[0] != graph_embed.shape[0]:
            graph_embed = graph_embed[batch]  # 匹配每个节点所在图

        concat = torch.cat([node_embed, graph_embed], dim=-1)
        return self.fc(concat)


class AttentionalFusion(nn.Module):
    def __init__(self, node_dim, graph_dim, out_dim, hidden_dim=128):
        super(AttentionalFusion, self).__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(node_dim + graph_dim, out_dim)

    def forward(self, node_embed, graph_embed, batch=None):
        if graph_embed.dim() == 2 and node_embed.size(0) != graph_embed.size(0):
            # 将图嵌入扩展到每个节点
            graph_embed = graph_embed[batch]

        # 计算注意力分数
        node_proj = torch.tanh(self.node_proj(node_embed))        # [N, H]
        graph_proj = torch.tanh(self.graph_proj(graph_embed))     # [N, H]

        combined = node_proj + graph_proj                         # [N, H]
        attn_weights = torch.sigmoid(self.attn_score(combined))   # [N, 1]

        # 使用注意力加权融合
        fused = attn_weights * node_embed + (1 - attn_weights) * graph_embed  # [N, D]
        out = self.output_proj(torch.cat([fused, node_embed], dim=-1))        # 可选：也可拼接 graph_embed
        return out


class CrossAttentionFusionModule(nn.Module):
    def __init__(self, node_dim, graph_dim, out_dim, hidden_dim=64):
        super(CrossAttentionFusionModule, self).__init__()

        # 映射到共同维度
        self.query_proj = nn.Linear(node_dim, hidden_dim)
        self.key_proj = nn.Linear(graph_dim, hidden_dim)
        self.value_proj = nn.Linear(graph_dim, hidden_dim)

        # 输出层
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, node_embed, graph_embed, batch=None):
        """
        node_embed: [N, node_dim]
        graph_embed: [B, graph_dim] or [N, graph_dim] (if broadcasted per node)
        batch: [N] 表示每个节点属于哪个图
        """
        # Step 1: 对 graph_embed 进行 broadcast 到每个节点
        if len(graph_embed.shape) == 2 and node_embed.shape[0] != graph_embed.shape[0]:
            assert batch is not None, "需要 batch 来 broadcast graph_embed"
            graph_embed = graph_embed[batch]  # [N, graph_dim]

        # Step 2: 线性映射 Q, K, V
        Q = self.query_proj(node_embed)  # [N, hidden_dim]
        K = self.key_proj(graph_embed)  # [N, hidden_dim]
        V = self.value_proj(graph_embed)  # [N, hidden_dim]

        # Step 3: Attention Score + Softmax
        attn_score = (Q * K).sum(dim=-1, keepdim=True) / (Q.size(-1) ** 0.5)  # [N, 1]
        attn_weight = F.softmax(attn_score, dim=0)  # softmax 可调整为 dim=1 视情形而定

        # Step 4: 加权聚合
        attn_output = attn_weight * V  # [N, hidden_dim]

        # Step 5: 输出映射
        fused = self.output_proj(attn_output)  # [N, out_dim]
        return fused
