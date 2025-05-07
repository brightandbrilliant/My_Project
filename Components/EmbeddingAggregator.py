import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close


class PersonalizedUserAggregator(nn.Module):
    def __init__(self, embed_dim, n_clients):
        """
        个性化用户嵌入聚合器，通过注意力机制聚合来自多个客户端的用户嵌入。

        Args:
            embed_dim (int): 用户嵌入的维度
            n_clients (int): 客户端数量，决定了聚合时的分支数
        """
        super(PersonalizedUserAggregator, self).__init__()
        self.embed_dim = embed_dim
        self.n_clients = n_clients

        # 每个客户端的注意力查询权重向量（可学习）
        self.attn_query = nn.Parameter(torch.randn(embed_dim, 1))
        nn.init.xavier_uniform_(self.attn_query)  # 使用 Xavier 初始化查询权重

        # 可选：增加一个小的 MLP 进行特征转换，增强非线性特征
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  # 特征映射
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)   # 输出与输入相同的维度
        )

    def forward(self, client_user_embeds, local_user_embeds, alpha):
        """
        聚合不同客户端的用户嵌入。

        Args:
            :param client_user_embeds: list of tensors
            :param local_user_embeds: 本地嵌入
            :param alpha: 加权系数

        Returns:
            torch.Tensor: 聚合后的个性化用户嵌入，形状为 [batch_size, embed_dim]

        """
        # 将所有客户端的嵌入堆叠为一个 Tensor，形状为 [batch_size, n_clients, embed_dim]
        client_user_embeds = torch.stack(client_user_embeds, dim=0)
        # 调整形状为形状为 [n_clients, batch_size, embed_dim]
        client_user_embeds = client_user_embeds.permute(1, 0, 2)

        # 特征变换（可选），通过 MLP 增加非线性映射
        transformed_embeds = self.mlp(client_user_embeds)  # 形状仍为 [n_clients, batch_size, embed_dim]

        # 计算注意力分数：通过查询向量与 transformed_embeds 的点积来获得每个客户端的注意力得分
        # attn_scores 形状为 [n_clients, batch_size, 1]
        attn_scores = torch.einsum('nbd, de->nbe', transformed_embeds, self.attn_query)
        # print("注意力分数:")
        # print(attn_scores)

        # 使用 Softmax 对注意力分数进行归一化，得到每个客户端对每个用户的权重
        attn_weights = F.softmax(attn_scores, dim=0)  # 形状为 [n_clients, batch_size, 1]
        # print("注意力权重:")
        # print(attn_weights)

        weighted_embeddings = attn_weights * transformed_embeds  # 利用广播机制

        aggregated_embeddings = weighted_embeddings.sum(dim=0)  # 在客户端维度上求和，形状为[1,batch_size,embed_dim]
        # print("聚合后嵌入:")
        # print(aggregated_embeddings)
        combined_embeds = alpha * local_user_embeds + (1 - alpha) * aggregated_embeddings
        # print(f'combined_embeds:{combined_embeds}')
        return combined_embeds


