import torch
from typing import Dict, List


class P2PCommunicator:
    def __init__(self, client_id, total_clients):
        """
        :param client_id: 当前客户端编号
        :param total_clients: 所有客户端编号列表（例如 [0, 1, 2]）
        """
        self.client_id = client_id
        self.peers = [cid for cid in total_clients if cid != client_id]

    def pack_user_embeddings(self, user_ids, user_embeds):
        """
        将用户嵌入打包为 P2P 通信格式。

        Args:
            client_id (int): 当前客户端编号
            user_ids (Tensor): shape = [B]
            user_embeds (Tensor): shape = [B, D]

        Returns:
            dict: {
                "client_id": int,
                "client_node_embeddings": [(user_id, user_embedding), ...]
            }
        """
        packed = {
            "client_id": self.client_id,
            "client_node_embeddings": [
                (int(uid), embed.detach().cpu()) for uid, embed in zip(user_ids, user_embeds)
            ]
        }
        return packed

    def send_to_all_peers(self, package: Dict, network: Dict[int, List[Dict]]):
        """
        模拟将打包数据广播给其他所有 clients
        :param package: 打包数据
        :param network: 全局模拟网络环境（dict：client_id -> list of received messages）
        """
        for peer_id in self.peers:
            network[peer_id].append(package)

    def organize_external_node_embeds(self, received_packets, user_ids, embed_dim, device):
        """
        将所有收到的客户端广播包，按 user_id 聚合为 external_node_embeds_dict。

        Args:
            received_packets (List[Dict]): 来自其他客户端的广播包，每个格式为：
                {
                    "client_id": int,
                    "client_node_embeddings": List[Tuple[int, Tensor]]
                }
            user_ids (Tensor): 当前 batch 的 user_id，shape = [B]
            embed_dim (int): 嵌入维度
            device: torch.device

        Returns:
            Dict[int, List[Tensor]]: external_node_embeds_dict[user_id] = [client1_embed, ..., clientN_embed]
        """
        n_clients = len(received_packets)
        external_node_embeds_dict = {}

        for uid in user_ids.tolist():
            external_node_embeds_dict[uid] = []

            for packet in received_packets:
                if packet["client_id"] == self.client_id:
                    continue  # 跳过自己

                # 构建 user_id -> embed 的映射
                embed_map = dict(packet["client_node_embeddings"])

                if uid in embed_map:
                    external_node_embeds_dict[uid].append(embed_map[uid].to(device))
                else:
                    external_node_embeds_dict[uid].append(torch.zeros(embed_dim, device=device))  # 缺失填零

        return external_node_embeds_dict


