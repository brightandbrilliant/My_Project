import torch
from torch_geometric.data import Data



def preprocess_social_graph(raw_users, group_id_to_idx):
    print(f"预处理开始，总共有 {len(raw_users)} 个用户")

    valid_users = []

    # 遍历 key-value
    for user_id, user_info in raw_users.items():
        if not isinstance(user_info, dict):
            continue
        if 'groups' not in user_info:
            continue
        valid_users.append((user_id, user_info))

    print(f"有效用户数量：{len(valid_users)}")
    print(f"使用的群组总数（来自全局映射）：{len(group_id_to_idx)}")

    if len(valid_users) == 0:
        raise ValueError("没有找到有效用户，请检查 raw_users 数据结构")

    user_id2idx = {user_id: idx for idx, (user_id, _) in enumerate(valid_users)}

    x_list = []
    edge_index_list = []
    user_ids = []
    target_labels = []

    n_users = len(valid_users)
    feat_dim = len(group_id_to_idx)

    for user_id, user_info in valid_users:
        group_feats = torch.zeros(feat_dim)
        for gid in user_info.get('groups', []):
            if gid in group_id_to_idx:
                group_feats[group_id_to_idx[gid]] = 1.0
        x_list.append(group_feats)
        user_ids.append(user_id2idx[user_id])

        # 多标签 target，0-1向量
        target_label = torch.zeros(n_users)
        for followee in user_info.get('following', []):
            if followee in user_id2idx:
                target_label[user_id2idx[followee]] = 1.0
        target_labels.append(target_label)

        for followee in user_info.get('following', []):
            if followee in user_id2idx:
                src = user_id2idx[user_id]
                dst = user_id2idx[followee]
                edge_index_list.append([src, dst])

    if len(x_list) == 0:
        raise ValueError("所有用户都无效，导致无法构建特征矩阵")

    x = torch.stack(x_list)
    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    user_ids = torch.tensor(user_ids, dtype=torch.long)
    target_labels = torch.stack(target_labels)  # [n_users, n_users] 大小的0/1矩阵
    batch = torch.zeros(x.size(0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)
    data.user_ids = user_ids
    data.target_labels = target_labels

    print(f"预处理完成：节点数 {x.size(0)}, 边数 {edge_index.size(1)}，label shape {target_labels.shape}")
    return data

