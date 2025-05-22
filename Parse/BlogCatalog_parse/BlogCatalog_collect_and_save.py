import torch
import os
import random
from collections import defaultdict, Counter
from BlogCatalog_parse import read_data
from BlogCatalog_build import preprocess_social_graph


def extract_global_group_map(user_dict):
    all_groups = set()
    for info in user_dict.values():
        all_groups.update(info.get('groups', []))
    return {gid: idx for idx, gid in enumerate(sorted(all_groups))}


def compute_group_to_user_map(user_dict):
    group_to_users = defaultdict(set)
    for uid, info in user_dict.items():
        for g in info.get('groups', []):
            group_to_users[g].add(uid)
    return group_to_users


def assign_groups_to_clients(group_to_users, n_clients=3, groups_per_client=20, top_k=40, seed=42):
    all_groups = sorted(group_to_users.items(), key=lambda kv: len(kv[1]), reverse=True)
    candidate_groups = [g for g, _ in all_groups[:top_k]]
    random.seed(seed)
    random.shuffle(candidate_groups)

    group_clusters = [candidate_groups[i::n_clients] for i in range(n_clients)]
    client_dominant_groups = [cluster[:groups_per_client] for cluster in group_clusters]
    return client_dominant_groups


def split_client_data_by_groups(user_dict, dominant_groups, alpha=0.7, test_ratio=0.3, seed=42):
    random.seed(seed)
    major_users = set()
    other_users = set()

    for uid, info in user_dict.items():
        groups = info.get('groups', [])
        if any(g in dominant_groups for g in groups):
            major_users.add(uid)
        else:
            other_users.add(uid)

    major_users = list(major_users)
    other_users = list(other_users)
    random.shuffle(major_users)
    random.shuffle(other_users)

    n_total = len(major_users) + len(other_users)
    n_major = int(alpha * n_total)
    n_other = int((1 - alpha) * n_total)

    selected_users = major_users[:n_major] + other_users[:n_other]

    random.shuffle(selected_users)
    n_test = int(test_ratio * len(selected_users))
    test_users = set(selected_users[:n_test])
    train_users = set(selected_users[n_test:])

    return train_users, test_users


def save_multi_subgraph_clients(user_dict, client_dominant_groups, save_dir, group_id_to_idx, alpha=0.7, test_ratio=0.3, edge_drop_ratio=0.3):
    os.makedirs(save_dir, exist_ok=True)

    for cid, dom_groups in enumerate(client_dominant_groups):
        seed = 42 + cid
        random.seed(seed)
        torch.manual_seed(seed)

        train_users, test_users = split_client_data_by_groups(
            user_dict, dom_groups, alpha=alpha, test_ratio=test_ratio, seed=seed
        )

        selected_users = train_users.union(test_users)

        # 构建子图字典：只保留这些用户及其内部边
        sub_user_dict = {}
        for uid in selected_users:
            info = user_dict[uid]
            sub_following = [f for f in info.get('following', []) if f in selected_users]
            sub_user_dict[uid] = {
                'groups': info.get('groups', []),
                'following': sub_following
            }

        # 构建 PYG 格式
        data = preprocess_social_graph(sub_user_dict, group_id_to_idx)

        # 创建 train/test mask
        uid_to_idx = {uid: i for i, uid in enumerate(data.user_ids.tolist())}
        total_nodes = data.num_nodes

        train_mask = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask = torch.zeros(total_nodes, dtype=torch.bool)

        for uid in train_users:
            if uid in uid_to_idx:
                train_mask[uid_to_idx[uid]] = True
        for uid in test_users:
            if uid in uid_to_idx:
                test_mask[uid_to_idx[uid]] = True

        # 随机丢弃内部边
        edge_index = data.edge_index
        n_edges = edge_index.size(1)
        n_keep = int((1 - edge_drop_ratio) * n_edges)
        perm = torch.randperm(n_edges)[:n_keep]
        data.edge_index = edge_index[:, perm]

        # 精简 target_labels，只保留有效列（当前 train/test 节点中至少被关注一次的）
        used_node_mask = train_mask | test_mask
        target_labels = data.target_labels  # shape [N, N]
        used_targets = target_labels[used_node_mask]

        # 保存
        data.train_mask = train_mask
        data.test_mask = test_mask

        path = os.path.join(save_dir, f'client{cid}.pt')
        torch.save(data, path)

        print(f"[Client {cid}] Saved to {path}")
        print(f"  Dominant Groups: {dom_groups}")
        print(f"  Train: {train_mask.sum().item()}, Test: {test_mask.sum().item()}")
        print(f"  Kept Edges: {data.edge_index.size(1)} / {n_edges} (after dropping {edge_drop_ratio * 100:.0f}%)")


def main():
    edge_file = '../../Dataset/BlogCatalog/BlogCatalog-dataset/data/edges.csv'
    node_file = '../../Dataset/BlogCatalog/BlogCatalog-dataset/data/group-edges.csv'
    save_dir = '../../Parsed_dataset/BlogCatalog'

    user_dict = read_data(edge_file, node_file)
    group_id_to_idx = extract_global_group_map(user_dict)
    group_to_users = compute_group_to_user_map(user_dict)

    n_clients = 3
    client_dominant_groups = assign_groups_to_clients(
        group_to_users,
        n_clients=n_clients,
        groups_per_client=20,
        top_k=39,
        seed=42
    )

    save_multi_subgraph_clients(
        user_dict,
        client_dominant_groups,
        save_dir,
        group_id_to_idx,
        alpha=0.7,
        test_ratio=0.2,
        edge_drop_ratio=0.3
    )


if __name__ == "__main__":
    main()
