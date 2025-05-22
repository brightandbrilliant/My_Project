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
    remaining_users = set(user_dict.keys()) - set(selected_users)

    random.shuffle(selected_users)
    n_test = int(test_ratio * len(selected_users))
    test_users = set(selected_users[:n_test])
    train_users = set(selected_users[n_test:])

    return train_users, test_users, remaining_users


def save_multi_clients_with_masks(user_dict, client_dominant_groups, save_dir, group_id_to_idx, alpha=0.7, test_ratio=0.3, edge_drop_ratio=0.3):
    os.makedirs(save_dir, exist_ok=True)
    original_data = preprocess_social_graph(user_dict, group_id_to_idx)
    uid_to_idx = {uid: i for i, uid in enumerate(original_data.user_ids.tolist())}
    total_nodes = original_data.num_nodes

    for cid, dom_groups in enumerate(client_dominant_groups):
        seed = 42 + cid
        random.seed(seed)
        torch.manual_seed(seed)

        train_users, test_users, invalid_users = split_client_data_by_groups(
            user_dict, dom_groups, alpha=alpha, test_ratio=test_ratio, seed=seed
        )

        train_mask = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask = torch.zeros(total_nodes, dtype=torch.bool)
        invalid_mask = torch.zeros(total_nodes, dtype=torch.bool)

        for uid in train_users:
            if uid in uid_to_idx:
                train_mask[uid_to_idx[uid]] = True
        for uid in test_users:
            if uid in uid_to_idx:
                test_mask[uid_to_idx[uid]] = True
        for uid in invalid_users:
            if uid in uid_to_idx:
                invalid_mask[uid_to_idx[uid]] = True

        # --- Step 1: 仅保留有效节点 ---
        valid_node_indices = torch.where(train_mask | test_mask)[0]
        valid_node_set = set(valid_node_indices.tolist())

        # --- Step 2: 提取内部边 ---
        edge_index = original_data.edge_index
        src, dst = edge_index
        mask_valid_edge = [
            i for i in range(src.size(0))
            if src[i].item() in valid_node_set and dst[i].item() in valid_node_set
        ]
        edge_index_valid = edge_index[:, mask_valid_edge]

        # --- Step 3: 随机删边 ---
        n_edges = edge_index_valid.size(1)
        n_keep = int((1 - edge_drop_ratio) * n_edges)
        perm = torch.randperm(n_edges)[:n_keep]
        edge_index_valid = edge_index_valid[:, perm]

        # --- Step 4: 构造新 data 对象 ---
        client_data = original_data.clone()
        client_data.edge_index = edge_index_valid
        client_data.train_mask = train_mask
        client_data.test_mask = test_mask
        client_data.invalid_mask = invalid_mask

        # --- Step 5: 精简 target_labels，只保留有效列 ---
        used_node_mask = train_mask | test_mask
        target_labels = client_data.target_labels  # shape [N, C]
        used_targets = target_labels[used_node_mask]  # shape [N_used, C]
        valid_target_cols = (used_targets.sum(dim=0) > 0)  # shape [C]

        client_data.target_labels = target_labels[:, valid_target_cols]  # 删除无效标签列

        path = os.path.join(save_dir, f'client{cid}.pt')
        torch.save(client_data, path)

        print(f"[Client {cid}] Saved to {path}")
        print(f"  Dominant Groups: {dom_groups}")
        print(f"  Train: {train_mask.sum().item()}, Test: {test_mask.sum().item()}, Invalid: {invalid_mask.sum().item()}")
        print(f"  Kept Edges: {edge_index_valid.size(1)} / {len(mask_valid_edge)} (after dropping {edge_drop_ratio * 100:.0f}%)")

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

    save_multi_clients_with_masks(
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
