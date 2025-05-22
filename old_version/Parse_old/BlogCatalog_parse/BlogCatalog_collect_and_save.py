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

def assign_users_to_clients(user_dict, n_clients=3, dominant_ratio=0.75, groups_per_client=20, seed=42):
    random.seed(seed)
    group_to_users = defaultdict(set)
    for uid, info in user_dict.items():
        for g in info['groups']:
            group_to_users[g].add(uid)

    all_groups = list(group_to_users.keys())
    random.shuffle(all_groups)
    group_clusters = [all_groups[i::n_clients] for i in range(n_clients)]
    client_dominant_groups = [cluster[:groups_per_client] for cluster in group_clusters]

    client_train_users = [set() for _ in range(n_clients)]
    all_users = set(user_dict.keys())

    for cid in range(n_clients):
        dom_groups = client_dominant_groups[cid]
        dom_users = set()
        for g in dom_groups:
            dom_users.update(group_to_users[g])
        dom_users = list(dom_users)
        random.shuffle(dom_users)
        n_dom = int(dominant_ratio * len(dom_users))
        client_train_users[cid].update(dom_users[:n_dom])

        # 混入其他用户
        remaining = list(all_users - client_train_users[cid])
        random.shuffle(remaining)
        n_other = int((1 - dominant_ratio) * len(dom_users))
        client_train_users[cid].update(remaining[:n_other])

    return client_train_users, client_dominant_groups

def save_full_graph_with_masks(user_dict, client_train_users, save_dir, group_id_to_idx):
    os.makedirs(save_dir, exist_ok=True)
    full_data = preprocess_social_graph(user_dict, group_id_to_idx)
    uid_to_idx = {uid: i for i, uid in enumerate(full_data.user_ids.tolist())}
    total_nodes = full_data.num_nodes

    for cid, train_users in enumerate(client_train_users):
        train_mask = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask = torch.ones(total_nodes, dtype=torch.bool)  # 所有节点默认是 test

        for uid in train_users:
            if uid in uid_to_idx:
                idx = uid_to_idx[uid]
                train_mask[idx] = True
                test_mask[idx] = False  # train 节点不是 test

        client_data = full_data.clone()
        client_data.train_mask = train_mask
        client_data.test_mask = test_mask

        path = os.path.join(save_dir, f'client{cid}.pt')
        torch.save(client_data, path)
        print(f"[Saved] {path}")
        print(f"Train Samples: {train_mask.sum().item()} | Test Samples: {test_mask.sum().item()}")

def main():
    edge_file = '../../Dataset/BlogCatalog/BlogCatalog-dataset/data/edges.csv'
    node_file = '../../Dataset/BlogCatalog/BlogCatalog-dataset/data/group-edges.csv'
    save_dir = '../../Parsed_dataset/BlogCatalog'

    user_dict = read_data(edge_file, node_file)
    group_id_to_idx = extract_global_group_map(user_dict)
    client_train_users, dom_groups = assign_users_to_clients(user_dict)

    for cid in range(len(client_train_users)):
        print(f"\n=== Client {cid} ===")
        print(f"Dominant Groups: {dom_groups[cid]}")
        counter = Counter()
        for uid in client_train_users[cid]:
            for g in user_dict[uid]['groups']:
                counter[g] += 1
        print(f"Top 5 Groups: {counter.most_common(5)}")

    save_full_graph_with_masks(user_dict, client_train_users, save_dir, group_id_to_idx)


if __name__ == "__main__":
    main()
