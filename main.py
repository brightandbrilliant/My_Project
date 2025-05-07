import argparse
import torch
from torch_geometric.loader import DataLoader
from Train.FedAVGTrainer import FedAvgTrainer
from Train.Train import Trainer
from Clients.Client import Client


# ===== 构建 Clients =====
def build_clients(n_clients, feature_dim, client_user_counts, device='cpu'):
    clients = {}
    for cid in range(n_clients):
        client = Client(
            node_feat_dim=feature_dim,
            node_hidden_dim=32,
            node_embed_dim=64,
            graph_hidden_dim=32,
            graph_style_dim=64,
            fusion_output_dim=128,
            node_num_layers=3,
            graph_num_layers=3,
            dropout=0.1,
            n_clients=n_clients - 1,
            n_users=client_user_counts[cid]  # 为每个 client 单独设置 n_users
        ).to(device)
        client.create_optimizer(lr=5e-3, weight_decay=1e-5)
        clients[cid] = client
    return clients


# ===== 主函数入口 =====
def main():
    parser = argparse.ArgumentParser(description="Federated Graph Learning Trainer")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_clients', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=39)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--local_steps', type=int, default=1)
    parser.add_argument('--total_rounds', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Step 1：获取每个 client 的最大 user_id + 1
    client_user_counts = []
    for cid in range(args.n_clients):
        data_path = f'./Parsed_dataset/BlogCatalog/client{cid}.pt'
        data = torch.load(data_path)

        if not hasattr(data, 'user_ids'):
            raise ValueError(f"Client {cid} 的数据缺少 `user_ids` 属性")

        max_uid = int(data.user_ids.max().item())
        client_user_counts.append(max_uid + 1)

    # Step 2：初始化每个 client
    clients = build_clients(
        n_clients=args.n_clients,
        feature_dim=args.feature_dim,
        client_user_counts=client_user_counts,
        device=args.device
    )

    # Step 3：构建每个 client 的 DataLoader
    train_loaders = {}
    for cid in range(args.n_clients):
        data_path = f'./Parsed_dataset/BlogCatalog/client{cid}.pt'
        data = torch.load(data_path)

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        if not hasattr(data, 'train_mask'):
            raise ValueError(f"Client {cid} 的数据缺少 `train_mask` 属性")

        # user_ids 应为本地节点的真实 ID
        if not hasattr(data, 'user_ids'):
            raise ValueError(f"Client {cid} 的数据缺少 `user_ids` 属性")
        user_ids = data.user_ids

        # 构造 full_x：将已有 user 特征填入完整维度中
        full_x = torch.zeros((client_user_counts[cid], args.feature_dim), dtype=data.x.dtype)
        full_x[user_ids] = data.x
        data.x = full_x

        # 构造 dataset 和 loader
        dataset = [data]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        train_loaders[cid] = loader

    # Step 4：创建 Trainer 并开始训练

    trainer = Trainer(
        clients=clients,
        train_loaders=train_loaders,
        device=args.device,
        local_steps=args.local_steps,
        total_rounds=args.total_rounds,
        alpha=args.alpha,
        save_every=200,
        checkpoint_dir='Check_new'
    )
    """
    trainer = FedAvgTrainer(
        clients=clients,
        train_loaders=train_loaders,
        device=args.device,
        local_steps=args.local_steps,
        total_rounds=args.total_rounds,
        checkpoint_dir='Check_new_FedAvg',
        save_every=100
    )
    """

    # trainer.train(resume_round=0, load_checkpoint=False)
    trainer.train()


if __name__ == "__main__":
    main()





