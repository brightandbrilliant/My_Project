import torch

def debug_client_data(client_id, feature_dim):
    data_path = f'./Parsed_dataset/BlogCatalog/client{client_id}.pt'
    data = torch.load(data_path)

    print(f"\n=== Debugging Client {client_id} Data ===")
    print(f"Total Nodes: {data.num_nodes}")
    print(f"Feature Shape: {data.x.shape}")  # [num_nodes, feature_dim]
    if hasattr(data, 'target_labels'):
        print(f"Label Shape: {data.target_labels.shape}")  # [num_nodes, num_labels] for multi-label
    else:
        print("⚠️ No `target_labels` in data")

    if hasattr(data, 'train_mask'):
        print(f"Train Mask: {data.train_mask.shape}, #True: {data.train_mask.sum().item()}")
    else:
        print("⚠️ No train_mask")

    if hasattr(data, 'user_ids'):
        print(f"user_ids: {data.user_ids.shape}")
        print(f"user_ids min: {data.user_ids.min().item()}, max: {data.user_ids.max().item()}")
    else:
        print("⚠️ No user_ids")

    # 如果你想模拟 client.forward() 的输入，打印 logits 和 labels 的大小
    print("\n=== Simulate Forward Pass Shape Check ===")
    train_mask = data.train_mask.bool()
    labels = data.target_labels[train_mask]

    # 假设 logits 是模型输出（你可以改为真实 client.forward 输出）
    num_classes = data.target_labels.shape[1]
    logits = torch.randn((train_mask.sum().item(), num_classes))

    print(f"Labels shape: {labels.shape}")
    print(f"Logits shape: {logits.shape}")

    if labels.shape != logits.shape:
        print("❌ Size mismatch between logits and target labels")
    else:
        print("✅ Logits and labels shapes match!")


if __name__ == "__main__":
    debug_client_data(client_id=0, feature_dim=39)  # 可以改 client_id 来切换调试目标
