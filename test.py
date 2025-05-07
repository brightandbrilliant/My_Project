import torch
from torch_geometric.loader import DataLoader
from Clients.Client import Client
from sklearn.metrics import precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

def load_model(checkpoint_path, model_args, device):
    model = Client(**model_args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, device, threshold=0.1):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # 获取 test_mask 指定的节点
            test_mask = batch.test_mask
            if test_mask.sum() == 0:
                continue  # 跳过没有测试节点的 batch

            node_embeds = model.node_encoder(batch)
            graph_embed = model.graph_encoder(batch)
            fused = model.fusion(node_embeds, graph_embed, batch.batch)
            logits = model.predictor(fused)

            probs = torch.sigmoid(logits)  # [num_nodes, n_users]
            preds = (probs > threshold).float()

            # 仅保留 test_mask 为 True 的节点
            preds = preds[test_mask]
            labels = batch.target_labels[test_mask]

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    if not all_preds:
        print("无有效 test_mask 节点可评估")
        return 0.0, 0.0

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)

    return precision, recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rounds = []
    precisions = []
    recalls = []

    # 加载测试数据（包含 test_mask）
    test_data = torch.load('./Parsed_dataset/BlogCatalog/client0.pt')
    test_dataset = [test_data]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取标签维度（num_users）
    n_users = test_data.target_labels.shape[1]

    model_args = dict(
        node_feat_dim=39,
        node_hidden_dim=64,
        node_embed_dim=128,
        graph_hidden_dim=64,
        graph_style_dim=128,
        fusion_output_dim=128,
        node_num_layers=3,
        graph_num_layers=3,
        dropout=0.4,
        n_clients=3,
        n_users=n_users
    )

    for i in range(1, 101):
        checkpoint_path = f'May_fifth/client_0_round_{10 * i}.pth'
        model = load_model(checkpoint_path, model_args, device)
        precision, recall = evaluate(model, test_loader, device)

        # 记录结果
        current_round = 10 * i  # 修正轮次计算
        rounds.append(current_round)
        precisions.append(precision)
        recalls.append(recall)
        # print(f"[Round {current_round}] Test Precision: {precision:.4f}, Recall: {recall:.4f}")

        # 绘制precision曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, precisions, 'b-o', linewidth=2, markersize=6)
    plt.title('Precision Trend over Training Rounds', fontsize=14)
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds[::5], rotation=45)  # 每5个轮次显示一个刻度
    plt.tight_layout()
    plt.savefig('precision_curve.png')
    plt.close()

    # 绘制recall曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, recalls, 'r-s', linewidth=2, markersize=6)
    plt.title('Recall Trend over Training Rounds', fontsize=14)
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds[::5], rotation=45)
    plt.tight_layout()
    plt.savefig('recall_curve.png')
    plt.close()


if __name__ == '__main__':
    main()


