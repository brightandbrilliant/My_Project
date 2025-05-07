# Train/FedAvgTrainer.py

import os
import copy
import torch


class FedAvgTrainer:
    def __init__(self, clients, train_loaders, device, local_steps, total_rounds,
                 checkpoint_dir='Checkpoints', save_every=1):
        self.clients = clients
        self.train_loaders = train_loaders
        self.device = device
        self.local_steps = local_steps
        self.total_rounds = total_rounds
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self):
        for round in range(self.total_rounds):
            print(f"\n--- Round {round + 1} ---")
            local_weights = []

            # 1. 每个 client 本地训练
            for cid, client in self.clients.items():
                dataloader = self.train_loaders[cid]
                local_losses = []

                for local_step in range(self.local_steps):
                    for batch in dataloader:
                        batch = batch.to(self.device)

                        if not hasattr(batch, 'user_ids') or not hasattr(batch, 'target_labels'):
                            print(f"[Warning] Batch missing user_ids or target_labels at Client {cid}. Skipping.")
                            continue
                        if batch.user_ids.numel() == 0 or batch.target_labels.numel() == 0:
                            print(f"[Warning] Empty user_ids or target_labels at Client {cid}. Skipping.")
                            continue

                        train_mask = batch.train_mask if hasattr(batch, 'train_mask') else None
                        user_ids = batch.user_ids.to(self.device)
                        target_labels = batch.target_labels.to(self.device)

                        dummy_external_dict = None
                        dummy_alpha = 0.0  # 可配成 self.alpha if needed

                        loss, _ = client.forward(
                            data=batch,
                            user_ids=user_ids,
                            target_labels=target_labels,
                            alpha=dummy_alpha,
                            external_node_embeds_dict=dummy_external_dict,
                            mask=train_mask
                        )

                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"[Error] Loss is NaN or Inf at Client {cid}, skipping this batch update.")
                            continue

                        loss.backward()
                        client.optimizer.step()
                        client.optimizer.zero_grad()
                        local_losses.append(loss.item())

                # 保存本地模型参数副本
                local_weights.append(copy.deepcopy(client.state_dict()))

            # 2. 聚合：参数平均
            global_state_dict = self.average_weights(local_weights)

            # 3. 更新所有 client 为全局模型
            for client in self.clients.values():
                self.safe_load_state_dict(client, global_state_dict)

            # 4. 保存 checkpoint
            if (round + 1) % self.save_every == 0:
                self.save_checkpoint(global_state_dict, round + 1)

    def average_weights(self, local_weights):
        avg_weights = copy.deepcopy(local_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights)):
                if avg_weights[key].shape != local_weights[i][key].shape:
                    continue
                avg_weights[key] += local_weights[i][key]
            avg_weights[key] = avg_weights[key] / len(local_weights)
        return avg_weights

    def save_checkpoint(self, state_dict, round_num):
        path = os.path.join(self.checkpoint_dir, f'global_model_round_{round_num}.pt')
        torch.save(state_dict, path)
        print(f"Checkpoint saved at round {round_num}: {path}")

    def safe_load_state_dict(self, model, state_dict):
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"[Skip] Skipping loading {k} due to shape mismatch. "
                      f"Model: {model_dict.get(k, None).shape if k in model_dict else 'Missing'}, "
                      f"Checkpoint: {v.shape}")
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)