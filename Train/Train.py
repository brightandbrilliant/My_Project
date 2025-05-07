import torch
import os
from torch_geometric.loader import DataLoader
from Clients.Client import Client
from Communication.P2P import P2PCommunicator
from typing import Dict
import matplotlib.pyplot as plt
import time

def to_device(data_dict, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}


class Trainer:
    def __init__(self, clients: Dict[int, Client], train_loaders: Dict[int, DataLoader], device, local_steps=1,
                 total_rounds=10, alpha=0.5, save_every=1, checkpoint_dir='Checkpoints', clean_old=True):
        self.clients = clients
        self.train_loaders = train_loaders
        self.device = device
        self.local_steps = local_steps
        self.total_rounds = total_rounds
        self.alpha = alpha
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.clean_old = clean_old

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.communicators = {
            cid: P2PCommunicator(client_id=cid, total_clients=list(clients.keys()))
            for cid in clients
        }

        self.network = {cid: [] for cid in clients}
        self.client_losses = {cid: [] for cid in clients}

    def train(self, resume_round=0, load_checkpoint=False):
        if load_checkpoint and resume_round > 0:
            self.load_checkpoint(resume_round)
            print(f"[Trainer] Loaded checkpoint from round {resume_round}.")

        for round in range(resume_round, self.total_rounds):
            print(f"\n=== Federated Round {round + 1} ===")

            for cid, client in self.clients.items():
                communicator = self.communicators[cid]
                dataloader = self.train_loaders[cid]

                local_losses = []

                for local_step in range(self.local_steps):
                    for batch in dataloader:
                        batch = batch.to(self.device)

                        # --- 加入保护 ---
                        if not hasattr(batch, 'user_ids') or not hasattr(batch, 'target_labels'):
                            print(f"[Warning] Batch missing user_ids or target_labels at Client {cid}. Skipping.")
                            continue
                        if batch.user_ids.numel() == 0 or batch.target_labels.numel() == 0:
                            print(f"[Warning] Empty user_ids or target_labels at Client {cid}. Skipping.")
                            continue

                        train_mask = batch.train_mask if hasattr(batch, 'train_mask') else None
                        user_ids = batch.user_ids.to(self.device)
                        target_labels = batch.target_labels.to(self.device)
                        """
                        try:
                            print(
                                f"[Debug] Client {cid} target_labels min: {target_labels.min().item()}, max: {target_labels.max().item()}")
                        except Exception as e:
                            print(f"[Warning] Cannot print target_labels stats for Client {cid}: {e}")
                        """
                        external_dict = communicator.organize_external_node_embeds(
                            received_packets=self.network[cid],
                            user_ids=user_ids,
                            embed_dim=client.node_encoder.embed_dim,
                            device=self.device
                        )

                        loss, _ = client.forward(
                            data=batch,
                            user_ids=user_ids,
                            target_labels=target_labels,
                            alpha=self.alpha,
                            external_node_embeds_dict=external_dict,
                            mask=train_mask  # ✅ 新增
                        )

                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"[Error] Loss is NaN or Inf at Client {cid}, skipping this batch update.")
                            continue

                        loss.backward()
                        client.optimizer.step()
                        client.optimizer.zero_grad()

                        local_losses.append(loss.item())

                if local_losses:
                    avg_loss = sum(local_losses) / len(local_losses)
                else:
                    avg_loss = 0.0
                    print(f"[Warning] Client {cid} has no batch at Round {round + 1}. Setting avg_loss=0.")

                self.client_losses[cid].append(avg_loss)
                print(f"[Trainer] Round {round + 1} Client {cid} Loss: {avg_loss:.6f}")

                # 广播本地嵌入
                with torch.no_grad():
                    for batch in dataloader:
                        batch = batch.to(self.device)

                        if not hasattr(batch, 'user_ids') or batch.user_ids.numel() == 0:
                            print(f"[Warning] Missing or empty user_ids during broadcast at Client {cid}. Skipping.")
                            continue

                        user_ids = batch.user_ids.to(self.device)

                        local_user_embeds = client.node_encoder(batch)

                        if torch.isnan(local_user_embeds).any() or torch.isinf(local_user_embeds).any():
                            print(
                                f"[Error] Found NaN or Inf in local_user_embeds during broadcast at Client {cid}. Skipping this batch.")
                            continue

                        package = communicator.pack_user_embeddings(user_ids, local_user_embeds)
                        communicator.send_to_all_peers(package, self.network)

            self.network = {cid: [] for cid in self.clients}

            if (round + 1) % self.save_every == 0:
                self.save_checkpoint(round + 1)

        self.plot_losses()

    def save_checkpoint(self, round):
        # if self.clean_old:
         #    self._clear_checkpoints()

        for cid, client in self.clients.items():
            save_path = os.path.join(self.checkpoint_dir, f'client_{cid}_round_{round}.pth')
            torch.save(client.state_dict(), save_path)
        print(f"[Checkpoint] Saved models at round {round}.")

    def load_checkpoint(self, round):
        for cid, client in self.clients.items():
            load_path = os.path.join(self.checkpoint_dir, f'client_{cid}_round_{round}.pth')
            if os.path.exists(load_path):
                state_dict = torch.load(load_path, map_location=self.device)
                client.load_state_dict(state_dict)
                print(f"[Checkpoint] Loaded client {cid} model from round {round}.")
            else:
                print(f"[Checkpoint] Warning: No checkpoint found for client {cid} at round {round}.")

    def _clear_checkpoints(self):
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                os.remove(filepath)
        print(f"[Checkpoint] Cleared old checkpoints.")

    def plot_losses(self, save_dir='loss_plots'):
        os.makedirs(save_dir, exist_ok=True)

        for cid, losses in self.client_losses.items():
            plt.figure()
            plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Client {cid}')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.title(f'Client {cid} Loss Over Rounds')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'client_{cid}_loss_curve.png'))
            plt.close()

        print(f"[Plot] Saved all client loss curves to '{save_dir}'.")


