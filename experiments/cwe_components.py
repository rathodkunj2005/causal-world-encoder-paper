import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalTimescaleBeliefState(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_slots=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.cells = nn.ModuleList([nn.GRUCell(input_dim, hidden_size) for _ in range(num_slots)])

    def forward(self, seq):
        # seq: [B, T, D]
        bsz, T, dim = seq.shape
        device = seq.device
        states = [torch.zeros(bsz, self.hidden_size, device=device) for _ in range(self.num_slots)]
        history = []
        for t in range(T):
            history.append(seq[:, t])
            for k, cell in enumerate(self.cells):
                period = 2 ** k
                if (t + 1) % period == 0:
                    start = max(0, t + 1 - period)
                    pooled = torch.stack(history[start:t + 1], dim=1).mean(dim=1)
                    states[k] = cell(pooled, states[k])
        return torch.cat(states, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TwinStreamPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.f_dynamics = MLP(state_dim, state_dim, hidden)
        self.f_intervention = MLP(state_dim + action_dim, state_dim, hidden)

    def forward(self, state, action):
        z_world = self.f_dynamics(state)
        z_delta = self.f_intervention(torch.cat([state, action], dim=-1))
        return z_world + z_delta, z_world, z_delta

    def losses(self, state, action, next_state):
        pred, z_world, z_delta = self.forward(state, action)
        residual_target = next_state - z_world.detach()
        total = 1.0 - F.cosine_similarity(pred, next_state, dim=-1).mean()
        world = 1.0 - F.cosine_similarity(z_world, next_state, dim=-1).mean()
        residual = F.mse_loss(z_delta, residual_target)
        return total + 0.3 * world + 0.3 * residual, {
            'total_cosine_loss': float(total.detach().cpu()),
            'world_cosine_loss': float(world.detach().cpu()),
            'residual_mse': float(residual.detach().cpu()),
        }

    @torch.no_grad()
    def predict_counterfactual(self, state, action_a, action_b):
        _, _, delta_a = self.forward(state, action_a)
        _, _, delta_b = self.forward(state, action_b)
        return delta_a - delta_b


class SparsePredictiveCommitmentGate(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, target_rate=0.3):
        super().__init__()
        self.predictor = MLP(state_dim + action_dim, state_dim, hidden)
        self.gate = nn.Linear(state_dim, 1)
        self.target_rate = target_rate

    def forward(self, state, action):
        gate_prob = torch.sigmoid(self.gate(state))
        pred = self.predictor(torch.cat([state, action], dim=-1))
        committed = gate_prob * pred + (1.0 - gate_prob) * state
        return committed, gate_prob.squeeze(-1)

    def loss(self, state, action, next_state):
        committed, gate_prob = self.forward(state, action)
        pred_loss = F.mse_loss(committed, next_state)
        budget_loss = F.mse_loss(gate_prob.mean(), torch.tensor(self.target_rate, device=state.device))
        return pred_loss + 0.5 * budget_loss, {
            'pred_loss': float(pred_loss.detach().cpu()),
            'budget_loss': float(budget_loss.detach().cpu()),
            'mean_gate': float(gate_prob.mean().detach().cpu()),
        }


class SequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128, out_dim=151):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return self.head(out[:, -1])


class CausalTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=4, num_layers=2, out_dim=151):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, seq):
        x = self.proj(seq)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        x = self.encoder(x, mask=mask)
        return self.head(x[:, -1])


class HTBSPredictor(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_slots=4, out_dim=151):
        super().__init__()
        self.htbs = HierarchicalTimescaleBeliefState(input_dim, hidden_size, num_slots)
        self.head = nn.Linear(hidden_size * num_slots, out_dim)

    def forward(self, seq):
        return self.head(self.htbs(seq))


class MLPControl(nn.Module):
    def __init__(self, input_dim, out_dim=151, hidden=256):
        super().__init__()
        self.mlp = MLP(input_dim, out_dim, hidden)

    def forward(self, seq):
        return self.mlp(seq[:, -1])
