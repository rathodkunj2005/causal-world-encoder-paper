import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import FIGURES_DIR, JSON_DIR, ensure_dirs, set_seed
from cwe_components import MLP, TwinStreamPredictor
from data_utils import load_minigrid_transitions, load_sentence_model
from run_real_experiments import action_embedding_matrix, batched_indices, local_decode_stats, make_transition_arrays


def train_models(train, action_embs, device, epochs=25):
    Xs, Xa, Y, _ = make_transition_arrays(train, action_embs)
    state_dim = Xs.shape[1]
    action_dim = Xa.shape[1]
    baseline = MLP(state_dim + action_dim, state_dim, hidden=256).to(device)
    twin = TwinStreamPredictor(state_dim, action_dim, hidden=256).to(device)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    opt_t = torch.optim.Adam(twin.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in batched_indices(len(Xs), batch_size=256, shuffle=True):
            s = torch.tensor(Xs[idx], device=device)
            a = torch.tensor(Xa[idx], device=device)
            y = torch.tensor(Y[idx], device=device)
            opt_b.zero_grad()
            pred = baseline(torch.cat([s, a], dim=-1))
            loss = F.mse_loss(pred, y) + (1.0 - F.cosine_similarity(pred, y, dim=-1).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
            opt_b.step()
            opt_t.zero_grad()
            tloss, _ = twin.losses(s, a, y)
            tloss.backward()
            torch.nn.utils.clip_grad_norm_(twin.parameters(), 1.0)
            opt_t.step()
    return baseline, twin


def evaluate(model, twin_mode, trs, action_embs, device):
    s, a, y, meta = make_transition_arrays(trs, action_embs)
    with torch.no_grad():
        s_t = torch.tensor(s, device=device)
        a_t = torch.tensor(a, device=device)
        if twin_mode:
            pred = model(s_t, a_t)[0].cpu().numpy()
        else:
            pred = model(torch.cat([s_t, a_t], dim=-1)).cpu().numpy()
    return local_decode_stats(pred, meta)


def main():
    ensure_dirs()
    set_seed(7)
    device = torch.device('cpu')
    sent = load_sentence_model()
    action_embs = action_embedding_matrix(sent)
    transitions, episodes = load_minigrid_transitions()
    train_eps = set(ep[0].episode_id for ep in episodes[:800])
    test_eps = set(ep[0].episode_id for ep in episodes[800:])
    splits = [
        {'name': 'pickup_toggle', 'held_out': {3, 5}},
        {'name': 'turning', 'held_out': {0, 1}},
        {'name': 'forward_drop', 'held_out': {2, 4}},
    ]
    all_results = {}
    for split in splits:
        held = split['held_out']
        seen = {0, 1, 2, 3, 4, 5} - held
        train = [t for t in transitions if t.episode_id in train_eps and t.action_id in seen]
        test_seen = [t for t in transitions if t.episode_id in test_eps and t.action_id in seen]
        test_held = [t for t in transitions if t.episode_id in test_eps and t.action_id in held]
        baseline, twin = train_models(train, action_embs, device)
        res = {
            'seen_actions': sorted(list(seen)),
            'held_out_actions': sorted(list(held)),
            'train_n': len(train),
            'test_seen_n': len(test_seen),
            'test_held_n': len(test_held),
            'single_stream_seen': evaluate(baseline, False, test_seen, action_embs, device),
            'single_stream_held_out': evaluate(baseline, False, test_held, action_embs, device),
            'twin_stream_seen': evaluate(twin, True, test_seen, action_embs, device),
            'twin_stream_held_out': evaluate(twin, True, test_held, action_embs, device),
        }
        res['heldout_cosine_gain'] = res['twin_stream_held_out']['cosine'] - res['single_stream_held_out']['cosine']
        res['heldout_top1_gain'] = res['twin_stream_held_out']['local_top1'] - res['single_stream_held_out']['local_top1']
        all_results[split['name']] = res

    with open(JSON_DIR / 'exp6_ticp_followup.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    labels = list(all_results.keys())
    cos = [all_results[k]['heldout_cosine_gain'] for k in labels]
    top1 = [all_results[k]['heldout_top1_gain'] for k in labels]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, cos, width=w, label='Held-out cosine gain')
    ax.bar(x + w/2, top1, width=w, label='Held-out Top-1 gain')
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(['pickup+toggle', 'turning', 'forward+drop'])
    ax.set_title('Twin-stream follow-up across held-out action splits')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp6_ticp_followup.png', dpi=180)
    plt.close(fig)
    print('wrote', JSON_DIR / 'exp6_ticp_followup.json')


if __name__ == '__main__':
    main()
