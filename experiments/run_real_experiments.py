import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from common import ACTION_NAMES, FIGURES_DIR, JSON_DIR, ensure_dirs, get_device, save_json, set_seed
from cwe_components import CausalTransformer, HTBSPredictor, MLP, MLPControl, SequenceLSTM, SparsePredictiveCommitmentGate, TwinStreamPredictor
from data_utils import build_tfidf, fetch_arxiv_entries, load_locomo, load_minigrid_transitions, load_sentence_model


ARXIV_IDS = [
    '2402.17753',  # LoCoMo
    '2310.08560',  # MemGPT
    '2312.00752',  # Mamba
    '2402.19427',  # Griffin
    '2512.12818',  # Hindsight
    '2601.04688',  # ToolGate
    '2601.03204',  # InfiAgent
    '2506.09985',  # V-JEPA2
]


def rank_metrics(rankings, gold_indices):
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0
    ndcg = 0.0
    for ranked, gold in zip(rankings, gold_indices):
        gold = set(gold)
        for k in [1, 5, 10]:
            hits[k] += int(any(idx in gold for idx in ranked[:k]))
        first = None
        dcg = 0.0
        for i, idx in enumerate(ranked[:10]):
            if idx in gold:
                if first is None:
                    first = i + 1
                dcg += 1.0 / math.log2(i + 2)
        if first is not None:
            rr += 1.0 / first
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
        ndcg += (dcg / ideal) if ideal else 0.0
    n = len(rankings)
    return {
        'hits@1': hits[1] / n,
        'hits@5': hits[5] / n,
        'hits@10': hits[10] / n,
        'mrr': rr / n,
        'ndcg@10': ndcg / n,
        'n_questions': n,
    }


def run_retrieval_experiment(sentence_model):
    memories, questions = load_locomo()
    texts = [m['text'] for m in memories]
    vectorizer, tfidf = build_tfidf(memories)
    dense = sentence_model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)

    tfidf_rankings, dense_rankings, hybrid_rankings, gold = [], [], [], []
    query_texts = [q['question'] for q in questions]
    q_tfidf = vectorizer.transform(query_texts)
    q_dense = sentence_model.encode(query_texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)

    for i, q in enumerate(questions):
        tfidf_scores = (q_tfidf[i] @ tfidf.T).toarray().ravel()
        dense_scores = dense @ q_dense[i]
        tfidf_rank = np.argsort(-tfidf_scores)
        dense_rank = np.argsort(-dense_scores)
        rrf = defaultdict(float)
        for rank, idx in enumerate(tfidf_rank[:100]):
            rrf[int(idx)] += 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(dense_rank[:100]):
            rrf[int(idx)] += 1.0 / (60 + rank + 1)
        hybrid_rank = [idx for idx, _ in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)]
        tfidf_rankings.append(tfidf_rank[:10].tolist())
        dense_rankings.append(dense_rank[:10].tolist())
        hybrid_rankings.append(hybrid_rank[:10])
        gold.append(q['memory_indices'])

    metrics = {
        'tfidf': rank_metrics(tfidf_rankings, gold),
        'dense_minilm': rank_metrics(dense_rankings, gold),
        'hybrid_rrf': rank_metrics(hybrid_rankings, gold),
        'n_memories': len(memories),
    }
    save_json(JSON_DIR / 'exp1_retrieval.json', metrics)

    names = ['hits@1', 'hits@5', 'hits@10']
    methods = ['tfidf', 'dense_minilm', 'hybrid_rrf']
    vals = [[metrics[m][n] for n in names] for m in methods]
    x = np.arange(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, name in enumerate(names):
        ax.bar(x + (j - 1) * width, [v[j] for v in vals], width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(['TF-IDF', 'Dense', 'Hybrid'])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel('Score')
    ax.set_title('LoCoMo retrieval on real data')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp1_retrieval.png', dpi=180)
    plt.close(fig)
    return metrics


def extract_keywords(summary, top_k=8):
    words = []
    for tok in summary.lower().replace('-', ' ').replace(',', ' ').replace('.', ' ').split():
        tok = ''.join(ch for ch in tok if ch.isalnum())
        if len(tok) < 5:
            continue
        if tok in {'using', 'their', 'which', 'these', 'those', 'where', 'about', 'through'}:
            continue
        words.append(tok)
    ordered = []
    seen = set()
    for w in words:
        if w not in seen:
            ordered.append(w)
            seen.add(w)
        if len(ordered) >= top_k:
            break
    return ordered


def run_workspace_experiment():
    entries = fetch_arxiv_entries(ARXIV_IDS)
    transcript = []
    notes = []
    cumulative_transcript = 0
    cumulative_workspace = 0
    title_coverage = []
    keyword_coverage = []
    transcript_sizes = []
    workspace_sizes = []

    for i, entry in enumerate(entries, start=1):
        kws = extract_keywords(entry['summary'])
        note = {
            'id': entry['id'],
            'title': entry['title'],
            'keywords': kws,
            'claim': entry['summary'].split('. ')[0].strip(),
        }
        observation = f"TITLE: {entry['title']}\nABSTRACT: {entry['summary']}"
        action = f"WRITE_NOTE[{entry['id']}]"
        transcript.extend([observation, action])
        notes.append(note)

        transcript_prompt = '\n\n'.join(transcript)
        workspace_prompt = '\n'.join([
            'WORKSPACE SNAPSHOT',
            *[f"- {n['title']} | {', '.join(n['keywords'][:5])}" for n in notes],
            'RECENT ACTIONS',
            *transcript[-2:],
        ])
        cumulative_transcript += len(transcript_prompt)
        cumulative_workspace += len(workspace_prompt)
        transcript_sizes.append(len(transcript_prompt))
        workspace_sizes.append(len(workspace_prompt))
        title_coverage.append(int(all(n['title'] in workspace_prompt for n in notes)))
        keyword_coverage.append(int(all(all(k in workspace_prompt for k in n['keywords'][:3]) for n in notes)))

    metrics = {
        'n_documents': len(entries),
        'final_prompt_size': {'full_transcript': transcript_sizes[-1], 'bounded_workspace': workspace_sizes[-1]},
        'cumulative_prompt_load': {'full_transcript': cumulative_transcript, 'bounded_workspace': cumulative_workspace},
        'title_coverage_final': title_coverage[-1],
        'keyword_coverage_final': keyword_coverage[-1],
        'doc_ids': ARXIV_IDS,
    }
    save_json(JSON_DIR / 'exp2_workspace.json', metrics)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(entries) + 1), np.cumsum(transcript_sizes), label='Transcript')
    ax.plot(range(1, len(entries) + 1), np.cumsum(workspace_sizes), label='Bounded workspace')
    ax.set_xlabel('Workflow step')
    ax.set_ylabel('Cumulative prompt chars')
    ax.set_title('Prompt growth on real arXiv workflow')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp2_workspace.png', dpi=180)
    plt.close(fig)
    return metrics


def action_embedding_matrix(sentence_model):
    names = [ACTION_NAMES[i] for i in sorted(k for k in ACTION_NAMES if k <= 5)]
    vecs = sentence_model.encode(names, normalize_embeddings=True)
    return {i: vecs[i] for i in range(len(names))}


def train_batches(model, optimizer, batch_iter, loss_fn, device, epochs=20):
    model.train()
    for _ in range(epochs):
        for batch in batch_iter():
            optimizer.zero_grad()
            loss = loss_fn(model, batch, device)
            loss.backward()
            optimizer.step()


def make_transition_arrays(transitions, action_embs):
    Xs, Xa, Y, meta = [], [], [], []
    for tr in transitions:
        Xs.append(tr.state)
        Xa.append(action_embs[tr.action_id])
        Y.append(tr.next_state)
        meta.append(tr)
    return np.stack(Xs).astype(np.float32), np.stack(Xa).astype(np.float32), np.stack(Y).astype(np.float32), meta


def batched_indices(n, batch_size=256, shuffle=True):
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def local_decode_stats(preds, metas):
    top1 = 0
    top5 = 0
    cosines = []
    grouped = defaultdict(list)
    for m in metas:
        grouped[m.episode_id].append(m.next_state)
    for pred, m in zip(preds, metas):
        bank = np.stack(grouped[m.episode_id])
        sims = cosine_similarity(pred[None, :], bank)[0]
        order = np.argsort(-sims)
        target = None
        for i, cand in enumerate(bank):
            if np.array_equal(cand, m.next_state):
                target = i
                break
        if target is None:
            target = int(np.argmax(cosine_similarity(m.next_state[None, :], bank)[0]))
        top1 += int(order[0] == target)
        top5 += int(target in order[:5])
        cosines.append(float(cosine_similarity(pred[None, :], m.next_state[None, :])[0, 0]))
    n = len(metas)
    return {'cosine': float(np.mean(cosines)), 'local_top1': top1 / n, 'local_top5': top5 / n}


def run_intervention_experiment(sentence_model, device):
    transitions, episodes = load_minigrid_transitions()
    action_embs = action_embedding_matrix(sentence_model)
    seen_actions = {0, 1, 2, 4}
    held_out_actions = {3, 5}
    train_eps = set(ep[0].episode_id for ep in episodes[:800])
    test_eps = set(ep[0].episode_id for ep in episodes[800:])

    train = [t for t in transitions if t.episode_id in train_eps and t.action_id in seen_actions]
    test_seen = [t for t in transitions if t.episode_id in test_eps and t.action_id in seen_actions]
    test_held = [t for t in transitions if t.episode_id in test_eps and t.action_id in held_out_actions]

    Xs, Xa, Y, _ = make_transition_arrays(train, action_embs)
    state_dim = Xs.shape[1]
    action_dim = Xa.shape[1]

    baseline = MLP(state_dim + action_dim, state_dim, hidden=256).to(device)
    twin = TwinStreamPredictor(state_dim, action_dim, hidden=256).to(device)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    opt_t = torch.optim.Adam(twin.parameters(), lr=1e-3)

    def train_batch_iter():
        for idx in batched_indices(len(Xs), batch_size=256, shuffle=True):
            yield (
                torch.tensor(Xs[idx], device=device),
                torch.tensor(Xa[idx], device=device),
                torch.tensor(Y[idx], device=device),
            )

    for _ in range(25):
        for s, a, y in train_batch_iter():
            opt_b.zero_grad()
            pred = baseline(torch.cat([s, a], dim=-1))
            loss = F.mse_loss(pred, y) + (1.0 - F.cosine_similarity(pred, y, dim=-1).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
            opt_b.step()

            opt_t.zero_grad()
            t_loss, _ = twin.losses(s, a, y)
            t_loss.backward()
            torch.nn.utils.clip_grad_norm_(twin.parameters(), 1.0)
            opt_t.step()

    def evaluate(model, trs, twin_mode=False):
        s, a, y, meta = make_transition_arrays(trs, action_embs)
        with torch.no_grad():
            s_t = torch.tensor(s, device=device)
            a_t = torch.tensor(a, device=device)
            if twin_mode:
                pred = twin(s_t, a_t)[0].cpu().numpy()
            else:
                pred = baseline(torch.cat([s_t, a_t], dim=-1)).cpu().numpy()
        return local_decode_stats(pred, meta)

    results = {
        'train_transitions': len(train),
        'test_seen_transitions': len(test_seen),
        'test_held_transitions': len(test_held),
        'single_stream_seen': evaluate(baseline, test_seen, False),
        'single_stream_held_out': evaluate(baseline, test_held, False),
        'twin_stream_seen': evaluate(twin, test_seen, True),
        'twin_stream_held_out': evaluate(twin, test_held, True),
        'seen_actions': sorted(list(seen_actions)),
        'held_out_actions': sorted(list(held_out_actions)),
    }
    save_json(JSON_DIR / 'exp3_intervention.json', results)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['Seen cosine', 'Held-out cosine', 'Held-out Top-1']
    baseline_vals = [results['single_stream_seen']['cosine'], results['single_stream_held_out']['cosine'], results['single_stream_held_out']['local_top1']]
    twin_vals = [results['twin_stream_seen']['cosine'], results['twin_stream_held_out']['cosine'], results['twin_stream_held_out']['local_top1']]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, baseline_vals, width=w, label='Single-stream')
    ax.bar(x + w/2, twin_vals, width=w, label='Twin-stream')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title('Intervention decomposition on held-out actions')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp3_intervention.png', dpi=180)
    plt.close(fig)
    return results


def build_sequence_examples(episodes, action_embs, length, episode_ids=None):
    X, Y, meta = [], [], []
    for ep in episodes:
        if episode_ids is not None and ep[0].episode_id not in episode_ids:
            continue
        if len(ep) < length:
            continue
        for end in range(length - 1, len(ep)):
            window = ep[end - length + 1:end + 1]
            seq = [np.concatenate([tr.state, action_embs[tr.action_id]], axis=0) for tr in window]
            X.append(np.stack(seq))
            Y.append(window[-1].next_state)
            meta.append(window[-1])
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), meta


def train_sequence_model(model, X_train, Y_train, device, epochs=12):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    model.train()
    start = time.time()
    for _ in range(epochs):
        for idx in batched_indices(len(X_train), batch_size=128, shuffle=True):
            x = torch.tensor(X_train[idx], device=device)
            y = torch.tensor(Y_train[idx], device=device)
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y) + (1.0 - F.cosine_similarity(pred, y, dim=-1).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return time.time() - start


def eval_sequence_model(model, X_test, meta, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, device=device)).cpu().numpy()
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
    return local_decode_stats(pred, meta)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def run_timescale_experiment(sentence_model, device):
    transitions, episodes = load_minigrid_transitions()
    action_embs = action_embedding_matrix(sentence_model)
    train_ids = set(ep[0].episode_id for ep in episodes[:800])
    test_ids = set(ep[0].episode_id for ep in episodes[800:])
    lengths = [4, 8, 12, 16]
    results = {}
    for L in lengths:
        X_train, Y_train, _ = build_sequence_examples(episodes, action_embs, L, train_ids)
        X_test, Y_test, meta_test = build_sequence_examples(episodes, action_embs, L, test_ids)
        input_dim = X_train.shape[-1]
        out_dim = Y_train.shape[-1]
        models = {
            'htbs': HTBSPredictor(input_dim, hidden_size=64, num_slots=4, out_dim=out_dim),
            'lstm': SequenceLSTM(input_dim, hidden=128, out_dim=out_dim),
            'transformer': CausalTransformer(input_dim, d_model=128, num_heads=4, num_layers=2, out_dim=out_dim),
            'mlp_control': MLPControl(input_dim, out_dim=out_dim, hidden=256),
        }
        Lres = {}
        for name, model in models.items():
            train_s = train_sequence_model(model, X_train, Y_train, device, epochs=10)
            metrics = eval_sequence_model(model, X_test, meta_test, device)
            metrics['params'] = count_params(model)
            metrics['train_seconds'] = train_s
            Lres[name] = metrics
        results[str(L)] = {
            'train_examples': int(len(X_train)),
            'test_examples': int(len(X_test)),
            'models': Lres,
        }
    save_json(JSON_DIR / 'exp4_timescale.json', results)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name, label in [('htbs', 'HTBS'), ('lstm', 'LSTM'), ('transformer', 'Transformer'), ('mlp_control', 'MLP')]:
        xs = lengths
        ys = [results[str(L)]['models'][model_name]['local_top1'] for L in lengths]
        ax.plot(xs, ys, marker='o', label=label)
    ax.set_xlabel('Episode length')
    ax.set_ylabel('Local decode Top-1')
    ax.set_title('Timescale factorization vs flat alternatives')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp4_timescale.png', dpi=180)
    plt.close(fig)
    return results


def run_sparse_experiment(sentence_model, device):
    transitions, episodes = load_minigrid_transitions()
    action_embs = action_embedding_matrix(sentence_model)
    train_eps = set(ep[0].episode_id for ep in episodes[:800])
    test_eps = set(ep[0].episode_id for ep in episodes[800:])
    train = [t for t in transitions if t.episode_id in train_eps]
    test = [t for t in transitions if t.episode_id in test_eps]
    Xs, Xa, Y, _ = make_transition_arrays(train, action_embs)
    state_dim = Xs.shape[1]
    action_dim = Xa.shape[1]

    dense = MLP(state_dim + action_dim, state_dim, hidden=256).to(device)
    sparse = SparsePredictiveCommitmentGate(state_dim, action_dim, hidden=256, target_rate=0.3).to(device)
    opt_d = torch.optim.Adam(dense.parameters(), lr=1e-3)
    opt_s = torch.optim.Adam(sparse.parameters(), lr=1e-3)

    for _ in range(25):
        for idx in batched_indices(len(Xs), batch_size=256, shuffle=True):
            s = torch.tensor(Xs[idx], device=device)
            a = torch.tensor(Xa[idx], device=device)
            y = torch.tensor(Y[idx], device=device)

            opt_d.zero_grad()
            pred = dense(torch.cat([s, a], dim=-1))
            d_loss = F.mse_loss(pred, y)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(dense.parameters(), 1.0)
            opt_d.step()

            opt_s.zero_grad()
            s_loss, _ = sparse.loss(s, a, y)
            s_loss.backward()
            torch.nn.utils.clip_grad_norm_(sparse.parameters(), 1.0)
            opt_s.step()

    s_test, a_test, y_test, meta = make_transition_arrays(test, action_embs)
    with torch.no_grad():
        dense_pred = dense(torch.tensor(np.concatenate([s_test, a_test], axis=1), device=device)).cpu().numpy()
        sparse_pred, gates = sparse(torch.tensor(s_test, device=device), torch.tensor(a_test, device=device))
        sparse_pred = sparse_pred.cpu().numpy()
        gates = gates.cpu().numpy()
    dense_metrics = local_decode_stats(dense_pred, meta)
    sparse_metrics = local_decode_stats(sparse_pred, meta)

    changes = np.array([float(cosine_similarity(m.state[None, :], m.next_state[None, :])[0, 0]) for m in meta])
    high_mask = changes < 0.5
    low_mask = ~high_mask
    results = {
        'dense': dense_metrics,
        'sparse': sparse_metrics,
        'target_commitment_rate': 0.3,
        'actual_commitment_rate': float((gates > 0.5).mean()),
        'high_change_threshold': 0.5,
        'high_change_commitment_rate': float((gates[high_mask] > 0.5).mean()),
        'low_change_commitment_rate': float((gates[low_mask] > 0.5).mean()),
        'accuracy_per_commit_dense': float(dense_metrics['local_top1']),
        'accuracy_per_commit_sparse': float(sparse_metrics['local_top1'] / max((gates > 0.5).mean(), 1e-8)),
    }
    save_json(JSON_DIR / 'exp5_sparse.json', results)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = [results['high_change_commitment_rate'], results['low_change_commitment_rate'], results['actual_commitment_rate']]
    ax.bar(['High-change', 'Low-change', 'Overall'], bars)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Commitment rate')
    ax.set_title('Sparse commitment gate behavior')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp5_sparse.png', dpi=180)
    plt.close(fig)
    return results


def main():
    ensure_dirs()
    set_seed(7)
    device = torch.device('cpu')
    sentence_model = load_sentence_model()
    all_results = {
        'device': str(device),
        'exp1_retrieval': run_retrieval_experiment(sentence_model),
        'exp2_workspace': run_workspace_experiment(),
        'exp3_intervention': run_intervention_experiment(sentence_model, device),
        'exp4_timescale': run_timescale_experiment(sentence_model, device),
        'exp5_sparse': run_sparse_experiment(sentence_model, device),
    }
    save_json(JSON_DIR / 'all_results.json', all_results)
    print('Wrote results to', JSON_DIR / 'all_results.json')


if __name__ == '__main__':
    main()
