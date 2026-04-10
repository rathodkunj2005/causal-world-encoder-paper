import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from common import ACTION_NAMES, DATA_DIR


@dataclass
class Transition:
    dataset: str
    episode_id: str
    step: int
    state: np.ndarray
    action_id: int
    action_text: str
    next_state: np.ndarray
    state_text: str
    next_state_text: str
    mission: str


def load_locomo():
    path = DATA_DIR / 'locomo' / 'locomo10.json'
    with open(path) as f:
        raw = json.load(f)
    memories = []
    questions = []
    for sample in raw:
        dia_to_idx = {}
        for k, v in sample['conversation'].items():
            if not k.startswith('session_') or k.endswith('date_time'):
                continue
            for turn in v:
                text = turn['text']
                if 'blip_caption' in turn:
                    text = f"{text} [image: {turn['blip_caption']}]"
                mem = {
                    'sample_id': sample['sample_id'],
                    'dia_id': turn['dia_id'],
                    'text': text,
                    'speaker': turn['speaker'],
                }
                dia_to_idx[turn['dia_id']] = len(memories)
                memories.append(mem)
        for qa in sample['qa']:
            q = dict(qa)
            q['sample_id'] = sample['sample_id']
            q['memory_indices'] = [dia_to_idx[e] for e in qa['evidence'] if e in dia_to_idx]
            questions.append(q)
    return memories, questions


def load_sentence_model(name='sentence-transformers/all-MiniLM-L6-v2'):
    return SentenceTransformer(name)


def flatten_state(image, direction, mission):
    image = np.asarray(image, dtype=np.float32).reshape(-1) / 10.0
    direction_oh = np.zeros(4, dtype=np.float32)
    direction_oh[int(direction)] = 1.0
    return np.concatenate([image, direction_oh], axis=0)


def state_to_text(image, direction, mission):
    cells = []
    arr = np.asarray(image)
    for r in range(arr.shape[0]):
        row = []
        for c in range(arr.shape[1]):
            obj_idx, color_idx, state_idx = map(int, arr[r, c])
            row.append(f'{obj_idx}:{color_idx}:{state_idx}')
        cells.append(' '.join(row))
    grid = ' | '.join(cells)
    if isinstance(mission, bytes):
        mission = mission.decode('utf-8')
    return f'mission={mission}; direction={int(direction)}; grid={grid}'


def load_minigrid_transitions(limit_episodes_per_file=None):
    files = [
        ('KeyCorridorS3R1', DATA_DIR / 'minigrid' / 'keycorridor_main_data.hdf5'),
        ('UnlockPickup', DATA_DIR / 'minigrid' / 'unlockpickup_main_data.hdf5'),
    ]
    transitions = []
    episodes = []
    for dataset_name, path in files:
        with h5py.File(path, 'r') as f:
            episode_keys = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))
            if limit_episodes_per_file is not None:
                episode_keys = episode_keys[:limit_episodes_per_file]
            for ep_key in episode_keys:
                ep = f[ep_key]
                image = ep['observations']['image'][:]
                direction = ep['observations']['direction'][:]
                mission = ep['observations']['mission'][:]
                actions = ep['actions'][:]
                ep_transitions = []
                for t, act in enumerate(actions):
                    state = flatten_state(image[t], direction[t], mission[t])
                    next_state = flatten_state(image[t + 1], direction[t + 1], mission[t + 1])
                    state_text = state_to_text(image[t], direction[t], mission[t])
                    next_state_text = state_to_text(image[t + 1], direction[t + 1], mission[t + 1])
                    tr = Transition(
                        dataset=dataset_name,
                        episode_id=f'{dataset_name}:{ep_key}',
                        step=t,
                        state=state,
                        action_id=int(act),
                        action_text=ACTION_NAMES.get(int(act), f'action_{int(act)}'),
                        next_state=next_state,
                        state_text=state_text,
                        next_state_text=next_state_text,
                        mission=mission[t].decode('utf-8') if isinstance(mission[t], bytes) else str(mission[t]),
                    )
                    transitions.append(tr)
                    ep_transitions.append(tr)
                episodes.append(ep_transitions)
    return transitions, episodes


def fetch_arxiv_entries(arxiv_ids):
    entries = []
    for aid in arxiv_ids:
        url = f'https://export.arxiv.org/api/query?search_query=id:{aid}&start=0&max_results=1'
        text = requests.get(url, timeout=30).text
        root = ET.fromstring(text)
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        entry = root.find('a:entry', ns)
        if entry is None:
            raise RuntimeError(f'No arXiv entry found for {aid}')
        title = ' '.join(entry.findtext('a:title', default='', namespaces=ns).split())
        summary = ' '.join(entry.findtext('a:summary', default='', namespaces=ns).split())
        entries.append({'id': aid, 'title': title, 'summary': summary})
    return entries


def build_tfidf(memories):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform([m['text'] for m in memories])
    return vectorizer, matrix
