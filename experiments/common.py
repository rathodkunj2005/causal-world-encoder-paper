import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
JSON_DIR = RESULTS_DIR / 'json'

ACTION_NAMES = {
    0: 'turn left',
    1: 'turn right',
    2: 'move forward',
    3: 'pick up object',
    4: 'drop object',
    5: 'toggle object',
    6: 'done',
}


def ensure_dirs():
    for p in [RESULTS_DIR, FIGURES_DIR, JSON_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def cosine_np(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
