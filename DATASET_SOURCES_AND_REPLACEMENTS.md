Real dataset/code sources for CWE paper experiments

Source paper inspected: /Users/kunjrathod/Downloads/polished_arxiv_submission_fixed.tex
Target repo: /Users/kunjrathod/projects/causal-world-encoder-paper

1) LoCoMo (best match for Experiment 1)

Official repo:
- https://github.com/snap-research/locomo
- Raw dataset file: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
- README: https://raw.githubusercontent.com/snap-research/locomo/main/README.MD
- License: https://raw.githubusercontent.com/snap-research/locomo/main/LICENSE.txt

Verified facts:
- locomo10.json is directly accessible
- HEAD content-length: 632473 bytes
- Repo README states the dataset is in ./data/locomo10.json
- License is CC BY-NC 4.0
- Note from README: image URLs/captions are included, but images themselves are not released

Suggested commands:
- mkdir -p data/locomo
- curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o data/locomo/locomo10.json
- curl -L https://raw.githubusercontent.com/snap-research/locomo/main/LICENSE.txt -o data/locomo/LICENSE.txt

Feasibility:
- Excellent for overnight MacBook runs; file is tiny (<1 MB)
- Enough to reproduce the retrieval experiment in the draft with TF-IDF + MiniLM + hybrid fusion
- Caveat: non-commercial license

2) Real gridworld/action data route (best replacement for Experiments 3-5)

Preferred public dataset: Minari MiniGrid/BabyAI offline trajectories on Hugging Face
- Dataset namespace: https://huggingface.co/datasets/farama-minari/minigrid
- Example key/door dataset metadata:
  - https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-KeyCorridorS3R1/optimal-v0/data/metadata.json
  - https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-UnlockPickup/optimal-v0/data/metadata.json
- Example HDF5 trajectory files:
  - https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-KeyCorridorS3R1/optimal-v0/data/main_data.hdf5
  - https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-UnlockPickup/optimal-v0/data/main_data.hdf5

Verified facts:
- BabyAI-KeyCorridorS3R1 optimal-v0 metadata says 1000 episodes and 17971 steps
- BabyAI-UnlockPickup optimal-v0 metadata says 1000 episodes and 19764 steps
- HDF5 files are directly reachable and are about 30 MB each
- The dataset collection includes many key/door/unlock tasks, including KeyCorridor, UnlockLocal, UnlockPickup, BlockedUnlockPickup, OpenDoor, etc.

Suggested commands (direct download):
- mkdir -p data/minigrid/KeyCorridorS3R1
- curl -L https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-KeyCorridorS3R1/optimal-v0/data/main_data.hdf5 -o data/minigrid/KeyCorridorS3R1/main_data.hdf5
- curl -L https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-KeyCorridorS3R1/optimal-v0/data/metadata.json -o data/minigrid/KeyCorridorS3R1/metadata.json

Suggested commands (via Hugging Face Hub):
- python -m pip install huggingface_hub h5py numpy
- python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='farama-minari/minigrid', repo_type='dataset', filename='BabyAI-KeyCorridorS3R1/optimal-v0/data/main_data.hdf5', local_dir='data/minigrid')
hf_hub_download(repo_id='farama-minari/minigrid', repo_type='dataset', filename='BabyAI-KeyCorridorS3R1/optimal-v0/data/metadata.json', local_dir='data/minigrid')
PY

Why this is the best replacement:
- Real public trajectories, not hand-made mock transitions
- Closely matches key-door/unlock planning claims in the draft
- Supports exact action prediction, latent dynamics, local/global decode, and sequence-model comparisons
- Runs comfortably on an M1 Pro overnight because files are small and trajectory counts are modest

Recommended mapping to draft experiments:
- Exp 3: replace synthetic 5x5 custom key-door gridworld with BabyAI-KeyCorridorS3R1 or UnlockPickup held-out evaluation episodes
- Exp 4: train next-state embedding predictor from real offline trajectories instead of custom enumerated transitions
- Exp 5: train transformer vs recurrent proxy vs MLP on episode sequences from the same Minari task

3) Code route if you want to generate trajectories yourself instead of downloading them

MiniGrid:
- https://github.com/Farama-Foundation/Minigrid
- README: https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/README.md
- License: Apache-2.0

BabyAI expert/demo generation code:
- https://github.com/mila-iqia/babyai
- README: https://raw.githubusercontent.com/mila-iqia/babyai/master/README.md
- Demo script: https://raw.githubusercontent.com/mila-iqia/babyai/master/scripts/make_agent_demos.py
- Scripts README: https://raw.githubusercontent.com/mila-iqia/babyai/master/scripts/README.md
- License: BSD-3-Clause

Verified facts:
- BabyAI README says environments are now part of Minigrid
- BabyAI repo still contains scripts to produce demonstrations using the BabyAI bot
- make_agent_demos.py exposes --env, --model BOT, --episodes, --seed, etc.

Suggested commands:
- python -m pip install minigrid gymnasium blosc numpy
- git clone https://github.com/mila-iqia/babyai.git external/babyai
- python external/babyai/scripts/make_agent_demos.py --env BabyAI-KeyCorridorS3R1-v0 --model BOT --episodes 1000 --seed 0 --demos keycorridor_bot

Feasibility:
- Good fallback if you want a fully reproducible trajectory generator under your control
- Slightly more maintenance than using Minari/HF directly
- Still lightweight enough for laptop generation

4) Lightweight real text source for Experiment 2 replacement/workspace benchmark

Simplest real-data route: arXiv API for the cited papers already used in the manuscript
- API docs: https://info.arxiv.org/help/api/user-manual.html
- Query endpoint example: https://export.arxiv.org/api/query?search_query=id:2407.10671&start=0&max_results=1

Suggested commands:
- python -m pip install feedparser requests
- python - <<'PY'
import requests, feedparser
ids = [
    '2407.10671',  # Qwen2 Technical Report
    '2310.08560',  # MemGPT
    '2301.04104',  # DreamerV3
]
for aid in ids:
    url = f'https://export.arxiv.org/api/query?search_query=id:{aid}&start=0&max_results=1'
    txt = requests.get(url, timeout=30).text
    feed = feedparser.parse(txt)
    for e in feed.entries:
        print({'id': aid, 'title': e.title, 'summary': e.summary[:300]})
PY

Recommendation:
- Use 8 cited arXiv papers already present in the draft and fetch title/abstract live via API
- This is cleaner than storing copied abstracts manually, and it keeps the benchmark fully real
- Runtime is trivial on a MacBook

5) Models actually needed by the draft and confirmed public

Sentence embeddings:
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Confirmed via HF API; very widely downloaded; Apache-2.0

Open-weight planners:
- https://huggingface.co/distilbert/distilgpt2
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Suggested commands:
- python -m pip install torch transformers sentence-transformers accelerate
- python - <<'PY'
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
AutoTokenizer.from_pretrained('distilbert/distilgpt2')
AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('ok')
PY

Feasibility on M1 Pro:
- all-MiniLM-L6-v2: easy
- distilgpt2: easy
- Qwen2.5-0.5B-Instruct: feasible for inference/scoring on M1 Pro, but slower than distilgpt2; still realistic overnight

6) Optional harder world-model replacement if you want a richer environment than MiniGrid

Crafter:
- Code: https://github.com/danijar/crafter
- License: MIT-style permissive license in repo
- Human dataset listing: https://archive.org/details/crafter_human_dataset
- Direct dataset zip: https://archive.org/download/crafter_human_dataset/dataset.zip

Verified facts:
- archive.org metadata is accessible
- dataset.zip exists and is about 131969180 bytes (~126 MB)

Suggested commands:
- python -m pip install crafter
- mkdir -p data/crafter
- curl -L https://archive.org/download/crafter_human_dataset/dataset.zip -o data/crafter/dataset.zip
- unzip -q data/crafter/dataset.zip -d data/crafter/human

Feasibility:
- Good if you want a more convincing world-model setting than gridworld
- More ambitious than needed for an overnight M1 run, but still possible on subsets
- I would treat this as optional v2, not the first reproducibility target

Bottom-line recommendation

Use this stack for a clean, real-data, laptop-feasible paper:
1. Experiment 1: official LoCoMo locomo10.json from snap-research/locomo
2. Experiment 2: live arXiv abstracts fetched via arXiv API for the cited papers
3. Experiments 3-5: farama-minari/minigrid BabyAI-KeyCorridorS3R1 and/or BabyAI-UnlockPickup offline trajectories
4. Embeddings/planner models: all-MiniLM-L6-v2, distilgpt2, Qwen2.5-0.5B-Instruct

This removes dependence on mock data while staying realistic for an Apple M1 Pro overnight run.