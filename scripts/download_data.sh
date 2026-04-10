#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/locomo" "$ROOT/data/minigrid"

curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o "$ROOT/data/locomo/locomo10.json"
curl -L https://raw.githubusercontent.com/snap-research/locomo/main/LICENSE.txt -o "$ROOT/data/locomo/LICENSE.txt"

curl -L https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-KeyCorridorS3R1/optimal-v0/data/main_data.hdf5 -o "$ROOT/data/minigrid/keycorridor_main_data.hdf5"
curl -L https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-KeyCorridorS3R1/optimal-v0/data/metadata.json -o "$ROOT/data/minigrid/keycorridor_metadata.json"

curl -L https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-UnlockPickup/optimal-v0/data/main_data.hdf5 -o "$ROOT/data/minigrid/unlockpickup_main_data.hdf5"
curl -L https://huggingface.co/datasets/farama-minari/minigrid/resolve/main/BabyAI-UnlockPickup/optimal-v0/data/metadata.json -o "$ROOT/data/minigrid/unlockpickup_metadata.json"

echo "Data downloaded under $ROOT/data"
