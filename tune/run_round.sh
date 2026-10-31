#!/bin/sh
# Realistic-benchmark round, end to end. Offline; touches no audio devices.
# Each line is one reproducible experiment; all results append to tune/results.jsonl.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python

# data (deterministic; regenerate any time; the large files are deleted after the round)
$PY augment.py --decode          # MUSDB18 -> data/xsim_audio.npy (16 kHz int16, ~2.3 GB) + data/xsim_index.npz
$PY augment.py --speech          # LibriSpeech pool -> data/xsim_speech_{audio,offs}.npy
$PY augment.py --build           # realistic set, same chunks as climb's clean full set -> data/realistic_signals.npz

# step 1: old GBM (best_config.json) on clean and realistic dev, clean- and realistic-trained, seeds 0-2
$PY gbm_real.py

# step 2: end-to-end cross-similarity model (5 OOF folds + full model per run, ~14 min each)
for s in 0 1 2; do $PY xsim.py --variant main --seed $s; done   # 350 steps x 128 per training
$PY xsim.py --variant noxsim --seed 0
$PY xsim.py --variant weakbleed --seed 0
$PY xsim.py --variant w3 --seed 0

# step 3: holdout, once, best model only (+ the GBM reference)
# for s in 0 1 2; do $PY xsim.py --variant main --seed $s --holdout; done
# $PY gbm_real.py --final

$PY xsim.py --examples
/usr/bin/python3 figs.py         # needs matplotlib (system python); writes tune/figs/*.png
