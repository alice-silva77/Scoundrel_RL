# Mastering Scoundrel: Curriculum RL with Action Masking and Eligibility Traces

AE4350 Bio-inspired Intelligence — TU Delft, 2025/26 Q4  
M.A. de Matos Ferreira da Silva — 6539602

---

## Requirements

```
pip install stable-baselines3 sb3-contrib gymnasium numpy matplotlib
```

Python 3.10 or later. Training requires a CUDA-capable GPU for reasonable speed. Figure generation runs on CPU.

---

## File overview

### Core environment

| File | Description |
|---|---|
| `scoundrel.py` | Standalone Scoundrel card game logic (no RL dependencies) |
| `scoundrel_env.py` | Gymnasium environment wrapping the game. Implements `action_masks()` for MaskablePPO, VecNormalize-compatible observation, configurable reward function and curriculum support |

### Training scripts

All training scripts write output (logs, checkpoints, plots) to their own directory. The `RUN_CONFIG_NAME` or equivalent constant at the top of each file selects which run to execute.

| File | Description |
|---|---|
| `scoundrel_agent.py` | Trains the **baseline** agent (7-phase curriculum, `c_ent=0.10`). Saves a training log JSON and periodic checkpoints |
| `scoundrel_sensitivity.py` | Trains one **sensitivity run** at a time (phases 0–4, 42-card deck). Change `RUN_CONFIG_NAME` at the top to select: `ent_low`, `ent_med`, `ent_high`, `lr_low`, `lr_high`, `no_kwb`, `no_cc` |
| `scoundrel_ace_extension.py` | Continues the `ent_low` checkpoint through **phases 5–6** (full 44-card deck). Produces the final `ent_low_ace` agent |

### Analysis and figure generation

Run these locally after downloading the training logs and the final checkpoint.

| File | Description |
|---|---|
| `sensitivity_analysis.py` | Reads all `training_log_*.json` files and prints a summary table of results across sensitivity runs |
| `gen_sensitivity_figs.py` | Generates the entropy, learning rate, and ablation sensitivity figures from training logs |
| `gen_entlow_full.py` | Generates the full curriculum comparison figure (baseline vs. `ent_low_ace` across all 7 phases) from training logs |
| `gen_solution_analysis.py` | Loads the final checkpoint and runs **2000 evaluation episodes**, writing per-step behavioural data to `solution_analysis_data.json` |
| `gen_solution_figures.py` | Reads `solution_analysis_data.json` and generates the 4-panel behavioural analysis figure |

### Data

| File | Description |
|---|---|
| `training_log_baseline.json` | Baseline agent training log (`c_ent=0.10`, 88 M steps, phases 0–5) |
| `training_log_ent_low.json` | Sensitivity run: `c_ent=0.01` |
| `training_log_ent_med.json` | Sensitivity run: `c_ent=0.05` (reference run) |
| `training_log_ent_high.json` | Sensitivity run: `c_ent=0.20` |
| `training_log_lr_low.json` | Sensitivity run: `α=1×10⁻⁵` |
| `training_log_lr_high.json` | Sensitivity run: `α=2×10⁻⁴` |
| `training_log_no_kwb.json` | Ablation: king-weapon bonus disabled |
| `training_log_no_cc.json` | Ablation: card counting disabled |
| `training_log_ent_low_ace.json` | Full curriculum log for `ent_low_ace` (phases 5–6, 35 M steps) |
| `checkpoint_ent_low_ace_35000000_steps.zip` | Final trained agent (MaskablePPO, 35 M steps into phase 6) |
| `checkpoint_ent_low_ace_vecnormalize_35000000_steps.pkl` | Companion VecNormalize statistics for the final checkpoint |

---

## How to reproduce the figures

### Sensitivity and curriculum figures (from training logs only)

```bash
python sensitivity_analysis.py        # prints summary table
python gen_sensitivity_figs.py        # fig_sensitivity_entropy.png, fig_sensitivity_lr.png, fig_sensitivity_ablations.png
python gen_entlow_full.py             # fig_entlow_full.png
```

### Behavioural analysis figures (requires the checkpoint)

```bash
# Step 1: evaluate the agent (takes a few minutes, writes solution_analysis_data.json)
python gen_solution_analysis.py

# Step 2: generate figures from the evaluation data
python gen_solution_figures.py        # fig_solution_analysis.png, fig_solution_card_counting.png
```

All scripts read and write from their own directory; no path configuration is needed when running locally.
