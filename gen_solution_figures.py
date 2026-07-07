"""
gen_solution_figures.py

Generates two analysis figures from solution_analysis_data.json:
  - fig_solution_analysis.png   (2x2 panel)
  - fig_solution_card_counting.png (single panel)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
DATA_FILE  = SCRIPT_DIR / "solution_analysis_data.json"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
with open(DATA_FILE) as f:
    data = json.load(f)

meta     = data["meta"]
episodes = data["episodes"]
steps    = data["steps"]

print(f"Episodes : {meta['n_episodes']}")
print(f"Win rate : {meta['win_rate']:.1%}")
print(f"Steps    : {len(steps)}")

# ── Derived arrays ──────────────────────────────────────────────────────────────
ep_won    = {e["episode_id"]: e["won"]           for e in episodes}
ep_rooms  = np.array([e["rooms_cleared"]          for e in episodes])
ep_hp     = np.array([e["final_hp"]               for e in episodes])
ep_won_arr = np.array([e["won"]                   for e in episodes])

rooms_won  = ep_rooms[ep_won_arr]
rooms_lost = ep_rooms[~ep_won_arr]

# ── Figure 1: 2×2 analysis panel ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Learned Strategy Analysis — ent\\_low agent (full 44-card deck)",
             fontsize=12, fontweight="bold")

BLUE   = "#3a7ebf"
RED    = "#bf3a3a"
GREEN  = "#3abf5e"
ORANGE = "#bf8c3a"

# ── Panel A: Rooms cleared histogram ───────────────────────────────────────────
ax = axes[0, 0]
bins = np.arange(0, 16) - 0.5

ax.hist(rooms_lost, bins=bins, color=RED,  alpha=0.75, label=f"Lost  (n={len(rooms_lost)})",  edgecolor="white", linewidth=0.4)
ax.hist(rooms_won,  bins=bins, color=BLUE, alpha=0.90, label=f"Won   (n={len(rooms_won)})",   edgecolor="white", linewidth=0.4)
ax.axvline(14, color="black", linestyle="--", linewidth=1.2, alpha=0.6, label="Max (14 rooms)")

ax.set_xlabel("Rooms cleared", fontsize=10)
ax.set_ylabel("Episodes", fontsize=10)
ax.set_title("(A)  Episode outcome by rooms cleared", fontsize=10, fontweight="bold")
ax.set_xticks(range(0, 15))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

stats_text = (f"Mean (all): {ep_rooms.mean():.1f}\n"
              f"Win rate: {meta['win_rate']:.1%}")
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# ── Panel B: Weapon usage heatmap ──────────────────────────────────────────────
ax = axes[0, 1]

# Count times weapon chosen vs barehanded, indexed by (monster_val, weapon_val)
# Only steps where a monster was fought AND weapon was available
weapon_chosen   = defaultdict(int)
weapon_possible = defaultdict(int)

for s in steps:
    if s["action_type"] in ("weapon", "barehanded") and s["weapon_available"]:
        mv = s["monster_val"]
        wv = s["weapon_val"]
        if mv is None or wv is None or wv == 0:
            continue
        key = (mv, wv)
        weapon_possible[key] += 1
        if s["action_type"] == "weapon":
            weapon_chosen[key] += 1

# Build matrix: rows=weapon_val 2-14, cols=monster_val 2-14
vals = range(2, 15)
n    = len(list(vals))  # 13
mat  = np.full((n, n), np.nan)
cnt  = np.zeros((n, n))

for r_i, wv in enumerate(range(2, 15)):
    for c_i, mv in enumerate(range(2, 15)):
        key = (mv, wv)
        if weapon_possible[key] >= 5:
            mat[r_i, c_i] = weapon_chosen[key] / weapon_possible[key]
            cnt[r_i, c_i] = weapon_possible[key]

im = ax.imshow(mat, origin="lower", aspect="auto",
               vmin=0, vmax=1, cmap="RdYlGn")
plt.colorbar(im, ax=ax, label="Weapon usage rate")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([str(v) if v not in (11,12,13,14) else {11:'J',12:'Q',13:'K',14:'A'}[v]
                    for v in range(2,15)], fontsize=7)
ax.set_yticklabels([str(v) if v not in (11,12,13,14) else {11:'J',12:'Q',13:'K',14:'A'}[v]
                    for v in range(2,15)], fontsize=7)
ax.set_xlabel("Monster value", fontsize=10)
ax.set_ylabel("Equipped weapon value", fontsize=10)
ax.set_title("(B)  Weapon usage when available\n(green = always uses weapon)", fontsize=10, fontweight="bold")

# Add diagonal reference line: weapon_val == monster_val (break-even)
for i in range(n):
    for j in range(n):
        if not np.isnan(mat[i, j]) and cnt[i, j] >= 20:
            ax.text(j, i, f"{mat[i,j]:.0%}", ha="center", va="center",
                    fontsize=5.5, color="black" if 0.2 < mat[i,j] < 0.8 else "white")

# ── Panel C: HP at potion consumption ──────────────────────────────────────────
ax = axes[1, 0]

hp_at_potion = [s["hp"] for s in steps if s["action_type"] == "potion"]
hp_vals = range(0, 21)

counts_potion = [hp_at_potion.count(v) for v in hp_vals]
total_potion  = len(hp_at_potion)
frac_potion   = [c / total_potion if total_potion else 0 for c in counts_potion]

ax.bar(hp_vals, frac_potion, color=GREEN, alpha=0.8, edgecolor="white", linewidth=0.4)

# Reference: uniform distribution over 1-20 (agent always has hp >=1 when taking potion)
uniform_frac = 1 / 20
ax.axhline(uniform_frac, color="gray", linestyle="--", linewidth=1.2,
           label=f"Uniform baseline (1/{20})")

ax.set_xlabel("HP at moment of potion use", fontsize=10)
ax.set_ylabel("Fraction of all potion actions", fontsize=10)
ax.set_title("(C)  Potion timing (n=" + str(total_potion) + " events)", fontsize=10, fontweight="bold")
ax.set_xticks(range(0, 21, 2))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

mean_hp_pot = float(np.mean(hp_at_potion)) if hp_at_potion else 0
ax.text(0.97, 0.97, f"Mean HP: {mean_hp_pot:.1f}",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# ── Panel D: Flee rate heatmap ─────────────────────────────────────────────────
ax = axes[1, 1]

# When flee was possible (can_run=True, cards_played=0), did the agent flee?
# Group by (HP quartile, room_danger quartile)
flee_possible_steps = [s for s in steps
                        if s["action_type"] in ("flee", "barehanded", "weapon",
                                                 "potion", "new_weapon")
                        and s.get("rooms_cleared", 0) >= 0]

# More targeted: find steps where flee was a valid action (can_run checks not logged,
# so use proxy: action==8 means flee was valid at that step, and for non-flee steps
# with rooms_cleared changing. Instead, let's look at only steps where flee action
# is actually valid by checking action==8 or not in rooms where flee cannot happen.
# Since we logged all steps, we check: flee was valid if action==8 (confirmed)
# OR we can approximate by looking at all first-card-in-room decisions.
# Use cards_played==0 as proxy for "flee was potentially possible" (but not guaranteed).

# Build flee decision data: steps where cards_played=0 (flee might be possible)
# Action 8 = fled, other actions = chose not to flee (when flee was possible)
# This isn't perfect but gives a good heuristic.
# Better approach: use the action type directly.

# HP buckets: 1-5, 6-10, 11-15, 16-20
# Room danger buckets: 0-9, 10-14, 15-19, 20+
hp_bins     = [0, 5, 10, 15, 20]
danger_bins = [0, 9, 14, 19, 100]
hp_labels      = ["1-5", "6-10", "11-15", "16-20"]
danger_labels  = ["0-9", "10-14", "15-19", "20+"]

flee_matrix   = np.zeros((4, 4))
total_matrix  = np.zeros((4, 4))

for s in steps:
    if s["action_type"] not in ("flee", "barehanded", "weapon", "potion", "new_weapon"):
        continue
    hp     = s["hp"]
    danger = s["room_danger"]
    is_flee = s["action_type"] == "flee"

    # HP bucket
    hp_b = 0
    if hp <= 5:   hp_b = 0
    elif hp <= 10: hp_b = 1
    elif hp <= 15: hp_b = 2
    else:          hp_b = 3

    # Danger bucket
    if danger <= 9:    d_b = 0
    elif danger <= 14: d_b = 1
    elif danger <= 19: d_b = 2
    else:              d_b = 3

    total_matrix[hp_b, d_b] += 1
    if is_flee:
        flee_matrix[hp_b, d_b] += 1

with np.errstate(invalid="ignore"):
    flee_rate = np.where(total_matrix > 0, flee_matrix / total_matrix, np.nan)

im2 = ax.imshow(flee_rate, origin="lower", aspect="auto",
                vmin=0, vmax=0.3, cmap="Blues")
plt.colorbar(im2, ax=ax, label="Flee rate")

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(danger_labels, fontsize=8)
ax.set_yticklabels(hp_labels, fontsize=8)
ax.set_xlabel("Room danger (total monster HP)", fontsize=10)
ax.set_ylabel("Agent HP", fontsize=10)
ax.set_title("(D)  Flee rate by HP and room danger", fontsize=10, fontweight="bold")

for i in range(4):
    for j in range(4):
        if not np.isnan(flee_rate[i, j]) and total_matrix[i, j] >= 10:
            ax.text(j, i, f"{flee_rate[i,j]:.0%}",
                    ha="center", va="center", fontsize=9,
                    color="white" if flee_rate[i,j] > 0.15 else "black")

plt.tight_layout(pad=2.5)
out1 = SCRIPT_DIR / "fig_solution_analysis.png"
fig.savefig(str(out1), dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig)


# ── Figure 2: Card-counting exploitation ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
fig2.suptitle("Card-counting exploitation — weapon use on high-value monsters",
              fontsize=11, fontweight="bold")

# For monster values 11-14 (J, Q, K, A) where weapon management matters most
# Compare weapon usage rate when K/A is known to still be in deck vs already seen
high_vals = [11, 12, 13, 14]
val_labels = ["J (11)", "Q (12)", "K (13)", "A (14)"]

w_use_threat_in  = []   # king or ace still in deck
w_use_threat_out = []   # king and ace both already seen
n_threat_in  = []
n_threat_out = []

for mv in high_vals:
    threat_in_yes  = 0
    threat_in_tot  = 0
    threat_out_yes = 0
    threat_out_tot = 0

    for s in steps:
        if s["action_type"] not in ("weapon", "barehanded"):
            continue
        if not s["weapon_available"]:
            continue
        if s["monster_val"] != mv:
            continue
        threat_present = s["king_in_deck"] or s["ace_in_deck"]
        if threat_present:
            threat_in_tot += 1
            if s["action_type"] == "weapon":
                threat_in_yes += 1
        else:
            threat_out_tot += 1
            if s["action_type"] == "weapon":
                threat_out_yes += 1

    w_use_threat_in.append(threat_in_yes  / threat_in_tot  if threat_in_tot  >= 5 else np.nan)
    w_use_threat_out.append(threat_out_yes / threat_out_tot if threat_out_tot >= 5 else np.nan)
    n_threat_in.append(threat_in_tot)
    n_threat_out.append(threat_out_tot)

x     = np.arange(len(high_vals))
width = 0.35

bars1 = ax2.bar(x - width/2, w_use_threat_in,  width, color=RED,  alpha=0.8,
                label="K/A still in deck (threat present)",  edgecolor="white")
bars2 = ax2.bar(x + width/2, w_use_threat_out, width, color=BLUE, alpha=0.8,
                label="K/A already drawn (no threat)",       edgecolor="white")

ax2.set_xticks(x)
ax2.set_xticklabels(val_labels, fontsize=10)
ax2.set_xlabel("Monster value being fought", fontsize=11)
ax2.set_ylabel("Weapon usage rate (when available)", fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

# Annotate with counts
for i, (b1, b2) in enumerate(zip(bars1, bars2)):
    if not np.isnan(w_use_threat_in[i]):
        ax2.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02,
                 f"n={n_threat_in[i]}", ha="center", va="bottom", fontsize=7)
    if not np.isnan(w_use_threat_out[i]):
        ax2.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02,
                 f"n={n_threat_out[i]}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
out2 = SCRIPT_DIR / "fig_solution_card_counting.png"
fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("Done.")
