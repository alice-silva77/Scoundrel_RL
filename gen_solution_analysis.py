"""
gen_solution_analysis.py

Evaluates checkpoint_ent_low_ace_35000000_steps.zip on the full 44-card deck.
Logs per-step behavioral data and per-episode outcomes.
Saves solution_analysis_data.json for figure generation.
"""

import json
import sys
import math
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "checkpoint_ent_low_ace_35000000_steps.zip"
VN_PATH    = SCRIPT_DIR / "checkpoint_ent_low_ace_vecnormalize_35000000_steps.pkl"
OUT_JSON   = SCRIPT_DIR / "solution_analysis_data.json"
N_EPISODES = 2000

def main():
    # Imports here so errors are informative
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from scoundrel_env import ScoundrelEnv, Card

    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}")
    if not VN_PATH.exists():
        sys.exit(f"VecNormalize not found: {VN_PATH}")

    def make_env():
        return ScoundrelEnv(
            card_counting           = True,
            max_monster_value       = 14,
            max_spades_value        = 14,
            king_weapon_bonus       = 2.0,
            weapon_efficiency_bonus = 0.05,
            win_reward              = 25.0,
            win_hp_bonus            = 15.0,
            death_penalty           = 3.0,
            potion_waste_penalty    = 0.5,
        )

    raw_venv  = DummyVecEnv([make_env])
    eval_env  = VecNormalize.load(str(VN_PATH), raw_venv)
    eval_env.training    = False
    eval_env.norm_reward = False

    model = MaskablePPO.load(str(MODEL_PATH), env=eval_env, device="cpu")
    print(f"Loaded model  : {MODEL_PATH.name}")
    print(f"Loaded VecNorm: {VN_PATH.name}")
    print(f"Running {N_EPISODES} evaluation episodes ...\n")

    # Access raw ScoundrelEnv through VecNormalize → DummyVecEnv
    scoundrel_env = eval_env.venv.envs[0]

    ACE_CLUBS  = Card('C', 14)
    ACE_SPADES = Card('S', 14)
    KING_CLUBS  = Card('C', 13)
    KING_SPADES = Card('S', 13)

    steps_data    = []
    episodes_data = []

    wins = 0
    obs  = eval_env.reset()

    for ep_id in range(N_EPISODES):
        if ep_id > 0:
            obs = eval_env.reset()

        step_num = 0
        done     = False

        while not done:
            game = scoundrel_env.game

            # ── read raw game state before action ──────────────
            raw_hp               = int(game.hp)
            raw_weapon_val       = int(game.weapon_val)
            raw_weapon_constraint = int(game.weapon_constraint)
            raw_deck_remaining   = int(len(game.deck))
            raw_rooms_cleared    = int(game.rooms_cleared)
            raw_room             = list(game.room)
            raw_room_danger      = int(game.room_danger)
            raw_max_monster      = int(game.max_monster_in_room)
            raw_n_weapon_targets = int(game.n_weapon_targets)

            ace_in_deck  = (ACE_CLUBS  not in game.seen) or (ACE_SPADES  not in game.seen)
            king_in_deck = (KING_CLUBS not in game.seen) or (KING_SPADES not in game.seen)

            # ── get action masks and predict ───────────────────
            masks  = np.array(eval_env.env_method("action_masks"))  # (1, 9)
            action, _ = model.predict(obs, action_masks=masks, deterministic=False)
            action = int(action[0])

            # ── interpret action ───────────────────────────────
            weapon_available = raw_n_weapon_targets > 0

            if action == 8:
                action_type = "flee"
                monster_val = None
                card_type   = None
            elif action >= 4:
                card_idx    = action - 4
                card        = raw_room[card_idx] if card_idx < len(raw_room) else None
                action_type = "weapon"
                monster_val = int(card.value) if card else None
                card_type   = card.type if card else None
            else:
                card_idx = action
                if card_idx < len(raw_room):
                    card      = raw_room[card_idx]
                    card_type = card.type
                    if card.type == 'monster':
                        action_type = "barehanded"
                        monster_val = int(card.value)
                    elif card.type == 'potion':
                        action_type = "potion"
                        monster_val = None
                    else:
                        action_type = "new_weapon"
                        monster_val = None
                else:
                    action_type = "unknown"
                    monster_val = None
                    card_type   = None

            steps_data.append({
                "episode_id":          ep_id,
                "step_num":            step_num,
                "hp":                  raw_hp,
                "weapon_val":          raw_weapon_val,
                "weapon_constraint":   raw_weapon_constraint,
                "deck_remaining":      raw_deck_remaining,
                "rooms_cleared":       raw_rooms_cleared,
                "room_danger":         raw_room_danger,
                "max_monster_in_room": raw_max_monster,
                "action":              action,
                "action_type":         action_type,
                "card_type":           card_type,
                "monster_val":         monster_val,
                "weapon_available":    weapon_available,
                "ace_in_deck":         ace_in_deck,
                "king_in_deck":        king_in_deck,
            })

            obs, _, done_arr, info = eval_env.step([action])
            done = bool(done_arr[0])
            step_num += 1

        # ── episode complete ───────────────────────────────────
        terminal = info[0]
        won      = bool(terminal.get("victory", False))
        wins    += int(won)

        episodes_data.append({
            "episode_id":    ep_id,
            "won":           won,
            "rooms_cleared": int(terminal.get("rooms_cleared", 0)),
            "final_hp":      int(terminal.get("hp", 0)),
            "episode_length": step_num,
        })

        if (ep_id + 1) % 200 == 0:
            wr = wins / (ep_id + 1)
            print(f"  ep {ep_id+1:4d}/{N_EPISODES}  win rate so far: {wr:.1%}")

    win_rate   = wins / N_EPISODES
    mean_rooms = np.mean([e["rooms_cleared"] for e in episodes_data])
    mean_hp    = np.mean([e["final_hp"]      for e in episodes_data])
    mean_steps = np.mean([e["episode_length"] for e in episodes_data])

    print(f"\n-- Results ------------------------------------------")
    print(f"Win rate       : {win_rate:.1%}  ({wins}/{N_EPISODES})")
    print(f"Mean rooms     : {mean_rooms:.2f}")
    print(f"Mean final HP  : {mean_hp:.2f}")
    print(f"Mean steps/ep  : {mean_steps:.1f}")
    print(f"Total steps    : {len(steps_data)}")

    data = {
        "meta": {
            "n_episodes":  N_EPISODES,
            "win_rate":    win_rate,
            "mean_rooms":  float(mean_rooms),
            "mean_hp":     float(mean_hp),
            "mean_steps":  float(mean_steps),
        },
        "episodes": episodes_data,
        "steps":    steps_data,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(data, f)

    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
