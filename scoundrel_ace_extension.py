"""
scoundrel_ace_extension.py

Continue the ent_low agent through curriculum phases 5 and 6:
  Phase 5: 43-card deck (A♣ only,  clubs=14, spades=13), threshold 0.12
  Phase 6: 44-card deck (both Aces, clubs=14, spades=14), no threshold — final

Loads the latest checkpoint_ent_low_* file found in SCRIPT_DIR.
Logs to training_log_ent_low_ace.json and plots to training_curves_ent_low_ace.png.
"""

import glob
import json
import re
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scoundrel_env import ScoundrelEnv


SCRIPT_DIR = (
    Path("/kaggle/working")
    if Path("/kaggle/working").exists()
    else Path(__file__).parent
)

RUN_NAME = "ent_low_ace"
LOG_FILE  = str(SCRIPT_DIR / f"training_log_{RUN_NAME}.json")
PLOT_FILE = str(SCRIPT_DIR / f"training_curves_{RUN_NAME}.png")
PLOT_EVERY = 5

ACE_PHASES = [
    ({"clubs": 14, "spades": 13}, 0.30),  # phase 5: A♣ only
    ({"clubs": 14, "spades": 14}, None),   # phase 6: full 44-card deck — final
]

config = {
    "card_counting":           True,
    "heal_scale":              1.0,
    "damage_scale":            1.0,
    "room_reward":             0.1,
    "room_reward_scale":       0.15,
    "win_reward":              25.0,
    "win_hp_bonus":            15.0,
    "death_penalty":           3.0,
    "potion_waste_penalty":    0.5,
    "weapon_efficiency_bonus": 0.05,
    "king_weapon_bonus":       2.0,
    "n_envs":                  8,
    "timesteps":               45_000_000,
    "n_steps":                 1024,
    "ent_coef":                0.01,
    "gae_lambda":              0.95,
    "learning_rate":           5e-5,
    "n_epochs":                10,
    "net_arch":                [256, 256],
    "device":                  "cuda",
    "seed":                    0,
}


def _find_checkpoint() -> tuple[str, str]:
    """Return (model_zip, vecnorm_pkl) for the latest ent_low checkpoint."""
    zips = sorted(
        glob.glob(str(SCRIPT_DIR / "checkpoint_ent_low_*_steps.zip")),
        key=lambda p: int(re.search(r"(\d+)_steps", p).group(1)),
    )
    if not zips:
        raise FileNotFoundError(
            "No checkpoint_ent_low_*_steps.zip found in " + str(SCRIPT_DIR) + ".\n"
            "Upload the ent_low checkpoint files to /kaggle/working/ first."
        )
    latest = zips[-1]
    steps  = int(re.search(r"(\d+)_steps", latest).group(1))
    vn     = str(SCRIPT_DIR / f"checkpoint_ent_low_vecnormalize_{steps}_steps.pkl")
    if not Path(vn).exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vn}")
    print(f"Loaded checkpoint at {steps:,} steps: {Path(latest).name}")
    return latest, vn


def _build_venv(clubs: int, spades: int) -> DummyVecEnv:
    def _make():
        return Monitor(ScoundrelEnv(
            card_counting           = config["card_counting"],
            max_monster_value       = clubs,
            max_spades_value        = spades,
            heal_scale              = config["heal_scale"],
            damage_scale            = config["damage_scale"],
            room_reward             = config["room_reward"],
            room_reward_scale       = config["room_reward_scale"],
            win_reward              = config["win_reward"],
            win_hp_bonus            = config["win_hp_bonus"],
            death_penalty           = config["death_penalty"],
            potion_waste_penalty    = config["potion_waste_penalty"],
            weapon_efficiency_bonus = config["weapon_efficiency_bonus"],
            king_weapon_bonus       = config["king_weapon_bonus"],
        ))
    return DummyVecEnv([_make for _ in range(config["n_envs"])])


class AceCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ace_phase  = 0   # 0 = phase 5 (A♣), 1 = phase 6 (both aces)
        self._transitions: list[int] = []
        self._ep_wins  = deque(maxlen=200)
        self._ep_hp    = deque(maxlen=200)
        self._ep_rooms = deque(maxlen=200)
        self._n_dumps  = 0
        self._last_ev  = float("nan")

        self.log = {
            "timesteps": [], "ep_rew_mean": [], "ep_len_mean": [],
            "explained_var": [], "value_loss": [], "entropy_loss": [],
            "approx_kl": [], "win_rate": [], "mean_final_hp": [],
            "mean_rooms_cleared": [], "curriculum_phase": [],
        }

        self.fig, self.axes = plt.subplots(2, 4, figsize=(16, 8))
        self.fig.suptitle("Scoundrel — ent_low ace extension", fontsize=13)
        plt.tight_layout(pad=3.0)

    def _on_training_start(self) -> None:
        orig = self.model.logger.dump
        def patched(step=None):
            self._capture()
            orig(step=step)
        self.model.logger.dump = patched

    def _capture(self) -> None:
        kv = self.model.logger.name_to_value
        if "rollout/ep_rew_mean" not in kv:
            return
        self._n_dumps += 1
        g   = lambda k: float(kv.get(k, float("nan")))
        wr  = float(np.mean(self._ep_wins))  if self._ep_wins  else float("nan")
        hp  = float(np.mean(self._ep_hp))    if self._ep_hp    else float("nan")
        rms = float(np.mean(self._ep_rooms)) if self._ep_rooms else float("nan")

        self.log["timesteps"].append(self.num_timesteps)
        self.log["ep_rew_mean"].append(g("rollout/ep_rew_mean"))
        self.log["ep_len_mean"].append(g("rollout/ep_len_mean"))
        self.log["explained_var"].append(self._last_ev)
        self.log["value_loss"].append(g("train/value_loss"))
        self.log["entropy_loss"].append(g("train/entropy_loss"))
        self.log["approx_kl"].append(g("train/approx_kl"))
        self.log["win_rate"].append(wr)
        self.log["mean_final_hp"].append(hp)
        self.log["mean_rooms_cleared"].append(rms)
        self.log["curriculum_phase"].append(float(self._ace_phase + 5))

        with open(LOG_FILE, "w") as f:
            json.dump(self.log, f, indent=2)

        if not np.isnan(wr):
            self._maybe_advance(wr)

        if self._n_dumps % PLOT_EVERY == 0:
            self._update_plot()

    def _maybe_advance(self, wr: float) -> None:
        if self._ace_phase >= len(ACE_PHASES) - 1:
            return
        _, threshold = ACE_PHASES[self._ace_phase]
        if threshold is None or wr < threshold:
            return
        self._ace_phase += 1
        limits, _ = ACE_PHASES[self._ace_phase]
        self.training_env.env_method(
            "set_max_monster_value", limits["clubs"], limits["spades"]
        )
        self._transitions.append(self.num_timesteps)
        print(
            f"\n[CURRICULUM] Phase {self._ace_phase + 5} — "
            f"clubs={limits['clubs']}, spades={limits['spades']}  "
            f"(win_rate={wr:.1%} at step {self.num_timesteps:,})\n"
        )

    def _on_rollout_end(self) -> None:
        from stable_baselines3.common.utils import explained_variance
        rb = self.model.rollout_buffer
        self._last_ev = float(
            explained_variance(rb.values.flatten(), rb.returns.flatten())
        )

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []), self.locals.get("dones", [])
        ):
            if done:
                self._ep_wins.append(1 if info.get("victory", False) else 0)
                self._ep_hp.append(info.get("hp", 0))
                self._ep_rooms.append(info.get("rooms_cleared", 0))
        return True

    def _update_plot(self) -> None:
        specs = [
            ("ep_rew_mean",        "Mean Reward",             "royalblue"),
            ("ep_len_mean",        "Mean Episode Length",     "darkorange"),
            ("win_rate",           "Win Rate  (last 200)",    "limegreen"),
            ("mean_final_hp",      "Mean Final HP",           "crimson"),
            ("explained_var",      "Explained Variance",      "mediumpurple"),
            ("value_loss",         "Value Loss",              "saddlebrown"),
            ("entropy_loss",       "Entropy Loss",            "teal"),
            ("mean_rooms_cleared", "Rooms Cleared",           "olive"),
        ]
        ts = np.array(self.log["timesteps"])
        for ax, (key, title, color) in zip(self.axes.flatten(), specs):
            vals = np.array(self.log[key], dtype=float)
            ax.clear()
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(ts[:len(vals)][valid], vals[valid], color=color, linewidth=1.3)
            for pt in self._transitions:
                ax.axvline(pt, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel("Timesteps", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        self.fig.tight_layout(pad=2.5)
        self.fig.savefig(PLOT_FILE, dpi=120, bbox_inches="tight")

    def _on_training_end(self) -> None:
        self._update_plot()
        self.fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
        print(f"\nFinal plot saved → {PLOT_FILE}")


def main():
    model_path, vn_path = _find_checkpoint()

    start_limits = ACE_PHASES[0][0]
    env = VecNormalize.load(
        vn_path,
        _build_venv(start_limits["clubs"], start_limits["spades"]),
    )
    env.training = True
    env.norm_obs = True

    model = MaskablePPO.load(model_path, env=env, device=config["device"])
    model.ent_coef      = config["ent_coef"]
    model.learning_rate = config["learning_rate"]

    print(f"ent_coef      = {config['ent_coef']}")
    print(f"learning_rate = {config['learning_rate']}")
    print(f"timesteps     = {config['timesteps']:,}")
    print(f"Starting at ace phase 5: clubs=14, spades=13\n")

    ckpt_cb = CheckpointCallback(
        save_freq         = max(500_000 // config["n_envs"], 1),
        save_path         = str(SCRIPT_DIR),
        name_prefix       = f"checkpoint_{RUN_NAME}",
        save_vecnormalize = True,
        verbose           = 1,
    )

    model.learn(
        total_timesteps     = config["timesteps"],
        callback            = CallbackList([AceCallback(), ckpt_cb]),
        reset_num_timesteps = True,
    )

    model.save(str(SCRIPT_DIR / f"scoundrel_model_{RUN_NAME}"))
    env.save(str(SCRIPT_DIR / f"scoundrel_vecnormalize_{RUN_NAME}.pkl"))

    with open(SCRIPT_DIR / f"run_config_{RUN_NAME}.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved        → scoundrel_model_{RUN_NAME}.zip")
    print(f"VecNormalize stats → scoundrel_vecnormalize_{RUN_NAME}.pkl")
    print(f"Metrics saved      → {LOG_FILE}")


if __name__ == "__main__":
    main()
