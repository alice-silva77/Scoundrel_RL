import glob
import json
import re
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
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
LOG_FILE   = str(SCRIPT_DIR / "training_log.json")
PLOT_EVERY = 5   # save plot every N PPO iterations

# ── Experiment configuration (module-level so resume() can read it) ──────────
config = {
    # Environment
    "card_counting":      True,
    "max_clubs_value":    8,   # curriculum start — matches CURRICULUM_PHASES[0]
    "max_spades_value":   8,
    "heal_scale":         1.0,
    "damage_scale":       1.0,
    "room_reward":          0.1,
    "room_reward_scale":    0.15,
    "win_reward":           25.0,
    "win_hp_bonus":         15.0,
    "death_penalty":        3.0,
    "potion_waste_penalty": 0.5,
    "weapon_efficiency_bonus": 0.05,
    "king_weapon_bonus":       2.0,
    # Training
    "n_envs":        8,
    "timesteps":     100_000_000,
    "n_steps":       1024,
    "ent_coef":      0.10,
    "gae_lambda":    0.95,
    "learning_rate": 5e-5,
    "n_epochs":      10,
    "net_arch":      [256, 256],
    "device":        "cuda",
    "seed":          0,
}


class ScoundrelCallback(BaseCallback):

    def __init__(self, log_path=LOG_FILE, plot_every=PLOT_EVERY, verbose=0):
        super().__init__(verbose)
        self.log_path   = log_path
        self.plot_every = plot_every

        self.log = {
            "timesteps":    [],
            "ep_rew_mean":  [],
            "ep_len_mean":  [],
            "explained_var": [],
            "value_loss":   [],
            "entropy_loss": [],
            "approx_kl":    [],
            "win_rate":           [],
            "mean_final_hp":      [],
            "mean_rooms_cleared": [],
            "curriculum_phase":   [],
        }

        self._ep_wins   = deque(maxlen=200)
        self._ep_hp     = deque(maxlen=200)
        self._ep_rooms  = deque(maxlen=200)
        self._n_dumps = 0
        self._last_explained_var = float("nan")

        self._curriculum_phase = 0
        self._phase_transitions: list[int] = []

        self.fig, self.axes = plt.subplots(2, 4, figsize=(16, 8))
        self.fig.suptitle("Scoundrel — Training Metrics", fontsize=13)
        plt.tight_layout(pad=3.0)

    # Curriculum phases: ({"clubs": max_clubs, "spades": max_spades}, win_rate_threshold)
    # Phase 3 uses clubs=13, spades=12 so only K♣ is in the deck (one-king stepping stone).
    # Phase 4 introduces K♠ (both kings). Phase 5 is the full 44-card deck.
    CURRICULUM_PHASES = [
        ({"clubs":  8, "spades":  8}, 0.90),  # phase 0: 32-card deck
        ({"clubs": 11, "spades": 11}, 0.55),  # phase 1: 38-card deck
        ({"clubs": 12, "spades": 12}, 0.40),  # phase 2: 40-card deck
        ({"clubs": 13, "spades": 12}, 0.30),  # phase 3: 41-card deck, K♣ only
        ({"clubs": 13, "spades": 13}, 0.20),  # phase 4: 42-card deck, both kings
        ({"clubs": 14, "spades": 13}, 0.12),  # phase 5: 43-card deck, A♣ only
        ({"clubs": 14, "spades": 14}, None),  # phase 6: full 44-card deck — final
    ]

    def _maybe_advance_curriculum(self, win_rate: float) -> None:
        phase = self._curriculum_phase
        if phase >= len(self.CURRICULUM_PHASES) - 1:
            return
        _, threshold = self.CURRICULUM_PHASES[phase]
        if not np.isnan(win_rate) and win_rate >= threshold:
            self._curriculum_phase += 1
            new_limits, _ = self.CURRICULUM_PHASES[self._curriculum_phase]
            clubs  = new_limits["clubs"]
            spades = new_limits["spades"]
            self.training_env.env_method("set_max_monster_value", clubs, spades)
            self._phase_transitions.append(self.num_timesteps)
            print(f"\n[CURRICULUM] Phase {self._curriculum_phase} — "
                  f"max_clubs={clubs}, max_spades={spades}  "
                  f"(win_rate was {win_rate:.1%}  at step {self.num_timesteps:,})\n")

    def _on_training_start(self) -> None:
        original_dump = self.model.logger.dump

        def _patched_dump(step=None):
            self._capture_from_logger()
            original_dump(step=step)

        self.model.logger.dump = _patched_dump

    def _capture_from_logger(self) -> None:
        kv = self.model.logger.name_to_value

        if "rollout/ep_rew_mean" not in kv:
            return

        self._n_dumps += 1

        def g(k):
            return float(kv.get(k, float("nan")))

        self.log["timesteps"].append(self.num_timesteps)
        self.log["ep_rew_mean"].append(g("rollout/ep_rew_mean"))
        self.log["ep_len_mean"].append(g("rollout/ep_len_mean"))
        self.log["explained_var"].append(self._last_explained_var)
        self.log["value_loss"].append(g("train/value_loss"))
        self.log["entropy_loss"].append(g("train/entropy_loss"))
        self.log["approx_kl"].append(g("train/approx_kl"))
        self.log["win_rate"].append(
            float(np.mean(self._ep_wins)) if self._ep_wins else float("nan")
        )
        self.log["mean_final_hp"].append(
            float(np.mean(self._ep_hp)) if self._ep_hp else float("nan")
        )
        self.log["mean_rooms_cleared"].append(
            float(np.mean(self._ep_rooms)) if self._ep_rooms else float("nan")
        )
        self.log["curriculum_phase"].append(float(self._curriculum_phase))

        win_rate = self.log["win_rate"][-1]
        self._maybe_advance_curriculum(win_rate)

        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2)

        if self._n_dumps % self.plot_every == 0:
            self._update_plot()

    def _on_rollout_end(self) -> None:
        from stable_baselines3.common.utils import explained_variance
        rb = self.model.rollout_buffer
        ev = explained_variance(rb.values.flatten(), rb.returns.flatten())
        self._last_explained_var = float(ev)

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", []),
        ):
            if done:
                self._ep_wins.append(1 if info.get("victory", False) else 0)
                self._ep_hp.append(info.get("hp", 0))
                self._ep_rooms.append(info.get("rooms_cleared", 0))
        return True

    def _update_plot(self):
        specs = [
            ("ep_rew_mean",        "Mean Reward",             "royalblue"),
            ("ep_len_mean",        "Mean Episode Length",     "darkorange"),
            ("win_rate",           "Win Rate  (last 200 ep)", "limegreen"),
            ("mean_final_hp",      "Mean Final HP",           "crimson"),
            ("explained_var",      "Explained Variance",      "mediumpurple"),
            ("value_loss",         "Value Loss",              "saddlebrown"),
            ("entropy_loss",       "Entropy Loss",            "teal"),
            ("mean_rooms_cleared", "Mean Rooms Cleared",      "olive"),
        ]

        ts = np.array(self.log["timesteps"])
        for ax, (key, title, color) in zip(self.axes.flatten(), specs):
            vals = np.array(self.log[key], dtype=float)
            ax.clear()
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(ts[:len(vals)][valid], vals[valid], color=color, linewidth=1.3)
            for pt in self._phase_transitions:
                ax.axvline(pt, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel("Timesteps", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=2.5)
        self.fig.savefig(str(SCRIPT_DIR / "training_curves.png"), dpi=120, bbox_inches="tight")

    def _on_training_end(self) -> None:
        self._update_plot()
        self.fig.savefig(str(SCRIPT_DIR / "training_curves.png"), dpi=150, bbox_inches="tight")
        print("\nFinal plot saved → /kaggle/working/training_curves.png")


# ── Shared helper — builds the vectorised env for a given phase ───────────────

def _build_venv(clubs_val: int, spades_val: int) -> DummyVecEnv:
    def _make_env():
        return Monitor(ScoundrelEnv(
            card_counting           = config["card_counting"],
            max_monster_value       = clubs_val,
            max_spades_value        = spades_val,
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
    return DummyVecEnv([_make_env for _ in range(config["n_envs"])])


def _checkpoint_cb() -> CheckpointCallback:
    return CheckpointCallback(
        save_freq         = max(500_000 // config["n_envs"], 1),
        save_path         = str(SCRIPT_DIR),
        name_prefix       = "checkpoint",
        save_vecnormalize = True,
        verbose           = 1,
    )


# ── Fresh training run ────────────────────────────────────────────────────────

def main():
    limits = ScoundrelCallback.CURRICULUM_PHASES[0][0]
    env = VecNormalize(
        _build_venv(limits["clubs"], limits["spades"]),
        norm_obs=True, norm_reward=False,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps       = config["n_steps"],
        ent_coef      = config["ent_coef"],
        gae_lambda    = config["gae_lambda"],
        learning_rate = config["learning_rate"],
        n_epochs      = config["n_epochs"],
        device        = config["device"],
        seed          = config["seed"],
        policy_kwargs = dict(net_arch=config["net_arch"]),
    )

    model.learn(
        total_timesteps = config["timesteps"],
        callback        = CallbackList([ScoundrelCallback(), _checkpoint_cb()]),
    )

    model.save(str(SCRIPT_DIR / "scoundrel_model"))
    env.save(str(SCRIPT_DIR / "scoundrel_vecnormalize.pkl"))

    with open(str(SCRIPT_DIR / "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Model saved        → scoundrel_model.zip")
    print("VecNormalize stats → scoundrel_vecnormalize.pkl")
    print(f"Metrics saved      → {LOG_FILE}")


# ── Resume from the latest checkpoint ────────────────────────────────────────

def resume():
    # Find the latest checkpoint saved by CheckpointCallback
    zips = sorted(
        glob.glob(str(SCRIPT_DIR / "checkpoint_*_steps.zip")),
        key=lambda p: int(re.search(r"(\d+)_steps", p).group(1)),
    )
    if not zips:
        raise FileNotFoundError(
            "No checkpoint files found in /kaggle/working/. "
            "Run main() first, or check the dataset output tab."
        )

    latest_zip      = zips[-1]
    latest_steps    = int(re.search(r"(\d+)_steps", latest_zip).group(1))
    latest_vecnorm  = str(SCRIPT_DIR / f"checkpoint_vecnormalize_{latest_steps}_steps.pkl")
    print(f"Resuming from: {latest_zip}  (step {latest_steps:,})")

    # Restore curriculum state from the training log, truncated to this checkpoint
    with open(LOG_FILE) as f:
        saved_log = json.load(f)

    cutoff = next(
        (i for i, t in enumerate(saved_log["timesteps"]) if t > latest_steps),
        len(saved_log["timesteps"]),
    )
    saved_log = {k: v[:cutoff] for k, v in saved_log.items()}

    last_phase = int(saved_log["curriculum_phase"][-1])
    limits     = ScoundrelCallback.CURRICULUM_PHASES[last_phase][0]
    clubs, spades = limits["clubs"], limits["spades"]
    print(f"Curriculum phase {last_phase}: max_clubs={clubs}, max_spades={spades}")

    # Reconstruct phase transitions up to the checkpoint
    phase_transitions = []
    prev = -1
    for i, p in enumerate(saved_log["curriculum_phase"]):
        if p != prev and prev >= 0:
            phase_transitions.append(saved_log["timesteps"][i])
        prev = p

    # Build env and load VecNormalize stats from checkpoint
    env = VecNormalize.load(latest_vecnorm, _build_venv(clubs, spades))
    env.training = True
    env.norm_obs = True

    model = MaskablePPO.load(latest_zip, env=env, device=config["device"])

    # Restore callback state so the log and plot are continuous
    cb = ScoundrelCallback()
    cb._curriculum_phase  = last_phase
    cb._phase_transitions = phase_transitions
    cb.log                = saved_log
    cb._n_dumps           = len(saved_log["timesteps"])

    remaining = config["timesteps"] - latest_steps
    print(f"Continuing for {remaining:,} more steps "
          f"(total budget: {config['timesteps']:,})")

    model.learn(
        total_timesteps     = remaining,
        callback            = CallbackList([cb, _checkpoint_cb()]),
        reset_num_timesteps = False,
    )

    model.save(str(SCRIPT_DIR / "scoundrel_model"))
    env.save(str(SCRIPT_DIR / "scoundrel_vecnormalize.pkl"))

    with open(str(SCRIPT_DIR / "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Training complete.")
    print("Model saved        → scoundrel_model.zip")
    print("VecNormalize stats → scoundrel_vecnormalize.pkl")


if __name__ == "__main__":
    main()
