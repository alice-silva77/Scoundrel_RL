"""
ScoundrelEnv — Gymnasium environment for the Scoundrel card game.

Action space  Discrete(9)
─────────────────────────────────────────────────────────────────
  0–3  Play the card at room position i  (monsters: fight barehanded)
  4–7  Play the card at room position i  (monsters: use equipped weapon)
  8    Flee the room

  Actions 4–7 are only valid when:
    • room[i] is a monster
    • a weapon is equipped (weapon_val > 0)
    • monster value < weapon_constraint  (weapon hasn't dulled past it)

  Action 8 is only valid when:
    • can_run is True  (not immediately after a previous flee)
    • cards_played == 0  (room not yet engaged — once you touch a card you
                          must finish the room)
    • NOT (deck is empty AND room has fewer than 4 cards)  (nowhere to go)

Observation vector  float32
─────────────────────────────────────────────────────────────────
  Base (15 features, always present):
  idx  field               range
   0   hp                  [0, 20]
   1   weapon_val          [0, 14]   (0 = no weapon)
   2   weapon_constraint   [2, 15]   (15 = NO_CONSTRAINT, weapon is fresh)
   3   potion_drunk        {0, 1}
   4   can_run             {0, 1}
   5   cards_played        {0, 1, 2} cards played in the current room
   6   deck_size           [0, 44]
  7–8  room slot 0: (type_id, value)
  9–10 room slot 1: (type_id, value)
 11–12 room slot 2: (type_id, value)
 13–14 room slot 3: (type_id, value)
  card type_id: 0=empty slot, 1=potion(♥), 2=weapon(♦), 3=monster(♣/♠)

  Card-counting extension (44 features, present when card_counting=True):
  15–58  one binary flag per deck card in canonical order (H2..H10, D2..D10,
          C2..C14, S2..S14).  1 = card has been drawn from the deck at some
          point (so it is NOT in the unknown part of the remaining deck).

  Total observation size: 15 (base) or 59 (with card counting).

Reward function  (all components are configurable via constructor)
─────────────────────────────────────────────────────────────────
  heal_scale   × heal / MAX_HP     when drinking a potion
  damage_scale × damage / MAX_HP   (negative) when taking damage
  room_reward                       flat bonus when a room is cleared
  win_reward                        flat bonus on victory
  death_penalty                     flat penalty on death (applied negative)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────────────────────
#  Card
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Card:
    suit: str   # 'H' | 'D' | 'C' | 'S'
    value: int  # 2–14

    @property
    def type(self) -> str:
        if self.suit == 'H':
            return 'potion'
        if self.suit == 'D':
            return 'weapon'
        return 'monster'

    @property
    def type_id(self) -> int:
        return {'H': 1, 'D': 2, 'C': 3, 'S': 3}[self.suit]

    def __repr__(self) -> str:
        face = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(self.value, str(self.value))
        sym  = {'H': '♥', 'D': '♦', 'C': '♣', 'S': '♠'}[self.suit]
        return f"{face}{sym}"


def _build_deck(max_clubs_value: int = 14, max_spades_value: int = 14) -> list[Card]:
    """
    Return the Scoundrel dungeon deck.
    Clubs and spades can have separate max values for the one-king curriculum phase.
    Full deck = 44 cards (max_clubs=14, max_spades=14).
    """
    deck = []
    for suit in ('H', 'D', 'C', 'S'):
        for val in range(2, 15):
            if suit in ('H', 'D') and val >= 11:
                continue
            if suit == 'C' and val > max_clubs_value:
                continue
            if suit == 'S' and val > max_spades_value:
                continue
            deck.append(Card(suit, val))
    return deck


# Canonical ordering for the card-counting observation vector.
# Always built from the full 44-card deck so indices are stable across phases.
_CANONICAL_DECK: list[Card] = _build_deck()
_CARD_TO_IDX: dict[Card, int] = {c: i for i, c in enumerate(_CANONICAL_DECK)}


# ──────────────────────────────────────────────────────────────
#  ScoundrelGame  —  pure logic, no GUI
# ──────────────────────────────────────────────────────────────

class ScoundrelGame:
    """
    All Scoundrel game rules in one self-contained class.
    No randomness beyond the initial shuffle; call reset() to start a game.
    """

    MAX_HP        = 20
    NO_CONSTRAINT = 15   # sentinel: weapon hasn't been used yet, any target is legal

    def __init__(
        self,
        heal_scale:              float = 1.0,
        damage_scale:            float = 1.0,
        room_reward:             float = 0.1,
        room_reward_scale:       float = 0.15,
        win_reward:              float = 10.0,
        win_hp_bonus:            float = 5.0,
        death_penalty:           float = 3.0,
        potion_waste_penalty:    float = 0.5,
        weapon_efficiency_bonus: float = 0.0,
        king_weapon_bonus:       float = 0.0,
        max_monster_value:       int   = 14,
        max_spades_value:        int   = None,  # defaults to max_monster_value
    ) -> None:
        self.heal_scale              = heal_scale
        self.damage_scale            = damage_scale
        self.room_reward             = room_reward
        self.room_reward_scale       = room_reward_scale
        self.win_reward              = win_reward
        self.win_hp_bonus            = win_hp_bonus
        self.death_penalty           = death_penalty
        self.potion_waste_penalty    = potion_waste_penalty
        self.weapon_efficiency_bonus = weapon_efficiency_bonus
        self.king_weapon_bonus       = king_weapon_bonus
        self.max_monster_value       = max_monster_value
        self.max_spades_value        = max_spades_value if max_spades_value is not None else max_monster_value

        self.deck: list[Card]  = []
        self.room: list[Card]  = []
        self.seen: set[Card]   = set()
        self.hp: int           = self.MAX_HP
        self.weapon_val: int   = 0
        self.weapon_constraint: int = self.NO_CONSTRAINT
        self.potion_drunk: bool  = False
        self.can_run: bool       = True
        self.cards_played: int   = 0
        self.rooms_cleared: int  = 0

    # ── setup ──────────────────────────────────────────────────

    def reset(self, rng: Optional[random.Random] = None) -> None:
        self.deck = _build_deck(self.max_monster_value, self.max_spades_value)
        (rng or random).shuffle(self.deck)
        self.room              = []
        self.seen              = set()
        self.hp                = self.MAX_HP
        self.weapon_val        = 0
        self.weapon_constraint = self.NO_CONSTRAINT
        self.potion_drunk      = False
        self.can_run           = True
        self.cards_played      = 0
        self.rooms_cleared     = 0
        self._fill_room()

    # ── properties ─────────────────────────────────────────────

    @property
    def terminated(self) -> bool:
        return self.hp <= 0 or (len(self.deck) == 0 and len(self.room) == 0)

    @property
    def victory(self) -> bool:
        return self.hp > 0 and len(self.deck) == 0 and len(self.room) == 0

    # ── internals ──────────────────────────────────────────────

    def set_max_monster_value(self, clubs_value: int, spades_value: int = None) -> None:
        """Update the curriculum cap live (called between episodes via env_method)."""
        self.max_monster_value = clubs_value
        self.max_spades_value  = spades_value if spades_value is not None else clubs_value

    def _fill_room(self) -> None:
        """Draw cards from the deck until the room has 4 cards (or deck runs out)."""
        needed = 4 - len(self.room)
        for _ in range(needed):
            if self.deck:
                card = self.deck.pop(0)
                self.room.append(card)
                self.seen.add(card)

    # ── strategic observation features ─────────────────────────

    @property
    def max_monster_in_room(self) -> int:
        vals = [c.value for c in self.room if c.type == 'monster']
        return max(vals) if vals else 0

    @property
    def weapon_can_kill_max(self) -> int:
        if self.weapon_val == 0 or not self.room:
            return 0
        return int(self.max_monster_in_room > 0 and
                   self.max_monster_in_room < self.weapon_constraint)

    @property
    def n_weapon_targets(self) -> int:
        if self.weapon_val == 0:
            return 0
        return sum(1 for c in self.room if self._can_use_weapon_on(c))

    @property
    def room_danger(self) -> int:
        return sum(c.value for c in self.room if c.type == 'monster')

    def _can_flee(self) -> bool:
        if not self.can_run:
            return False
        if self.cards_played > 0:
            return False
        if len(self.deck) == 0 and len(self.room) < 4:
            return False
        return True

    def _can_use_weapon_on(self, card: Card) -> bool:
        return (
            card.type == 'monster'
            and self.weapon_val > 0
            and card.value < self.weapon_constraint
        )

    # ── public API ─────────────────────────────────────────────

    def valid_actions(self) -> list[int]:
        if self.terminated:
            return []

        actions: list[int] = []

        if self._can_flee():
            actions.append(8)

        for i, card in enumerate(self.room):
            actions.append(i)
            if self._can_use_weapon_on(card):
                actions.append(i + 4)

        return actions

    def step(self, action: int) -> tuple[float, bool]:
        """Apply one action. Returns (reward, done)."""
        assert action in self.valid_actions(), (
            f"Illegal action {action}. Legal: {self.valid_actions()}"
        )

        reward = 0.0

        # ── FLEE ───────────────────────────────────────────────
        if action == 8:
            self.deck.extend(self.room)
            self.room         = []
            self.can_run      = False
            self.cards_played = 0
            self.potion_drunk = False
            self._fill_room()
            return reward, self.terminated

        # ── PLAY CARD ──────────────────────────────────────────
        use_weapon = action >= 4
        card_idx   = action % 4
        card       = self.room.pop(card_idx)

        if card.type == 'potion':
            if not self.potion_drunk:
                hp_before  = self.hp
                effective  = min(card.value, self.MAX_HP - self.hp)
                wasted     = card.value - effective
                self.hp   += effective
                self.potion_drunk = True
                urgency = 1.0 - (hp_before / self.MAX_HP)
                reward += self.heal_scale * effective * urgency / self.MAX_HP
                reward -= self.potion_waste_penalty * wasted / self.MAX_HP

        elif card.type == 'weapon':
            self.weapon_val        = card.value
            self.weapon_constraint = self.NO_CONSTRAINT

        elif card.type == 'monster':
            if use_weapon:
                damage                 = max(0, card.value - self.weapon_val)
                self.weapon_constraint = card.value
                if damage == 0:
                    reward += self.weapon_efficiency_bonus
                if card.value >= 13:
                    reward += self.king_weapon_bonus
            else:
                damage = card.value
            self.hp -= damage
            reward  -= self.damage_scale * damage / self.MAX_HP

        self.cards_played += 1

        # ── DEATH ──────────────────────────────────────────────
        if self.hp <= 0:
            self.hp = 0
            return reward - self.death_penalty, True

        # ── ROOM COMPLETE ──────────────────────────────────────
        if self.cards_played >= 3:
            self.cards_played   = 0
            self.potion_drunk   = False
            self.can_run        = True
            self.rooms_cleared += 1
            self._fill_room()
            reward += self.room_reward * (1 + self.rooms_cleared * self.room_reward_scale)

        # ── VICTORY ────────────────────────────────────────────
        if self.victory:
            hp_bonus = (self.hp / self.MAX_HP) * self.win_hp_bonus
            return reward + self.win_reward + hp_bonus, True

        return reward, False


# ──────────────────────────────────────────────────────────────
#  ScoundrelEnv  —  Gymnasium wrapper
# ──────────────────────────────────────────────────────────────

class ScoundrelEnv(gym.Env):
    """
    Gymnasium-compatible environment for Scoundrel.

    Parameters
    ----------
    card_counting : bool
        When True the observation is extended with a 44-element binary vector
        tracking which cards have been drawn from the deck (observation size
        grows from 15 to 59).
    max_monster_value : int
        Cap on clubs monster values (curriculum control).
    max_spades_value : int or None
        Cap on spades monster values. Defaults to max_monster_value.
        Set lower than max_monster_value to create a one-king phase.
    """

    metadata = {'render_modes': ['ansi']}

    def __init__(
        self,
        render_mode:             Optional[str] = None,
        card_counting:           bool  = False,
        heal_scale:              float = 1.0,
        damage_scale:            float = 1.0,
        room_reward:             float = 0.1,
        room_reward_scale:       float = 0.15,
        win_reward:              float = 10.0,
        win_hp_bonus:            float = 5.0,
        death_penalty:           float = 3.0,
        potion_waste_penalty:    float = 0.5,
        weapon_efficiency_bonus: float = 0.0,
        king_weapon_bonus:       float = 0.0,
        max_monster_value:       int   = 14,
        max_spades_value:        int   = None,
    ) -> None:
        super().__init__()
        self.render_mode   = render_mode
        self.card_counting = card_counting
        self.game = ScoundrelGame(
            heal_scale              = heal_scale,
            damage_scale            = damage_scale,
            room_reward             = room_reward,
            room_reward_scale       = room_reward_scale,
            win_reward              = win_reward,
            win_hp_bonus            = win_hp_bonus,
            death_penalty           = death_penalty,
            potion_waste_penalty    = potion_waste_penalty,
            weapon_efficiency_bonus = weapon_efficiency_bonus,
            king_weapon_bonus       = king_weapon_bonus,
            max_monster_value       = max_monster_value,
            max_spades_value        = max_spades_value,
        )

        self.action_space = spaces.Discrete(9)

        base_low  = np.array([ 0,  0,  2, 0, 0, 0,  0] + [0,  0]*4 + [ 0,  0, 0,  0,  0], dtype=np.float32)
        base_high = np.array([20, 14, 15, 1, 1, 3, 44] + [3, 14]*4 + [15, 14, 1,  4, 56], dtype=np.float32)

        if card_counting:
            cc_low  = np.zeros(44, dtype=np.float32)
            cc_high = np.ones(44,  dtype=np.float32)
            obs_low  = np.concatenate([base_low,  cc_low])
            obs_high = np.concatenate([base_high, cc_high])
        else:
            obs_low, obs_high = base_low, base_high

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def set_max_monster_value(self, clubs_value: int, spades_value: int = None) -> None:
        """Curriculum hook: update per-suit monster caps on the underlying game."""
        self.game.set_max_monster_value(clubs_value, spades_value)

    # ── Gymnasium API ──────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = random.Random(seed) if seed is not None else None
        self.game.reset(rng=rng)
        return self._obs(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward, terminated = self.game.step(int(action))
        obs  = self._obs()
        info = self._info()
        if self.render_mode == 'ansi':
            self.render()
        return obs, reward, terminated, False, info

    def render(self) -> None:
        g = self.game
        room_str = '  '.join(repr(c) for c in g.room) if g.room else '—'
        if g.weapon_val:
            limit = 'any' if g.weapon_constraint == ScoundrelGame.NO_CONSTRAINT \
                          else f'<{g.weapon_constraint}'
            weapon_str = f'{g.weapon_val} ({limit})'
        else:
            weapon_str = 'none'
        print(
            f"HP {g.hp:2}/{g.MAX_HP}  "
            f"Weapon {weapon_str:14}  "
            f"Deck {len(g.deck):2}  "
            f"Room [{room_str}]  "
            f"Played {g.cards_played}/3  "
            f"Run {'✓' if g.can_run else '✗'}  "
            f"Valid {g.valid_actions()}"
        )

    # ── Action masking (sb3-contrib MaskablePPO) ───────────────

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(9, dtype=bool)
        for a in self.game.valid_actions():
            mask[a] = True
        return mask

    # ── Internals ──────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        g = self.game
        card_feats: list[float] = []
        for i in range(4):
            if i < len(g.room):
                c = g.room[i]
                card_feats += [float(c.type_id), float(c.value)]
            else:
                card_feats += [0.0, 0.0]

        base: list[float] = [
            float(g.hp),
            float(g.weapon_val),
            float(g.weapon_constraint),
            float(g.potion_drunk),
            float(g.can_run),
            float(g.cards_played),
            float(len(g.deck)),
        ] + card_feats + [
            float(g.rooms_cleared),
            float(g.max_monster_in_room),
            float(g.weapon_can_kill_max),
            float(g.n_weapon_targets),
            float(g.room_danger),
        ]

        if self.card_counting:
            seen_vec = [1.0 if c in g.seen else 0.0 for c in _CANONICAL_DECK]
            base = base + seen_vec

        return np.array(base, dtype=np.float32)

    def _info(self) -> dict:
        g = self.game
        return {
            'hp':                g.hp,
            'weapon_val':        g.weapon_val,
            'weapon_constraint': g.weapon_constraint,
            'deck_size':         len(g.deck),
            'room':              list(g.room),
            'can_run':           g.can_run,
            'cards_played':      g.cards_played,
            'rooms_cleared':     g.rooms_cleared,
            'valid_actions':     g.valid_actions(),
            'victory':           g.victory,
        }
