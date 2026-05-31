"""Environment interface + a few built-in environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    obs: int               # discrete state index (or feature vector for cont.)
    reward: float
    done: bool


class Env(ABC):
    """Generic environment. Discrete state + action by default."""
    n_states: int
    n_actions: int

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> int:
        ...

    @abstractmethod
    def step(self, action: int) -> StepResult:
        ...


class FrozenLake(Env):
    """A simple grid-world. Agent starts in (0,0); goal at (H-1, W-1).

    Optional `slip_prob` makes the floor slippery — with that probability
    the requested action is replaced by a uniformly random action.
    Reward: +1 at the goal, -1 in holes, -0.01 step cost otherwise.
    """

    ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right

    def __init__(self, height: int = 4, width: int = 4,
                 holes: tuple[tuple[int, int], ...] = ((1, 1), (2, 2)),
                 slip_prob: float = 0.0, max_steps: int = 50) -> None:
        self.height = height
        self.width = width
        self.n_states = height * width
        self.n_actions = 4
        self.holes = set(holes)
        self.goal = (height - 1, width - 1)
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self._rng = np.random.default_rng(0)
        self._pos = (0, 0)
        self._steps = 0

    def _state(self) -> int:
        r, c = self._pos
        return r * self.width + c

    def reset(self, *, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = (0, 0)
        self._steps = 0
        return self._state()

    def step(self, action: int) -> StepResult:
        if self.slip_prob > 0 and self._rng.random() < self.slip_prob:
            action = int(self._rng.integers(0, self.n_actions))
        dr, dc = self.ACTIONS[action]
        r, c = self._pos
        nr = max(0, min(self.height - 1, r + dr))
        nc = max(0, min(self.width - 1, c + dc))
        self._pos = (nr, nc)
        self._steps += 1
        if self._pos == self.goal:
            return StepResult(obs=self._state(), reward=+1.0, done=True)
        if self._pos in self.holes:
            return StepResult(obs=self._state(), reward=-1.0, done=True)
        if self._steps >= self.max_steps:
            return StepResult(obs=self._state(), reward=-0.5, done=True)
        return StepResult(obs=self._state(), reward=-0.01, done=False)


class TwoArmedTrap(Env):
    """A 2-action 'safe vs risky' env for policy-gradient demos.

    Action 0 ("safe"): deterministic reward = +0.5
    Action 1 ("risky"): reward = +2.0 w.p. p, else -1.0
    Single-step episodes (bandit framing).
    """

    def __init__(self, p_risky_good: float = 0.6) -> None:
        self.n_states = 1
        self.n_actions = 2
        self.p = p_risky_good
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return 0

    def step(self, action: int) -> StepResult:
        if action == 0:
            r = 0.5
        else:
            r = 2.0 if self._rng.random() < self.p else -1.0
        return StepResult(obs=0, reward=r, done=True)
