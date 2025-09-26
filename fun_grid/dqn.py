"""Deep Q-Network agent implementation used by the FunGrid agents."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Deque, Tuple

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


@dataclass
class DQNConfig:
    """Configuration values for the DQN agent."""

    state_size: int
    action_size: int
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    learning_rate: float = 0.001
    memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = field(
        default_factory=list
    )


class DQNAgent:
    """Simple Deep Q-Network agent for decision making."""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.model = self._build_model()

    @property
    def state_size(self) -> int:
        return self.config.state_size

    @property
    def action_size(self) -> int:
        return self.config.action_size

    @property
    def memory(self) -> Deque:
        return self.config.memory

    @property
    def gamma(self) -> float:
        return self.config.gamma

    @property
    def epsilon(self) -> float:
        return self.config.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self.config.epsilon = value

    @property
    def epsilon_min(self) -> float:
        return self.config.epsilon_min

    @property
    def epsilon_decay(self) -> float:
        return self.config.epsilon_decay

    @property
    def learning_rate(self) -> float:
        return self.config.learning_rate

    def _build_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return int(np.argmax(act_values[0]))

    def replay(self, batch_size: int) -> None:
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state.reshape(1, -1))[0]
                )
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.train_on_batch(state.reshape(1, -1), target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
