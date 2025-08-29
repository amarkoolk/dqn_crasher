import json
from copy import deepcopy
from typing import Union

import numpy as np
from dqn_agent import DQN_Agent


class ModelPool:
    def __init__(
        self,
        sampling: str = "uniform",
        adjustable_k: bool = False,
        version: str = "v1",
        n_obs: int = 10,
    ):
        self.models: list[Union[DQN_Agent, str]] = []
        self.n_observations = n_obs
        self.uniform_sampling = sampling == "uniform"
        self.prioritized_sampling = sampling == "prioritized"
        self.two_model_sampling = sampling == "two_model"
        self.adjustable_k = adjustable_k
        self.rng = np.random.default_rng()
        self.model_ep_freq: list[int] = []
        self.model_transition_freq: list[int] = []
        self.model_crash_freq: list[int] = []
        self.model_crash_window: list[list[int]] = []
        self.model_sr100: list[float] = []
        self.model_speed: list[list[float]] = []
        self.model_idx: int = None
        self.size = len(self.models)
        self.cycles = {}
        self.version = version  # v1, sampling among models, v2 uniform sampling among baseline and strongest model

        print(f"Model Pool Initialized with {sampling} sampling")

        # ELO
        self.opponent_elo = 1000.0
        self.model_elo: list[float] = []
        self.model_probability: list[float] = []
        self.K_factor = 32
        self.s_factor = 400

        # Eval
        self.n_eval_episodes = 0
        self.model_evals = []
        self.eval = False
        self.latest_model = False

        # ELO Logging
        self.model_pool_log = {}

    def init_eval(self, num_eval_episodes: int, latest_model: bool = False):
        self.eval = True
        self.n_eval_episodes = num_eval_episodes
        self.model_evals = [0 for i in range(self.size)]
        self.latest_model = latest_model

        return self.choose_eval_opponent()

    def choose_eval_opponent(self, randomized: bool = True):
        model_idx = list(range(self.size))
        # Choose a random model to evaluate against unless the model has reached the number of evals required
        if self.latest_model:
            if self.model_evals[self.size - 1] >= self.n_eval_episodes:
                self.eval = False
            self.model_idx = self.size - 1
            self.model_evals[self.model_idx] += 1
            return self.eval

        if randomized:
            for i in range(self.size):
                if self.model_evals[i] >= self.n_eval_episodes:
                    model_idx.remove(i)
            if len(model_idx) == 0:
                self.eval = False
                self.model_idx = self.rng.choice(list(range(self.size)))
            else:
                self.model_idx = self.rng.choice(model_idx)

            self.model_evals[self.model_idx] += 1

            return self.eval
        else:
            for i in range(self.size):
                if self.model_evals[i] >= self.n_eval_episodes:
                    model_idx.remove(i)
            if len(model_idx) == 0:
                self.eval = False
                self.model_idx = 0
            else:
                self.model_idx = model_idx[0]

            self.model_evals[self.model_idx] += 1

            return self.eval

    def set_opponent_elo(self, elo: float):
        self.opponent_elo = elo

    def add_model(
        self,
        model: Union[DQN_Agent, str],
        elo: float = 1000.0,
        opponent_elo: float = 1000.0,
        prepend: bool = False,
    ):
        if len(self.models) > 0:
            # Save Groups Performance
            self.cycles[self.size - 1] = (
                self.model_ep_freq,
                self.model_transition_freq,
                self.model_crash_freq,
                self.model_crash_window,
                self.model_sr100,
                self.model_elo,
            )
            # Zero out the lists
            for i in range(self.size):
                self.model_ep_freq[i] = 0
                self.model_transition_freq[i] = 0
                self.model_crash_freq[i] = 0
                self.model_crash_window[i] = []
                self.model_sr100[i] = 0
                self.model_speed[i] = []

        self.models.append(deepcopy(model))
        # Update Model Probabilities
        self.model_probability = [1 / len(self.models) for i in range(len(self.models))]
        self.size = len(self.models)

        # Resize the lists
        self.model_ep_freq.append(0)
        self.model_transition_freq.append(0)
        self.model_crash_freq.append(0)
        self.model_crash_window.append([])
        self.model_sr100.append(0)
        self.model_speed.append([])
        self.model_elo.append(elo)
        self.opponent_elo = opponent_elo

    def end_pool(self):
        if len(self.models) > 0:
            # Save Groups Performance
            self.cycles[self.size - 1] = (
                self.model_ep_freq,
                self.model_transition_freq,
                self.model_crash_freq,
                self.model_crash_window,
                self.model_sr100,
                self.model_elo,
            )

    def choose_model(self):
        if len(self.models) == 0:
            raise ValueError("No models in the pool")
        if self.uniform_sampling:
            self.model_idx = self.rng.integers(0, len(self.models))
            self.model_ep_freq[self.model_idx] += 1
        elif self.prioritized_sampling:
            self.model_idx = self.rng.choice(self.size, p=self.model_probability)
            self.model_ep_freq[self.model_idx] += 1
        elif self.two_model_sampling:
            models = [0]
            model_elos = np.asarray(self.model_elo)
            strongest_model_idx = model_elos.argsort()[::-1]
            if strongest_model_idx[0] == 0:
                models.append(strongest_model_idx[1])
            else:
                models.append(strongest_model_idx[0])
            self.model_idx = self.rng.choice(models)
            self.model_ep_freq[self.model_idx] += 1
        else:
            raise ValueError("No sampling method selected")

    def predict(self, state):
        if len(self.models) == 0:
            raise ValueError("No models in the pool")
        if self.model_idx is None:
            raise ValueError("No model chosen")
        if self.models[self.model_idx] == "mobil":
            self.model_transition_freq[self.model_idx] += 1
            return 0
        else:
            action = self.models[self.model_idx].predict(state)
            self.model_transition_freq[self.model_idx] += 1
        return action

    def update_model_crashes(self, crash: int):
        self.model_crash_freq[self.model_idx] += crash
        self.model_crash_window[self.model_idx].append(crash)
        if len(self.model_crash_window[self.model_idx]) > 100:
            self.model_crash_window[self.model_idx].pop(0)
        self.model_sr100[self.model_idx] = (
            sum(self.model_crash_window[self.model_idx]) / 100
        )

    def update_model_speed(self, speed: float):
        # If the window is full, pop the first element
        if len(self.model_speed[self.model_idx]) >= 100:
            self.model_speed[self.model_idx].pop(0)

        self.model_speed[self.model_idx].append(speed)

    def set_K_factor(self, config: str):
        if "forward" in config:
            self.K_factor = 32
        elif "behind" in config:
            self.K_factor = 32
        elif "adjacent" in config:
            self.K_factor = 8
        else:
            self.K_factor = 32

    def update_model_elo(self, Sa: int, Sb: int, config: str = ""):
        if self.adjustable_k:
            self.set_K_factor(config)
        Ra = self.opponent_elo
        Rb = self.model_elo[self.model_idx]

        Ea = 1 / (1 + 10.0 ** ((Rb - Ra) / self.s_factor))
        Eb = 1 / (1 + 10.0 ** ((Ra - Rb) / self.s_factor))

        self.opponent_elo = Ra + self.K_factor * (Sa - Ea)
        self.model_elo[self.model_idx] = Rb + self.K_factor * (Sb - Eb)

    # We track the probability that the selected model can beat the current opponent
    def update_probabilities(self, ego: bool):
        cum_sum = 0
        for i in range(self.size):
            self.model_probability[i] = 1 / (
                1 + np.exp(-(self.model_elo[i] - self.opponent_elo) / self.s_factor)
            )
            cum_sum += self.model_probability[i]
        for i in range(self.size):
            self.model_probability[i] /= cum_sum

    def glicko2(self, Sa: int, Sb: int):
        pass

    def log_model_pool(self, cycle, eval_iter, opponent_model, Sa, Sb):
        log = {
            "Sa": Sa,
            "Sb": Sb,
            "model_idx": self.model_idx,
            "opponent_model": opponent_model,
            "opponent_elo": self.opponent_elo,
            "model_elo": deepcopy(self.model_elo),
            "model_probability": deepcopy(self.model_probability),
        }
        if cycle in self.model_pool_log:
            self.model_pool_log[cycle][eval_iter] = log
        else:
            self.model_pool_log[cycle] = {}
            self.model_pool_log[cycle][eval_iter] = log

    def write_model_pool_log(self, path):
        with open(path, "w") as f:
            json.dump(self.model_pool_log, f, cls=NpEncoder)
        print(f"Model Pool Log Written to {path}")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
