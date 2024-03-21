from dqn_agent import DQN_Agent
import numpy as np

class ModelPool:
    def __init__(self):
        self.models : list[DQN_Agent] = []
        self.uniform_sampling = True
        self.prioritized_sampling = False
        self.rng = np.random.default_rng()
        self.model_ep_freq : list[int] = []
        self.model_transition_freq : list[int] = []
        self.model_crash_freq : list[int] = []
        self.model_crash_window : list[list[int]] = []
        self.model_sr100 : list[float] = []
        self.model_elo : list[float] = []
        self.model_idx : int = None
        self.size = len(self.models)
        self.cycles = {}

    def add_model(self, model : DQN_Agent):
        if len(self.models) > 0:
            # Save Groups Performance
            self.cycles[self.size-1] = (self.model_ep_freq, self.model_transition_freq, self.model_crash_freq, self.model_crash_window, self.model_sr100, self.model_elo)
            # Zero out the lists
            for i in range(self.size):
                self.model_ep_freq[i] = 0
                self.model_transition_freq[i] = 0
                self.model_crash_freq[i] = 0
                self.model_crash_window[i] = []
                self.model_sr100[i] = 0

        self.models.append(model)
        self.size = len(self.models)

        # Resize the lists
        self.model_ep_freq.append(0)
        self.model_transition_freq.append(0)
        self.model_crash_freq.append(0)
        self.model_crash_window.append([])
        self.model_sr100.append(0)

    def end_pool(self):
        if len(self.models) > 0:
            # Save Groups Performance
            self.cycles[self.size-1] = (self.model_ep_freq, self.model_transition_freq, self.model_crash_freq, self.model_crash_window, self.model_sr100, self.model_elo)

    def choose_model(self):
        if len(self.models) == 0:
            raise ValueError("No models in the pool")
        if self.uniform_sampling:
            self.model_idx = self.rng.integers(0, len(self.models))
            self.model_ep_freq[self.model_idx] += 1
        elif self.prioritized_sampling:
            raise NotImplementedError
        else:
            raise ValueError("No sampling method selected")
    
    def predict(self, state):
        if len(self.models) == 0:
            raise ValueError("No models in the pool")
        if self.model_idx is None:
            raise ValueError("No model chosen")
        action = self.models[self.model_idx].predict(state)
        self.model_transition_freq[self.model_idx] += 1
        return action
    
    def update_model_crashes(self, crash : int):
        self.model_crash_freq[self.model_idx] += crash
        self.model_crash_window[self.model_idx].append(crash)
        if len(self.model_crash_window[self.model_idx]) > 100:
            self.model_crash_window[self.model_idx].pop(0)
        self.model_sr100[self.model_idx] = sum(self.model_crash_window[self.model_idx])/100

        
    def update_model_elo(self, version : int, score : float):
        raise NotImplementedError
    