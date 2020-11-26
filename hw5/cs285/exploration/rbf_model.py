from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import numpy as np
import torch


class RBFModel(BaseExplorationModel):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.sigma = hparams['rbf_sigma']
        self.buffer_size = hparams['rbf_buffer_size']
        self.means = None

    def get_prob(self, ob_no):
        if self.means is None:
            # Return a uniform distribution if we don't have samples in the
            # replay buffer yet.
            return (1.0 / len(ob_no)) * np.ones(len(ob_no))
        else:
            # 1. Compute deltas
            deltas = ob_no[:, None] - self.means[None, :]

            # 2. Euclidean distance
            euc_dists = np.linalg.norm(deltas, axis=-1)

            # 3. Gaussian
            gaussians = np.exp(-euc_dists ** 2 / (2 * self.sigma ** 2))

            # 4. Average
            densities = np.mean(gaussians, axis=-1)

            return densities

    def bonus_function(self, prob):
        return -np.log(prob)

    def compute_reward_bonus(self, ob_no):
        prob = self.get_prob(ob_no)
        bonus = self.bonus_function(prob)
        return bonus

    def forward_np(self, ob_no):
        return self.compute_reward_bonus(ob_no)

    def update(self, ob_no, replays):
        self.means = replays
        return 0.0

