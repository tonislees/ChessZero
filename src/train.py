from pathlib import Path

import jax
import numpy as np
import jax.numpy as jnp

from src.utils import reconstruct_observations


class DataLoader:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.processed_dir = self.root / 'data/processed'


    def load_states(self, file_count: int):
        files = list(self.processed_dir.glob('*.npz'))
        for file in files:
            data = np.load(file)
            obs_bin = data['obs_bin']
            obs_scalar = data['obs_scalar']

            batch_obs = jax.jit(reconstruct_observations(obs_bin, obs_scalar))
            yield jnp.array(batch_obs)