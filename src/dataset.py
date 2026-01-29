import glob
import os
from abc import ABC
from typing import Any

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core import download, splits as splits_lib, split_builder as split_builder_lib


class Dataset(tfds.core.GeneratorBasedBuilder, ABC):
    VERSION = tfds.core.Version('1.0.0')
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        Download some PGN-files to data/raw directory and then run the data_processing.py file.
    """

    def _info(self) -> tfds.core.dataset_info.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'obs_bin': tfds.features.Tensor(shape=(8, 8, 117), dtype=np.bool_),
                'obs_scalar': tfds.features.Tensor(shape=(2,), dtype=np.float16),
                'action': tfds.features.Tensor(shape=(), dtype=np.int16),
            }),
            description='Pgx plane observations with action indexes.'
        )

    def _split_generators(
      self,
      dl_manager: download.DownloadManager,
    ):
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'path': dl_manager.manual_dir}
            )
        ]

    def _generate_examples(
      self, path, **kwargs: Any
    ) -> split_builder_lib.SplitGenerator:
        files = glob.glob(os.path.join(path, '*.npz'))
        files.sort()

        for file_path in files:
            try:
                with np.load(file_path) as data:
                    obs_bin = data['obs_bin']
                    obs_scalar = data['obs_scalar']
                    policy = data['policy']

                    for i in range(len(policy)):
                        key = f"{os.path.basename(file_path)}_{i}"
                        yield key, {
                            'obs_bin': obs_bin[i],
                            'obs_scalar': obs_scalar[i],
                            'action': policy[i],
                        }
            except Exception as e:
                print(f"Error reading {file_path}: {e}")