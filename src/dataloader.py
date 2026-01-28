import grain.python as grain
from grain import samplers, sharding
import tensorflow_datasets as tfds

from src.utils import reconstruct_observations


class Reconstruct(grain.MapTransform):

    def map(self, element: dict):
        obs_bin = element['obs_bin']
        obs_scalar = element['obs_scalar']
        element['observation'] = reconstruct_observations(obs_bin, obs_scalar)
        return element


def create_dataloader(
    batch_size: int,
    seed: int,
    worker_count: int,
    num_records: int,
    num_epochs: int
):
    """Create dataloaders for training and testing."""

    train_source = tfds.data_source('dataset', split='train[:90%]')
    test_source = tfds.data_source('dataset', split='train[90%:]')

    train_records_count = len(train_source) if num_records < 0 else num_records * 0.8
    test_records_count = len(test_source) if num_records < 0 else num_records * 0.2

    train_ds = (
        grain.MapDataset.source(train_source)
        .map(Reconstruct())
    )

    test_ds = (
        grain.MapDataset.source(test_source)
        .map(Reconstruct())
    )


    train_sampler = samplers.IndexSampler(
        num_records=train_records_count,
        shard_options=sharding.ShardOptions(shard_index=0, shard_count=1),
        shuffle=True,
        num_epochs=num_epochs,
        seed=seed
    )

    test_sampler = samplers.IndexSampler(
        num_records=test_records_count,
        shard_options=sharding.ShardOptions(shard_index=0, shard_count=1),
        shuffle=False,
        num_epochs=1,
        seed=seed
    )


    train_dataloader = grain.DataLoader(
        data_source=train_ds,
        sampler=train_sampler,
        worker_count=worker_count,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    test_dataloader = grain.DataLoader(
        data_source=test_ds,
        sampler=test_sampler,
        worker_count=worker_count,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    return train_dataloader, test_dataloader