import jax
import jax.numpy as jnp
import flax.linen as nn

class ConvBlock(nn.Module):
    filter_count: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = x.reshape((-1, 8, 8, 119))
        x = nn.Conv(features=self.filter_count, kernel_size=(3, 3), padding=1)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        return nn.relu(x)


class ResBlock(nn.Module):
    filter_count: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        residual = x

        x = nn.Conv(features=self.filter_count, kernel_size=(3, 3), padding=1)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(features=self.filter_count, kernel_size=(3, 3), padding=1)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        x = x + residual
        return nn.relu(x)


class PolicyOutBlock(nn.Module):
    filter_count: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = nn.Conv(features=73, kernel_size=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        x = x.reshape((x.shape[0], -1))

        return x


class ValueOutBlock(nn.Module):
    filter_count: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=1, kernel_size=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=3)(x)

        return x


class DeepForkNet(nn.Module):
    depth: int
    filter_count: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = ConvBlock(filter_count=self.filter_count)(x, train=train)

        for _ in range(self.depth):
            x = ResBlock(filter_count=self.filter_count)(x, train=train)

        policy = PolicyOutBlock(filter_count=self.filter_count)(x, train=train)
        value = ValueOutBlock(filter_count=self.filter_count)(x, train=train)

        return policy, value