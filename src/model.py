import jax.numpy as jnp
from flax import nnx
from jax import Array

BOARD_EDGE = 9
OBS_PLANES = 43
PLANES = 4 * (BOARD_EDGE - 1)

"""
Neural network for TablutZeroNet with separate policy and value heads per player.

Tablut is asymmetric: attacker and defender have different pieces, objectives, and
legal move sets. A single shared head would conflate these roles. Instead, the network
uses a shared convolutional trunk followed by independent (policy, value) head pairs
for player 0 (attacker) and player 1 (defender). The appropriate head is selected at
inference time based on the current player index.

Architecture: ConvBlock → N × ResBlock → {p0_policy, p0_value, p1_policy, p1_value}
All convolutions use bfloat16 for mixed-precision training on GPU.
"""
class ConvBlock(nnx.Module):
    def __init__(self, filter_count: int, rngs: nnx.Rngs):
        self.filter_count = filter_count
        self.conv = nnx.Conv(in_features=OBS_PLANES, out_features=filter_count,
                             kernel_size=(3, 3), padding=1, rngs=rngs, param_dtype=jnp.bfloat16)
        self.bn = nnx.BatchNorm(num_features=filter_count, rngs=rngs, param_dtype=jnp.bfloat16)

    def __call__(self, x: Array, train: bool) -> Array:
        x = x.astype(jnp.bfloat16)
        x = x.reshape((-1, BOARD_EDGE, BOARD_EDGE, OBS_PLANES))
        x = self.conv(x)
        x = self.bn(x, use_running_average=not train)

        return nnx.relu(x)


class ResBlock(nnx.Module):
    def __init__(self, filter_count: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=filter_count, out_features=filter_count,
                              kernel_size=(3, 3), padding=1, rngs=rngs, param_dtype=jnp.bfloat16)
        self.bn1 = nnx.BatchNorm(num_features=filter_count, rngs=rngs, param_dtype=jnp.bfloat16)

        self.conv2 = nnx.Conv(in_features=filter_count, out_features=filter_count,
                              kernel_size=(3, 3), padding=1, rngs=rngs, param_dtype=jnp.bfloat16)
        self.bn2 = nnx.BatchNorm(num_features=filter_count, rngs=rngs, param_dtype=jnp.bfloat16)

    def __call__(self, x: Array, train: bool) -> Array:
        x = x.astype(jnp.bfloat16)
        residual = x

        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)

        x = x + residual
        return nnx.relu(x)


class PolicyOutBlock(nnx.Module):
    def __init__(self, filter_count: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=filter_count, out_features=PLANES,
                             kernel_size=(1, 1), rngs=rngs, param_dtype=jnp.bfloat16)
        self.bn = nnx.BatchNorm(num_features=PLANES, rngs=rngs, param_dtype=jnp.bfloat16)

    def __call__(self, x: Array, train: bool) -> Array:
        x = x.astype(jnp.bfloat16)
        x = self.conv(x)
        x = self.bn(x, use_running_average=not train)

        x = x.reshape((x.shape[0], -1))
        return x


class ValueOutBlock(nnx.Module):
    def __init__(self, filter_count: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=filter_count, out_features=1,
                             kernel_size=(1, 1), rngs=rngs, param_dtype=jnp.bfloat16)
        self.bn = nnx.BatchNorm(num_features=1, rngs=rngs, param_dtype=jnp.bfloat16)

        self.dense1 = nnx.Linear(in_features=BOARD_EDGE ** 2, out_features=256, rngs=rngs, param_dtype=jnp.bfloat16)
        self.dense2 = nnx.Linear(in_features=256, out_features=1, rngs=rngs, param_dtype=jnp.bfloat16)

    def __call__(self, x: Array, train: bool) -> Array:
        x = x.astype(jnp.bfloat16)
        x = self.conv(x)
        x = self.bn(x, use_running_average=not train)
        x = nnx.relu(x)

        x = x.reshape((x.shape[0], -1))

        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)

        x = jnp.tanh(x)

        return x.squeeze(-1)


class TablutZeroNet(nnx.Module):
    def __init__(self, depth: int, filter_count: int, rngs: nnx.Rngs):
        self.depth = depth
        self.filter_count = filter_count

        self.conv_block = ConvBlock(filter_count, rngs=rngs)

        self.res_blocks = nnx.List([
            ResBlock(filter_count, rngs=rngs) for _ in range(depth)
        ])

        self.p0_policy_head = PolicyOutBlock(filter_count, rngs=rngs)
        self.p0_value_head = ValueOutBlock(filter_count, rngs=rngs)
        self.p1_policy_head = PolicyOutBlock(filter_count, rngs=rngs)
        self.p1_value_head = ValueOutBlock(filter_count, rngs=rngs)

    def __call__(self, x: Array, train: bool = False) -> tuple[Array, Array, Array, Array]:
        """
        Forward pass through the shared trunk and all four heads.

        Returns:
            (p0_policy, p0_value, p1_policy, p1_value) where:
                p0_policy: (batch, BOARD_SIZE * ACTION_PLANES) raw logits for attacker
                p0_value:  (batch,) scalar value estimate for attacker in [-1, 1]
                p1_policy: (batch, BOARD_SIZE * ACTION_PLANES) raw logits for defender
                p1_value:  (batch,) scalar value estimate for defender in [-1, 1]
        """
        x = x.astype(jnp.bfloat16)
        x = self.conv_block(x, train=train)

        for block in self.res_blocks:
            x = block(x, train=train)

        p0_policy = self.p0_policy_head(x, train=train).astype(jnp.float32)
        p0_value = self.p0_value_head(x, train=train).astype(jnp.float32)
        p1_policy = self.p1_policy_head(x, train=train).astype(jnp.float32)
        p1_value = self.p1_value_head(x, train=train).astype(jnp.float32)

        return p0_policy, p0_value, p1_policy, p1_value