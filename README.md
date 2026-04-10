# TablutZero

A fully vectorized AlphaZero implementation for the historical board game Tablut, built with JAX, Flax NNX, mctx, and Pgx.
TablutZero learns to play Tablut entirely through self-play, following the AlphaZero paradigm: a neural network guides Monte Carlo Tree Search (MCTS), and the search results are used to train the network in a continuous loop. No human game data or handcrafted heuristics are used.

### The Game

Tablut is an asymmetric strategy board game from the Tafl family, historically played in Northern Europe. It is played on a 9×9 board between two sides:

- **Attackers** (16 pieces) start on the edges and aim to capture the King.
- **Defenders** (8 pieces + 1 King) start in the center and aim to escort the King to any corner square.

All pieces (including the king) are captured by custodial capture (sandwiching between two enemies or an enemy and a hostile square). The throne and corners act as hostile squares. For more detailed rules, see `extended_abstract.pdf` Appendix A.1.

### Architecture

The system uses a dual-headed convolutional neural network with separate policy and value heads per player, sharing a common residual trunk. This accounts for the asymmetric nature of the game where attacker and defender strategies differ fundamentally.

- **Trunk**: Initial convolution block followed by residual blocks (default: 8 layers, 128 filters)
- **Heads**: Separate policy + value head pairs for attacker and defender
- **Search**: Gumbel MuZero MCTS via the `mctx` library
- **Training**: Self-play data generation → replay buffer → supervised training loop
- **Precision**: Mixed precision training (bfloat16 compute, float32 outputs)

The observation space encodes 8 steps of board history with piece positions, repetition counters, side to move, step count, and half-move clock (43 planes total). The action space uses AlphaZero-style directional labels (2592 actions = 81 squares × 32 direction-distance planes).

## Project Structure

```
├── src/
│   ├── model.py            # Neural network architecture
│   ├── mcts.py             # MCTS integration with mctx
│   ├── self_play.py        # Self-play data generation
│   ├── train.py            # Training loop
│   ├── evaluation.py       # Model evaluation and BayesElo rating
│   ├── metrics.py          # Training metrics tracking and plotting
│   ├── utils.py            # Loss functions, augmentation, buffer utilities
│   └── tablut/
│       ├── tablut.py       # Pgx environment wrapper
│       ├── tablut_jax.py   # Tablut game engine
│       ├── play.py         # Human vs AI interface
│       └── ui.py           # Pygame GUI
├── config.template.yaml    # Default hyperparameter configuration
├── setup.py
├── main.py                 # Entry point for the src module
└── requirements.txt
```

## Training

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s)
- Sufficient RAM for the replay buffer (scales with `buffer_multiplier` × `batch_size` × `self_play_steps`)

### Setup

1. Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

2. Create your configuration file:

```bash
cp config.template.yaml config.yaml
```

Edit `config.yaml` to match your hardware. Key parameters to adjust:

| Parameter | Description |
|---|---|
| `model.depth` | Number of residual blocks |
| `model.filter_count` | Filters per convolutional layer |
| `mcts.simulations` | MCTS simulations per move (128 recommended for meaningful training) |
| `train.batch_size` | Games run in parallel during self-play and training batch size |
| `train.self_play_steps` | Steps per self-play phase |
| `train.buffer_multiplier` | Replay buffer size as multiple of minimum |
| `train.iterations` | Total training iterations |

4. Run training:

```bash
python -m src.train
```

### Resuming Training

Set `load_checkpoint: True` in `config.yaml`. The system will restore the model, optimizer state, and replay buffer from `models/checkpoints/`. Ensure the `game_results.pgn` file is consistent with the latest metrics snapshot in `data/metrics/`.

### Outputs

- **Checkpoints**: `models/checkpoints/` (full training state including buffer)
- **Model weights**: `models/inference/` (weights only, suitable for inference)
- **Evaluation pool**: `models/eval_pool/` (past model snapshots for Elo tracking)
- **Metrics**: `data/metrics/` (JSON history + PGN backups)
- **Plots**: `data/plots/` (training dashboard PDFs)

### Hardware Requirements

The provided model was trained on two NVIDIA H200 GPUs with 320GB of system RAM. The replay buffer is held in RAM and can grow large depending on the `buffer_multiplier` setting. Multi-GPU training is supported via JAX's data-parallel sharding.

## Tech Stack

- **[JAX](https://github.com/jax-ml/jax)** — Vectorized computation and automatic differentiation
- **[Flax NNX](https://github.com/google/flax)** — Neural network modules
- **[Optax](https://github.com/google-deepmind/optax)** — Optimizer (AdamW + cosine schedule)
- **[mctx](https://github.com/google-deepmind/mctx)** — Gumbel MuZero MCTS
- **[Pgx](https://github.com/sotetsuk/pgx)** — Game environment framework
- **[Flashbax](https://github.com/instadeepai/flashbax)** — JAX-native replay buffer
- **[Orbax](https://github.com/google/orbax)** — Checkpointing
- **[Hydra](https://github.com/facebookresearch/hydra)** — Configuration management
- **[Pygame](https://www.pygame.org/)** — Game UI