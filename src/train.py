import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.dataloader import create_dataloaders
import optax
import orbax.checkpoint as ocp
from flax import nnx
import matplotlib.pyplot as plt

from src.model import ChessZeroNet


def loss_fn(model, batch, train=True):
    logits, _ = model(batch['observation'], train=train)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['action']
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model, metrics, optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['action'])
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model, metrics, batch):
    loss, logits = loss_fn(model=model, batch=batch, train=False)
    metrics.update(loss=loss, logits=logits, labels=batch['action'])


class Coach:
    def __init__(self, cfg: DictConfig):
        root_dir = self.root = Path(__file__).resolve().parents[1]
        self.model_dir = root_dir / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.model_dir / 'checkpoints'
        self.checkpointer = ocp.StandardCheckpointer()
        self.plot_dir = root_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = cfg.train.num_epochs
        self.seed = cfg.train.seed
        self.rngs: nnx.Rngs = nnx.Rngs(self.seed)
        self.model: nnx.Module = ChessZeroNet(depth=cfg.model.depth, filter_count=cfg.model.filter_count, rngs=self.rngs)

        if self.checkpoints_dir.exists() and any(self.checkpoints_dir.iterdir()):
            graphdef, abstract_state = nnx.split(self.model)
            restored_state = self.checkpointer.restore(self.checkpoints_dir, abstract_state)
            self.model = nnx.merge(graphdef, restored_state)
            print(f"Restored model from {self.checkpoints_dir}")

        self.optimizer: nnx.Optimizer = nnx.Optimizer(
        self.model, optax.adamw(learning_rate=cfg.train.learning_rate), wrt=nnx.Param
        )
        self.metrics: nnx.MultiMetric = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average(argname='loss'),
        )
        self.metrics_history = {
          'train_loss': [],
          'train_accuracy': [],
          'test_loss': [],
          'test_accuracy': [],
        }
        train_dl, test_dl = create_dataloaders(batch_size=cfg.train.batch_size, seed=self.seed,
                                               worker_count=cfg.train.worker_count, num_epochs=self.num_epochs,
                                               num_records=cfg.train.num_records,
                                               drop_remainder=cfg.train.drop_remainder)
        self.train_iterator = train_dl
        self.total_train_batches = len(self.train_iterator._sampler) // cfg.train.batch_size
        self.test_iterator = test_dl
        self.total_test_batches = len(self.test_iterator._sampler) // cfg.train.batch_size


    def _train_epoch(self):
        self.model.train()
        for step, batch in tqdm(enumerate(self.train_iterator), total=self.total_train_batches, unit='batch'):
            train_step(self.model, self.metrics, self.optimizer, batch)

        for metric, value in self.metrics.compute().items():
            self.metrics_history[f'train_{metric}'].append(value)
        self.metrics.reset()


    def evaluate(self):
        self.model.eval()
        for step, batch in tqdm(enumerate(self.test_iterator), total=self.total_test_batches, unit='batch'):
            eval_step(self.model, self.metrics, batch)

        for metric, value in self.metrics.compute().items():
            self.metrics_history[f'test_{metric}'].append(value)
        self.metrics.reset()


    def _plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.set_title('Loss')
        ax2.set_title('Accuracy')
        for dataset in ('train', 'test'):
            ax1.plot(self.metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
            ax2.plot(self.metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
        ax1.legend()
        ax2.legend()

        filename = f"Metrics_vs_Epoch_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        plt.savefig(self.plot_dir / filename)


    def train(self):
        for epoch in range(self.num_epochs):
            self._train_epoch()
            self.evaluate()
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} "
                f"Train loss: {self.metrics_history['train_loss'][-1]:.4f} "
                f"test loss: {self.metrics_history['test_loss'][-1]:.4f} "
                f"Train accuracy: {self.metrics_history['train_accuracy'][-1]:.4f} "
                f"test accuracy: {self.metrics_history['test_accuracy'][-1]:.4f}"
            )
            self.save_model()
        self._plot_metrics()


    def save_model(self):
        _, state = nnx.split(self.model)
        self.checkpointer.save(self.checkpoints_dir, state, force=True)
        print(f"Checkpoint saved to {self.checkpoints_dir}")


@hydra.main(version_base=None, config_path='..', config_name='config')
def main(cfg: DictConfig):
    coach = Coach(cfg)
    coach.train()


if __name__ == '__main__':
    main()