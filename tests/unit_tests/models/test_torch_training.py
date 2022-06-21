import sys
import numpy as np
import torch

from dependency_injector.wiring import Provide, inject

from azua import models as amodels
from azua.datasets.dataset import Dataset
from azua.datasets.variables import Variables
from azua.experiment.azua_context import AzuaContext
from azua.models.torch_training import create_optimizer_and_lr_scheduler, train_model
from azua.models.torch_model import TorchModel
from azua.models.torch_training_types import LossConfig, LossResults


class MockTorchModel(TorchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._param = torch.nn.Parameter(torch.zeros(1))

    def loss(self, loss_config: LossConfig, *_):
        return LossResults(loss=self._param, mask_sum=torch.tensor(1.0))

    @classmethod
    def name(cls):
        return "mock_torch_model"

    def run_train(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_new(cls, save_dir):
        return cls(model_id="model_id", variables=Variables([]), save_dir=save_dir, device=torch.device("cpu"))

    def validate_loss_config(self, *args, **kwargs):
        pass


def dummy_dataset():
    sz = (2, 3)
    return Dataset(np.zeros(sz), np.zeros(sz), np.zeros(sz), np.zeros(sz), None, None, variables=None)


def test_train_model(tmpdir):
    # Check train_model can be called without raising exceptions.
    azua_context = AzuaContext()
    azua_context.wire(modules=[sys.modules[__name__]], packages=[amodels])
    model = MockTorchModel.get_new(save_dir=tmpdir)
    train_model(
        model,
        dataset=dummy_dataset(),
        train_output_dir=tmpdir,
        report_progress_callback=None,
        learning_rate=1e-3,
        batch_size=13,
        iterations=3,
        rewind_to_best_epoch=False,
        early_stopping_patience_epochs=3,
        epochs=300,
    )


def test_create_optimizer_and_lr_scheduler(tmpdir):
    # Check the lr scheduler ramps up LR as expected.
    model = MockTorchModel.get_new(save_dir=tmpdir)
    optimizer, scheduler = create_optimizer_and_lr_scheduler(
        model, learning_rate=1.0, lr_warmup_epochs=10, use_lr_decay=False
    )
    for i in range(0, 10):
        assert optimizer.param_groups[0]["lr"] == (i + 1) / 11
        scheduler.step()
    i = 10
    assert optimizer.param_groups[0]["lr"] == (i + 1) / 11
    for _ in range(10, 20):
        assert optimizer.param_groups[0]["lr"] == 1.0
        scheduler.step()


def test_create_optimizer_and_lr_scheduler_lr_decay(tmpdir):
    # Check the lr scheduler ramps up and decays LR as expected.
    model = MockTorchModel.get_new(save_dir=tmpdir)
    optimizer, scheduler = create_optimizer_and_lr_scheduler(
        model, learning_rate=1.0, lr_warmup_epochs=10, use_lr_decay=True
    )
    scale = 11**-0.5
    for i in range(0, 10):
        assert np.isclose(optimizer.param_groups[0]["lr"], ((i + 1) / 11) * scale)
        scheduler.step()
    i = 10
    assert np.isclose(optimizer.param_groups[0]["lr"], ((i + 1) / 11) * scale)
    for i in range(10, 20):
        assert optimizer.param_groups[0]["lr"] == (i + 1) ** (-0.5)
        scheduler.step()


def test_early_stopping_stops_early(tmpdir):
    # If loss never improves, training should stop after (1 + early_stopping_patience_epochs) epochs
    azua_context = AzuaContext()
    azua_context.wire(modules=[sys.modules[__name__]], packages=[amodels])
    model = MockTorchModel.get_new(save_dir=tmpdir)
    train_results = train_model(
        model,
        dataset=dummy_dataset(),
        train_output_dir=tmpdir,
        report_progress_callback=None,
        learning_rate=0,
        batch_size=10,
        iterations=-1,
        rewind_to_best_epoch=False,
        early_stopping_patience_epochs=3,
        epochs=300,
    )
    assert len(train_results["train/loss-train"]) == 4


def test_early_stopping_does_not_stop(tmpdir):
    azua_context = AzuaContext()
    azua_context.wire(modules=[sys.modules[__name__]], packages=[amodels])
    model = MockTorchModel.get_new(save_dir=tmpdir)
    train_results = train_model(
        model,
        dataset=dummy_dataset(),
        train_output_dir=tmpdir,
        report_progress_callback=None,
        learning_rate=1.0,
        batch_size=10,
        iterations=-1,
        rewind_to_best_epoch=False,
        early_stopping_patience_epochs=3,
        epochs=10,
    )
    assert len(train_results["train/loss-train"]) == 10


# TODO test dataloader creation
# TODO test exceptions raised on bad inputs, e.g. rewind_to_best_epoch without validation data.
