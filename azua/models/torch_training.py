from __future__ import annotations
from ..models.imodel import IModelWithReconstruction
from typing import Union, Optional, Callable, Dict, List
import os
import time
import logging
from ..utils.imputation import plot_pairwise_comparison

import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from ..experiment.azua_context import AzuaContext
from ..utils.torch_utils import create_dataloader, set_random_seeds
from ..models.torch_training_types import LossResults, LossConfig, VAELossResults, EpochMetrics, VAEEpochMetrics
from ..utils.helper_functions import maintain_random_state
from ..models.torch_model import TorchModel
from ..datasets.dataset import Dataset, SparseDataset
from ..utils.io_utils import save_json
from ..utils.exceptions import ValidationDataNotAvailable
from ..utils.metrics import create_latent_distribution_plots

from dataclasses import asdict, fields
from dependency_injector.wiring import inject, Provide


@inject
def train_model(
    model,
    dataset: Union[Dataset, SparseDataset],
    train_output_dir: str,
    report_progress_callback: Optional[Callable[[str, int, int], None]],
    learning_rate: float,
    batch_size: int,
    iterations: int,
    epochs: int,
    rewind_to_best_epoch: bool,
    max_p_train_dropout: Optional[float] = None,
    lr_warmup_epochs: int = 0,
    use_lr_decay: bool = False,
    early_stopping_patience_epochs: Optional[int] = None,
    improvement_ratio_threshold: float = 1e-3,
    save_latent_plots_period_epochs: Optional[int] = None,
    score_reconstruction: Optional[bool] = None,
    score_imputation: Optional[bool] = None,
    extra_eval: Optional[bool] = False,
    azua_context: AzuaContext = Provide[AzuaContext],
) -> Dict[str, List[float]]:
    """
    Train the model using the given data.

    Args:
        model: model to train
        dataset: Dataset with data and masks in processed form.
        train_output_dir (str): Path to save any training information to, including tensorboard summary
            files.
        report_progress_callback: Function to report model progress for API.
        learning_rate (float): Learning rate for Adam optimiser.
        batch_size (int): Size of minibatches to use.
        iterations (int): Iterations to train for. -1 is all iterations per epoch.
        epochs (int): Number of epochs to train for.
        max_p_train_dropout (float): Maximum fraction of extra training features to drop for each row. 0 is none, 1 is all.
        rewind_to_best_epoch: if True, call model.save() when lowest validation loss is reached, and reload those parameters at the end of training.
        lr_warmup_epochs: number of epochs for learning rate warmup.
        use_lr_decay: Whether to, in addition to lr warmup, decay the lr proportionally to the inverse square root of the epoch number.
        early_stopping_patience_epochs: If validation loss does not improve for this many epochs, stop training. If None, training always continues
          for the full number of epochs. You must have validation data to use this option.
        improvement_ratio_threshold: The threshold of improvement ratio to determine early stopping. 
        save_latent_plots_period_epochs: create plots of latent space parameters at this interval.
        extra_eval: extra evaluation, creates pairwise plots
        
        
    Returns:
        train_results (dictionary): Train loss, KL divergence, and NLL for each epoch as a dictionary.
    """

    # Input checks
    if lr_warmup_epochs < 0:
        raise ValueError

    has_val_data = dataset.has_val_data
    if (rewind_to_best_epoch or (early_stopping_patience_epochs is not None)) and not has_val_data:
        raise ValidationDataNotAvailable("Early stopping requires validation data")

    loss_config = LossConfig(
        max_p_train_dropout=max_p_train_dropout,
        score_reconstruction=score_reconstruction,
        score_imputation=score_imputation,
    )

    model.validate_loss_config(loss_config)

    # Put model into train mode.
    model.train()

    writer = SummaryWriter(os.path.join(train_output_dir, "summary"), flush_secs=1)
    metrics_logger = azua_context.metrics_logger()
    logger = logging.getLogger()
    results_dict: Dict[str, List] = {"epoch_time": []}

    optimizer, lr_scheduler = create_optimizer_and_lr_scheduler(
        model, learning_rate=learning_rate, lr_warmup_epochs=lr_warmup_epochs, use_lr_decay=use_lr_decay
    )

    train_dataloader, val_dataloader = create_train_and_val_dataloaders(
        dataset, batch_size=batch_size, iterations=iterations
    )

    best_val_metrics = None
    best_val_loss = np.nan
    best_epoch = 0
    improvement_ratio = 0.5
    best_delta_val_loss = np.nan
    best_delta_epoch = 0
    is_quiet = logger.level > logging.INFO
    for epoch in trange(epochs, desc="Epochs", disable=is_quiet):
        epoch_start_time = time.time()
        train_metrics = _one_epoch(
            model=model, dataloader=train_dataloader, is_quiet=is_quiet, optimizer=optimizer, loss_config=loss_config
        )
        if has_val_data:
            with maintain_random_state():
                # Use the same random seed to evaluate validation loss on each epoch.
                set_random_seeds(0)
                with torch.no_grad():
                    val_metrics = _one_epoch(
                        model=model, dataloader=val_dataloader, is_quiet=is_quiet, loss_config=loss_config
                    )
            if epoch == 0 or val_metrics.loss < best_val_loss:
                best_val_metrics = val_metrics
                best_val_loss = best_val_metrics.loss
                best_epoch = epoch
                if epoch > 0:
                    improvement_ratio = (best_delta_val_loss - val_metrics.loss) / abs(best_delta_val_loss)
                    assert improvement_ratio > 0
                if epoch == 0 or improvement_ratio > improvement_ratio_threshold:
                    best_delta_val_loss = val_metrics.loss
                    best_delta_epoch = epoch
                if rewind_to_best_epoch:
                    # Save model.
                    model.save()

        lr_scheduler.step()

        # Save useful quantities.
        epoch_time = time.time() - epoch_start_time

        def _log_epoch_metrics(metrics: EpochMetrics, train_or_val: str):
            for k, value in asdict(metrics).items():
                name = f"train/{k}-{train_or_val}"
                writer.add_scalar(name, value, epoch)  # tensorboard
                metrics_logger.log_dict({name: value})  # AzureML
                if name not in results_dict:
                    results_dict[name] = []
                results_dict[name].append(value)  # for JSON file

        _log_epoch_metrics(train_metrics, "train")
        if has_val_data:
            _log_epoch_metrics(val_metrics, "val")

        results_dict["epoch_time"].append(epoch_time)

        lr = optimizer.param_groups[0]["lr"]
        metrics_logger.log_dict({"train/epoch": epoch, "train/epoch_time": epoch_time, "train/lr": lr})

        if report_progress_callback:
            report_progress_callback(model.model_id, epoch + 1, epochs)

        if (save_latent_plots_period_epochs is not None) and not (epoch % save_latent_plots_period_epochs):
            create_latent_distribution_plots(
                model=model, dataloader=train_dataloader, output_dir=train_output_dir, epoch=epoch, num_points_plot=1000
            )
        if early_stopping_patience_epochs is not None and epoch >= best_delta_epoch + early_stopping_patience_epochs:
            # Patience has run out. Stop training.
            logger.info(
                f"Validation loss has not improved more than {100*improvement_ratio_threshold}% for {early_stopping_patience_epochs} epochs. Exiting early at epoch {epoch}."
            )
            break

        if np.isnan(train_metrics.loss):
            logger.info("Training loss is NaN. Exiting early.")
            break
    if rewind_to_best_epoch:
        logger.info(f"Best model found at epoch {best_epoch}, with val metrics {best_val_metrics}")

        # Reload best parameters
        model_path = os.path.join(model.save_dir, model._model_file)
        model.load_state_dict(torch.load(model_path))
    else:
        # TODO save full checkpoints so that training can be resumed.
        logger.info(f"Saving after {epochs} epochs.")
        model.save()

    writer.close()

    # TODO: add support for PredictiveVAE by reconstruct() using input_tensors. Currently, it will fail instead
    if extra_eval and isinstance(model, IModelWithReconstruction):
        # Create pair plots
        # TODO: We also create pairwise plots as part of evaluation. Possible unify two

        train_data, train_mask = dataset.train_data_and_mask
        # We need special treatment for (Predictive)VAEM models, as reverting data is not supported
        # TODO: think about support of revert_data for dependency network's input (could also do squashing/unsquashing)
        is_vaem_model = all([variable.name.startswith("z_") for variable in model.variables])
        unprocessed_ground_truth = model.data_processor.revert_data(train_data) if not is_vaem_model else train_data
        plot_pairwise_comparison(
            unprocessed_ground_truth, model.variables, filename_suffix="ground_truth_data", save_dir=train_output_dir
        )

        # Auxilary method for calling reconstruct() and making pairwise plot
        def reconstruct_and_plot_pairwise(data: np.ndarray, mask: np.ndarray, filename_suffix: str):
            (dec_mean, dec_logvar), _, _ = model.reconstruct(torch.Tensor(data), torch.Tensor(mask))
            # TODO: add sampling?
            generated_values = (
                model.data_processor.revert_data(dec_mean.detach().numpy())
                if not is_vaem_model
                else dec_mean.detach().numpy()
            )
            plot_pairwise_comparison(
                generated_values, model.variables, filename_suffix=filename_suffix, save_dir=train_output_dir
            )

        reconstruct_and_plot_pairwise(train_data, np.zeros_like(train_mask, dtype=bool), "data_generation")
        reconstruct_and_plot_pairwise(train_data, train_mask, "data_reconstruction")

    # Save train results.
    # TODO it would make more sense to save this in train_output_dir, but for consistency with other models,
    # leave this in model.save_dir for now.
    train_results_save_path = os.path.join(model.save_dir, "training_results_dict.json")
    save_json(results_dict, train_results_save_path)

    return results_dict


def create_train_and_val_dataloaders(dataset: Union[Dataset, SparseDataset], *, batch_size, iterations):
    dataloader = create_dataloader(
        *dataset.train_data_and_mask, batch_size=batch_size, iterations=iterations, sample_randomly=True
    )
    val_dataloader: Optional[DataLoader]
    if dataset.has_val_data:
        val_dataloader = create_dataloader(
            *dataset.val_data_and_mask, batch_size=batch_size, iterations=iterations, sample_randomly=True
        )
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def create_optimizer_and_lr_scheduler(model, *, learning_rate: float, lr_warmup_epochs: int, use_lr_decay: bool):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def scale_lr(epoch):
        # First call to this is with epoch=0
        if use_lr_decay:
            return min((epoch + 1) ** (-0.5), (epoch + 1) * (lr_warmup_epochs + 1) ** (-1.5))
        else:
            return min((epoch + 1) / (lr_warmup_epochs + 1), 1)

    lr_scheduler = LambdaLR(optimizer, scale_lr)
    return optimizer, lr_scheduler


def _one_epoch(
    model: TorchModel,
    dataloader,
    is_quiet: bool,
    loss_config: LossConfig,
    optimizer: Optional[torch.optim.Adam] = None,
) -> Union[EpochMetrics, VAEEpochMetrics]:
    # Run a single epoch of training
    train = optimizer is not None

    accumulated_batch_results: Optional[EpochMetrics] = None
    for input_tensors in tqdm(dataloader, desc="Batches", disable=is_quiet):
        input_tensors = [x.to(model.get_device()) for x in input_tensors]
        batch_size = input_tensors[0].shape[0]
        batch_start_time = time.time()
        if train:
            assert optimizer is not None  # for mypy
            optimizer.zero_grad()

        loss_results = model.loss(loss_config, input_tensors)

        if train:
            assert optimizer is not None  # for mypy
            (loss_results.loss / batch_size).backward()
            optimizer.step()

        batch_results = convert_from_tensors_and_add_time(loss_results, inner_epoch_time=time.time() - batch_start_time)
        if accumulated_batch_results is None:
            accumulated_batch_results = batch_results
        else:
            accumulated_batch_results = accumulated_batch_results + batch_results

    if accumulated_batch_results is None:
        raise ValueError("There were no batches of data")
    else:
        return convert_accumulated_to_average(accumulated_batch_results)


def convert_from_tensors_and_add_time(
    tensor_results: Union[LossResults, VAELossResults], inner_epoch_time: float
) -> Union[EpochMetrics, VAEEpochMetrics]:
    # Convert scalar tensors to plain numbers, and add timing information.
    type_map = {LossResults: EpochMetrics, VAELossResults: VAEEpochMetrics}
    tensor_results_as_dict = {k.name: getattr(tensor_results, k.name) for k in fields(tensor_results)}
    return type_map[type(tensor_results)](
        inner_epoch_time=inner_epoch_time,
        **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in tensor_results_as_dict.items()},
    )


def convert_accumulated_to_average(
    epoch_metrics: Union[EpochMetrics, VAEEpochMetrics]
) -> Union[EpochMetrics, VAEEpochMetrics]:
    as_dict = asdict(epoch_metrics)
    zero_mask_sum = as_dict["mask_sum"] == 0
    if zero_mask_sum:
        logger = logging.getLogger(__name__)
        logger.warning("Nothing was scored in NLL, so averaging metrics gives NaNs")
    for k in ["loss", "kl", "nll"]:
        if k in as_dict:
            if zero_mask_sum:
                as_dict[k] = np.nan
            else:
                as_dict[k] = as_dict[k] / as_dict["mask_sum"]
    return type(epoch_metrics)(**as_dict)
