from ..datasets.dataset import CausalDataset
from dependency_injector.wiring import Provide, inject
from .azua_context import AzuaContext
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ..utils.io_utils import save_json, save_txt
from ..experiment.steps.active_learning_step import run_active_learning_main
from ..experiment.steps.train_step import run_train_main
from ..experiment.steps.eval_step import run_eval_main
from ..experiment.steps.step_func import load_data, preprocess_configs
from ..experiment.steps.eval_step import eval_treatment_effects, eval_causal_discovery
from ..models.models_factory import load_model
from ..models.imodel import IModelForObjective, IModelForImputation, IModelForCausalInference, IModelForInterventions
from ..models.transformer_imputer import TransformerImputer
import os
import time
import shutil


@inject
def run_single_seed_experiment(
    dataset_name: str,
    data_dir: str,
    model_type: str,
    model_dir: str,
    model_id: str,
    run_inference: bool,
    extra_eval: bool,
    active_learning: Optional[List[str]],
    max_steps: float,
    max_al_rows: float,
    causal_discovery: bool,
    treatment_effects: bool,
    device: str,
    quiet: bool,
    active_learning_users_to_plot,
    tiny: bool,
    dataset_config: Dict[str, Any],
    dataset_seed: Union[int, Tuple[int, int]],
    model_config: Dict[str, Any],
    train_hypers: Dict[str, Any],
    impute_config: Dict[str, Any],
    objective_config: Dict[str, Any],
    output_dir: str,
    experiment_name: str,
    model_seed: int,
    aml_tags: Dict[str, Any],
    logger_level: str,
    eval_likelihood: bool = True,
    azua_context: AzuaContext = Provide[AzuaContext],
):

    # Set up loggers
    logger = logging.getLogger()
    log_format = "%(asctime)s %(filename)s:%(lineno)d[%(levelname)s]%(message)s"
    if quiet:
        level = logging.ERROR
    else:
        level_dict = {
            "ERROR": logging.ERROR,
            "INFO": logging.INFO,
            "CRITICAL": logging.CRITICAL,
            "WARNING": logging.WARNING,
            "DEBUG": logging.DEBUG
        }
        level = level_dict[logger_level]
    logging.basicConfig(level=level, force=True, format=log_format)
    metrics_logger = azua_context.metrics_logger()
    metrics_logger.set_tags(aml_tags)
    running_times: Dict[str, float] = {}

    _clean_partial_results_in_aml_run(output_dir, logger, azua_context)

    # Log system's metrics
    system_metrics_logger = azua_context.system_metrics_logger()
    system_metrics_logger.start_log()

    # Load data
    logger.info("Loading data.")
    dataset = load_data(dataset_name, data_dir, dataset_seed, dataset_config, model_config, tiny)
    assert dataset.variables is not None

    # Preprocess configs based on args and dataset
    # Going forward, we probably want the preprocessing happen inside steps (e.g. in train_step)
    # TODO: Rethink how we work with tags before moving this logic into steps
    # TODO: remove data_dir param, and carry it in dataset object instead
    preprocess_configs(model_config, train_hypers, model_type, dataset, data_dir, tiny)

    # Loading/training model
    if model_id is not None:
        logger.info("Loading pretrained model")
        model = load_model(model_id, model_dir, device)
    else:
        start_time = time.time()
        model = run_train_main(
            logger=logger,
            model_type=model_type,
            output_dir=output_dir,
            variables=dataset.variables,
            dataset=dataset,
            device=device,
            model_config=model_config,
            train_hypers=train_hypers,
            metrics_logger=metrics_logger,
        )
        running_times["train/running-time"] = (time.time() - start_time) / 60
    save_json(dataset_config, os.path.join(model.save_dir, "dataset_config.json"))
    save_txt(dataset_name, os.path.join(model.save_dir, "dataset_name.txt"))

    # Imputation
    if run_inference:
        if not isinstance(model, IModelForImputation):
            raise ValueError("This model class does not support imputation.")
        # TODO 18412: move impute_train_data flag into each dataset's imputation config rather than hardcoding here
        impute_train_data = dataset_name not in [
            "chevron",
            "eedi_task_1_2_binary",
            "mnist",
            "neuropathic_pain",
            "eedi_task_3_4_topics",
            "neuropathic_pain_3",
            "neuropathic_pain_4",
        ]
        run_eval_main(
            logger=logger,
            model=model,
            dataset=dataset,
            vamp_prior_data=None,
            impute_config=impute_config,
            objective_config=objective_config,
            extra_eval=extra_eval,
            split_type=dataset_config.get("split_type", "rows"),
            seed=dataset_seed if isinstance(dataset_seed, int) else dataset_seed[0],
            metrics_logger=metrics_logger,
            impute_train_data=impute_train_data,
        )

    # Evaluate causal discovery (only for vicause at the moment)
    if causal_discovery:
        assert isinstance(model, IModelForCausalInference)
        causal_model = cast(IModelForCausalInference, model)
        eval_causal_discovery(logger, dataset, causal_model, metrics_logger)

    # Treatment effect estimation
    if treatment_effects:
        if not isinstance(model, IModelForInterventions):
            raise ValueError("This model class does not support treatment effect estimation.")
        if not isinstance(dataset, CausalDataset):
            raise ValueError("This dataset type does not support treatment effect estimation.")
        eval_treatment_effects(logger, dataset, model, metrics_logger, eval_likelihood)

    # Active learning
    if active_learning is not None:
        # TODO 'rand' active learning is valid for any imputation model, not just these two
        assert isinstance(model, (IModelForObjective, TransformerImputer))
        if "eddi" in active_learning or "sing" in active_learning or "cond_sing" in active_learning:
            assert isinstance(model, IModelForObjective)
        if "variance" in active_learning:
            assert isinstance(model, TransformerImputer)
        start_time = time.time()
        test_data, test_mask = dataset.test_data_and_mask
        assert test_data is not None
        assert test_mask is not None

        run_active_learning_main(
            logger,
            model,
            test_data,
            test_mask,
            None,
            active_learning,
            objective_config,
            impute_config,
            seed=model_seed,
            max_steps=max_steps,
            max_rows=max_al_rows,
            users_to_plot=active_learning_users_to_plot,
            metrics_logger=metrics_logger,
        )
        running_times["nbq/running-time"] = (time.time() - start_time) / 60

    # Log speed/system metrics
    system_metrics = system_metrics_logger.end_log()
    metrics_logger.log_dict(system_metrics)
    save_json(system_metrics, os.path.join(model.save_dir, "system_metrics.json"))
    metrics_logger.log_dict(running_times)
    save_json(running_times, os.path.join(model.save_dir, "running_times.json"))
    metrics_logger.finalize()

    _copy_results_in_aml_run(output_dir, azua_context)

    return model, model_config


def _clean_partial_results_in_aml_run(output_dir: str, logger: logging.Logger, azua_context: AzuaContext):
    if azua_context.is_azureml_run():
        # If node is preempted (e.g. long running exp), it's possible
        # that there will be some partial results created in output directory
        # Those partial results shouldn't be aggregated, thus remove them
        logger.info("Checking if partial outputs are present for the run (if AML node was preempted before).")
        if os.path.isdir(output_dir):
            logger.info("Partial results are present.")
            for folder in os.listdir(output_dir):
                if os.path.isdir(folder):
                    logger.info(f"Removing partial results' directory: {folder}.")
                    shutil.rmtree(folder)
                else:
                    logger.info(f"Removing partial results' file: {folder}.")
                    os.remove(folder)


def _copy_results_in_aml_run(output_dir: str, azua_context: AzuaContext):
    if azua_context.is_azureml_run():
        # Copy the results to 'outputs' dir so that we can easily view them in AzureML.
        # Workaround for port name collision issue in AzureML, which sometimes prevents us from setting outputs_dir='outputs'.
        # See #16728
        shutil.copytree(output_dir, "outputs", dirs_exist_ok=True)
