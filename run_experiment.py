"""
Train a model (defaults to PVAE) and optionally evaluate it.

To run:

The simplest way to run is using an existing UCI dataset:
(options: boston, energy, wine, bank) and run
e.g. python run_experiment.py boston

To overwrite hyperparameters, specify model config (-m) or inference config
(-ic) containing values to override.
e.g. python run_experiment.py boston -m parameters/model_config_sweep.json

To run on custom data, you will need to specify the data directory
e.g. python run_experiment.py csv -d data/dataset_name

To see information about other options, run this script with -h.
"""

import argparse

from dependency_injector.wiring import Provide, inject
import textwrap
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np

if __name__ == "__main__":
    from azua.experiment.azua_context import AzuaContext
    from azua.experiment.run_aggregation import run_aggregation
    from azua.experiment.run_single_seed_experiment import run_single_seed_experiment
    from azua.utils.configs import get_configs
    from azua.utils.run_utils import find_local_model_dir, create_models_dir
    from azua import models
else:
    from .azua.experiment.azua_context import AzuaContext
    from .azua.experiment.run_aggregation import run_aggregation
    from .azua.experiment.run_single_seed_experiment import run_single_seed_experiment
    from .azua.utils.configs import get_configs
    from .azua.utils.run_utils import find_local_model_dir, create_models_dir
    from .azua import models


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a partial VAE model.", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("dataset_name", help="Name of dataset to use.")
    parser.add_argument(
        "--data_dir", "-d", type=str, default="data", help="Directory containing saved datasets. Defaults to ./data",
    )
    parser.add_argument(
        "--model_type",
        "-mt",
        type=str,
        default="pvae",
        choices=[
            "pvae",
            "vaem",
            "vaem_predictive",
            "transformer_pvae",
            "transformer_encoder_pvae",
            "transformer_encoder_vaem",
            "bayesian_pvae",
            "mnar_pvae",
            "transformer_imputer",
            "graph_neural_network",
            "corgi",
            "graph_convolutional_network",
            "graphsage",
            "graph_attention_network",
            "gcmc",
            "grape",
            "deep_matrix_factorization",
            "vicause",
            "flow_vicause_param",
            "flow_vicause_inf",
            "min_imputing",
            "mean_imputing",
            "zero_imputing",
            "majority_vote",
            "mice",
            "missforest",
            "pc",
            "notears_linear",
            "notears_mlp",
            "notears_sob",
            "grandag",
            "icalingam",
        ],
        help=textwrap.dedent(
            """Type of model to train.
            pvae: Default Partial VAE model from EDDI paper. For the IWAE version, set the use_importance_sampling flag in the model config .
            vaem: Default model from VAEM paper.
            vaem_predictive: The target is treated as a separate node p(y|x,h).
            transformer_pvae: Partial-VAE with a transformer in encoder and decoder.
            transformer_encoder_pvae: Partial-VAE with set transformer in encoder.
            transformer_encoder_vaem: VAEM with set transformer in encoder.
            bayesian_pvae: Weights as stochastic variable. Using inducing point VI to infer.
            mnar_pvae: Default uses identifiable (P)-VAE instead of vanilla VAE for MNAR case. 
                With different settings (relative to the default setting for mnar_pvae), we can recover different models:
                     use_importance_sampling = False -> mnar_pvae without importance sampling
                     mask_net_coefficient = 0 -> MISSIWAE
                     mask_net_coefficient = 0, use_importance_sampling = False -> PVAE
                     use_prior_net_to_train = False, latent_connection = False -> Not-MissIWAE
                     use_prior_net_to_train = False -> Not-MIssIWAE with latent connection
                     use_prior_net_to_train = False, latent_connection = False, use_importance_sampling = False -> Not-MissPVAE
                     use_prior_net_to_train = False,  use_importance_sampling = False -> Not-MissPVAE with latent connection
            transformer_imputer: Directly using transformer for imputation. Only works with active learning strategies variance or rand.
            different graph_neural_network models: Graph neural network-based models for missing data imputation. Does not work with active learning. 
                Changing the model configuration leads to different GNN recommendation models developed over the past years
                (for detailed model configurations, please refer to the model_config json files in parameters/defaults/ with 
                the specified model names): 
                    CoRGi:
                        CoRGi is a GNN model that considers the rich data within nodes in the context of their neighbors. 
                        This is achieved by endowing CORGI’s message passing with a personalized attention
                        mechanism over the content of each node. This way, CORGI assigns user-item-specific 
                        attention scores with respect to the words that appear in items.
                    Graph Convolutional Network (GCN):
                        As a default, "average" is used for the aggregation function 
                        and nodes are randomly initialized. 
                        We adopt dropout with probability 0.5 for node embedding updates
                        as well as for the prediction MLPs.
                    GRAPE:
                        GRAPE is a GNN model that employs edge embeddings.
                        Also, it adopts edge dropouts that are applied throughout all message-passing layers.
                        Compared to the GRAPE proposed in the oroginal paper, because of the memory issue, 
                        we do not initialize nodes with one-hot vectors nor constants (ones).
                    Graph Convolutional Matrix Completion (GC-MC):
                        Compared to GCN, this model has a single message-passing layer. 
                        Also, For classification, each label is endowed with a separate message passing channel.
                        Here, we do not implement the weight sharing.
                    GraphSAGE:
                        GraphSAGE extends GCN by allowing the model to be trained on the part of the graph, 
                        making the model to be used in inductive settings.
                    Graph Attention Network (GAT):
                        During message aggregation, GAT uses the attention mechanism to allow the target nodes to
                        distinguish the weights of multiple messages from the source nodes for aggregation.

            deep_matrix_factorization: Deep matrix factorization (DMF) for missing data imputation. 
                It's a deterministic model that uses an arbitrary value to fill in  the missing entries of the imputation matrix. 
                The value that replaces NaN can be assigned in missing_fill_val in the training_hyperparams of the model config file. 
                Does not work with active learning.
            vicause: Simultaneous missing value imputation and causal discovery using neural relational inference.
            flow_vicause_inf: Causal discovery using flow based model while doing approximate inference over the adjacency matrix.
            flow_vicause_param: Like flow_vicause but treating the weighted adjacency as a parameter.
            min_imputing: Impute missing values using the minimum value for the corresponding variable.
            mean_imputing: Impute missing values using the mean observed value for the corresponding variable.
            zero_imputing: Impute missing values using the value 0.
            majority_vote: Impute missing values using the most common observed value for the feature.
            mice: Impute missing values using the iterative method Multiple Imputation by Chained Equations (MICE).
            missforest: Impute missing values using the iterative random forest method MissForest.
            notears_linear: Linear version of notears algorithm for causal discovery (https://arxiv.org/abs/1803.01422).
            notears_mlp: Nonlinear version (MLP) of notears algorithm for causal discovery (https://arxiv.org/abs/1909.13189).
            notears_sob: Nonlinear version (Sobolev) of notears algorithm for causal discovery (https://arxiv.org/abs/1909.13189).
            grandag: GraNDAG algorithm for causal discovery using MLP as nonlinearities (https://arxiv.org/abs/1906.02226).
            pc: PC algorithm for causal discovery (https://arxiv.org/abs/math/0510436).
            icalingam: ICA based causal discovery (https://dl.acm.org/doi/10.5555/1248547.1248619).
            """
        ),
    )
    parser.add_argument("--model_dir", "-md", default=None, help="Directory containing the model.")
    parser.add_argument("--model_config", "-m", type=str, help="Path to JSON containing model configuration.")
    parser.add_argument("--dataset_config", "-dc", type=str, help="Path to JSON containing dataset configuration.")
    parser.add_argument("--impute_config", "-ic", type=str, help="Path to JSON containing impute configuration.")
    parser.add_argument("--objective_config", "-oc", type=str, help="Path to JSON containing objective configuration.")

    # Whether or not to run inference and active learning
    parser.add_argument("--run_inference", "-i", action="store_true", help="Run inference after training.")
    parser.add_argument("--extra_eval", "-x", action="store_true", help="Run extra eval tests that take longer.")
    parser.add_argument(
        "--max_steps", "-ms", type=int, default=np.inf, help="Maximum number of active learning steps to take."
    )
    parser.add_argument(
        "--max_al_rows",
        "-mar",
        type=int,
        default=np.inf,
        help="Maximum number of rows on which to perform active learning.",
    )
    parser.add_argument(
        "--active-learning",
        "-a",
        nargs="+",
        choices=[
            "eddi",
            "eddi_mc",
            "eddi_rowwise",
            "rand",
            "sing",
            "ei",
            "nm_ei",
            "b_ei",
            "k_ei",
            "bin",
            "gls",
            "rand_im",
            "variance",
            "all",
        ],
        help="""Run active learning after train and test. 
                                ei = expected improvement, 
                                nm_ei = non myopic ei,
                                b_ei = batch ei
                                k_ei = k selection with ei
                                bin = binoculars algorithm
                                gls = glasses algorithm 
                                rand_im = random with imputation""",
    )
    parser.add_argument(
        "--users_to_plot", "-up", default=[0], nargs="+", help="Indices of users to plot info gain bar charts for."
    )
    # Whether or not to evaluate causal discovery (only vicause at the moment)
    parser.add_argument(
        "--eval_causal_discovery",
        "-c",
        action="store_true",
        help="Whether to evaluate causal discovery against a ground truth during evaluation.",
    )
    # Other options for saving output.

    parser.add_argument("--output_dir", "-o", type=str, default="runs", help="Output path. Defaults to ./runs/.")
    parser.add_argument("--name", "-n", type=str, help="Tag for this run. Output dir will start with this tag.")
    parser.add_argument(
        "--device", "-dv", default="cpu", help="Name (e.g. 'cpu', 'gpu') or ID (e.g. 0 or 1) of device to use."
    )
    parser.add_argument("--tiny", action="store_true", help="Use this flag to do a tiny run for debugging")
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for training. If not provided, a random seed will be taken from the model config JSON",
    )
    parser.add_argument(
        "--default_configs_dir",
        "-dcd",
        type=str,
        default="configs",
        help="Directory containing configs. Defaults to ./configs",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    """
    assert os.path.isdir(args.data_dir), f"{args.data_dir} is not a directory."

    # Config files
    if args.model_config is not None and not os.path.isfile(args.model_config):
        raise ValueError("Model config file %s does not exist." % args.model_config)

    if args.dataset_config is not None and not os.path.isfile(args.dataset_config):
        raise ValueError("Dataset config file %s does not exist." % args.dataset_config)

    if args.impute_config is not None and not os.path.isfile(args.impute_config):
        raise ValueError("Imputation config file %s does not exist." % args.impute_config)

    if args.objective_config is not None and not os.path.isfile(args.objective_config):
        raise ValueError("Objective config file %s does not exist." % args.objective_config)


def split_configs(
    model_config: Dict[str, Any], dataset_config: Dict[str, Any]
) -> List[Tuple[int, Dict[str, Any], Any, Dict[str, Any]]]:
    model_seeds = model_config["random_seed"]
    if not isinstance(model_seeds, list):
        model_seeds = [model_seeds]

    dataset_config_random_seed = dataset_config["random_seed"]
    if isinstance(dataset_config_random_seed, list):
        dataset_seeds = dataset_config_random_seed
    else:
        dataset_seeds = [dataset_config_random_seed]

    configs = []
    for dataset_seed in dataset_seeds:
        for model_seed in model_seeds:
            new_model_config = model_config.copy()
            new_model_config["random_seed"] = model_seed

            new_dataset_config = dataset_config.copy()
            new_dataset_config["random_seed"] = dataset_seed

            configs.append((model_seed, new_model_config, dataset_seed, new_dataset_config))

    return configs


@inject
def run_experiment(
    dataset_name,
    model_config_path,
    data_dir="data",
    model_type="pvae",
    model_dir=None,
    model_id=None,
    dataset_config_path: Optional[str] = None,
    impute_config_path: Optional[str] = None,
    objective_config_path: Optional[str] = None,
    run_inference=False,
    extra_eval=False,
    active_learning=None,
    max_steps=np.inf,
    max_al_rows=np.inf,
    eval_causal_discovery=False,
    output_dir="runs",
    device="cpu",
    name=None,
    quiet=False,
    active_learning_users_to_plot=None,
    tiny: bool = False,
    random_seed: Optional[int] = None,
    default_configs_dir: Optional[str] = "configs",
    azua_context: AzuaContext = Provide[AzuaContext],
):
    # Load configs
    model_config, train_hypers, dataset_config, impute_config, objective_config = get_configs(
        model_type=model_type,
        dataset_name=dataset_name,
        override_dataset_path=dataset_config_path,
        override_model_path=model_config_path,
        override_impute_path=impute_config_path,
        override_objective_path=objective_config_path,
        default_configs_dir=default_configs_dir,
    )
    if random_seed is not None:
        model_config["random_seed"] = random_seed
    # Change active learning method if imputation_method is not none and all methods are run
    if active_learning is not None and objective_config["imputation_method"] is not None:
        if active_learning == ["eddi", "rand", "sing"]:
            active_learning = ["rand_im", "ei", "k_ei", "b_ei", "bin", "gls"]

    # Create directories, record arguments and configs
    try:
        models_dir = create_models_dir(output_dir=output_dir, name=name)
    except FileExistsError:
        # Timestamp has 1-second resolution, causing trouble if we try to run several times in quick succession
        time.sleep(1)
        models_dir = create_models_dir(output_dir=output_dir, name=name)
    experiment_name = f"{dataset_name}.{model_type}" if name is None else name
    metrics_logger = azua_context.metrics_logger()
    aml_tags = {
        "model_type": model_type,
        "dataset_name": dataset_name,
        "model_config_path": model_config_path,
        "dataset_config_path": dataset_config_path,
        "impute_config_path": impute_config_path,
        "objective_config_path": objective_config_path,
        "run_inference": run_inference,
        "active_learning": active_learning,
        "eval_causal_discovery": eval_causal_discovery,
        "device": device,
        "run_train": model_id is None,
        "model_config": model_config,
        "dataset_config": dataset_config,
        "train_hypers": train_hypers,
        "impute_config": impute_config,
        "objective_config": objective_config,
    }
    metrics_logger.set_tags(aml_tags)

    # Make many model files with diff seed for each.
    configs = split_configs(model_config, dataset_config)
    metrics_logger.set_tags({"num_samples": len(configs)})

    pipeline = azua_context.pipeline()
    pipeline_creation_mode = pipeline is not None
    if pipeline_creation_mode:
        train_step_outputs: List[Any] = []
    for model_seed, model_config, dataset_seed, dataset_config in configs:
        # TODO: group parameters
        kwargs_file = azua_context.aml_step(run_single_seed_experiment, pipeline_creation_mode)(
            dataset_name=dataset_name,
            data_dir=data_dir,
            model_type=model_type,
            model_dir=model_dir,
            model_id=model_id,
            run_inference=run_inference,
            extra_eval=extra_eval,
            active_learning=active_learning,
            max_steps=max_steps,
            max_al_rows=max_al_rows,
            eval_causal_discovery=eval_causal_discovery,
            device=device,
            quiet=quiet,
            active_learning_users_to_plot=active_learning_users_to_plot,
            tiny=tiny,
            dataset_config=dataset_config,
            dataset_seed=dataset_seed,
            model_config=model_config,
            train_hypers=train_hypers,
            impute_config=impute_config,
            objective_config=objective_config,
            output_dir=models_dir,
            experiment_name=experiment_name,
            model_seed=model_seed,
            aml_tags=aml_tags,
        )
        if pipeline_creation_mode:
            step_ouput = pipeline.add_step(
                script_name="run_experiment_step.py",  # TODO: remove
                arguments=["--step", "single_seed_experiment", "--kwargs", kwargs_file],
                step_name=experiment_name,
                output_dir=f"outputs{len(train_step_outputs)}",  # specifying unique output_dir, see #16728
            )
            train_step_outputs.append(step_ouput)

    # For local runs, temporary logic to extract input dirs given models_dir
    # Going forward (i.e. once we use AML pipeline for local runs),
    # inputs dirs will be explicitly specified (as they are in remote runs)
    input_dirs = [f.path for f in os.scandir(models_dir) if f.is_dir()]

    kwargs_file = azua_context.aml_step(run_aggregation, pipeline_creation_mode)(
        input_dirs=input_dirs, output_dir=models_dir, experiment_name=experiment_name, aml_tags=aml_tags
    )
    if pipeline_creation_mode:
        pipeline.add_step(
            script_name="run_experiment_step.py",  # TODO: remove
            arguments=["--step", "aggregation", "--kwargs", kwargs_file, "--input_dirs"] + train_step_outputs,
            inputs=train_step_outputs,
            step_name=experiment_name,
        )
        pipeline.run(aml_tags)

    # TODO this return value is provided only for the sake of end_to_end tests. Remove it?
    return models_dir


def run_experiment_on_parsed_args(args: argparse.Namespace):

    # Expand args for active learning
    if args.active_learning is not None and "all" in args.active_learning:
        args.active_learning = ["eddi", "rand", "sing"]

    # Get model_dir, model_id from model_dir
    if args.model_dir is not None:
        args.model_dir, args.model_id = find_local_model_dir(args.model_dir)
    else:
        args.model_id = None

    run_experiment(
        dataset_name=args.dataset_name,
        model_config_path=args.model_config,
        data_dir=args.data_dir,
        model_type=args.model_type,
        model_dir=args.model_dir,
        model_id=args.model_id,
        dataset_config_path=args.dataset_config,
        impute_config_path=args.impute_config,
        objective_config_path=args.objective_config,
        run_inference=args.run_inference,
        extra_eval=args.extra_eval,
        active_learning=args.active_learning,
        max_steps=args.max_steps,
        max_al_rows=args.max_al_rows,
        eval_causal_discovery=args.eval_causal_discovery,
        output_dir=args.output_dir,
        device=args.device,
        name=args.name,
        quiet=False,
        active_learning_users_to_plot=args.users_to_plot,
        tiny=args.tiny,
        random_seed=args.random_seed,
        default_configs_dir=args.default_configs_dir,
    )


def main(user_args):
    parser = get_parser()
    args = parser.parse_args(user_args)
    validate_args(args)

    azua_context = AzuaContext()
    azua_context.wire(modules=[sys.modules[__name__]], packages=[models])
    run_experiment_on_parsed_args(args)


if __name__ == "__main__":
    main(sys.argv[1:])
