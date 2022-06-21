import logging
import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from azua.models.partial_vae import PartialVAE
from azua.datasets.dataset import Dataset, SparseDataset
from azua.datasets.variables import Variable, Variables
from azua.experiment.steps.eval_step import run_eval_main
from azua.utils.io_utils import read_json_as, save_json


@pytest.fixture(scope="function")
def model(variables, tmpdir_factory):
    model_config = {
        "random_seed": 0,
        "input_dim": 3,
        "output_dim": 3,
        "embedding_dim": 1,
        "set_embedding_dim": 1,
        "set_embedding_multiply_weights": True,
        "encoding_function": "sum",
        "metadata_filepath": None,
        "encoder_layers": [1],
        "latent_dim": 1,
        "decoder_layers": [1],
        "decoder_variances": 1.0,
        "categorical_likelihood_coefficient": 1.0,
        "kl_coefficient": 1.0,
        "set_encoder_type": "default",
        "variance_autotune": False,
        "use_importance_sampling": False,
    }
    return PartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("cts_input", True, "continuous", 0.0, 5.0),
            Variable("binary_input", True, "binary", 0, 1),
            Variable("cts_target", False, "continuous", 0.0, 5.0),
        ]
    )


def test_run_eval_main_user_id_too_high(model, variables):
    logger = logging.getLogger()
    impute_config = {"sample_count": 100, "batch_size": 100}
    objective_config = {"sample_count": 100, "use_vamp_prior": True, "imputation_method": None, "batch_size": 100}
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((1, 3)),
        test_mask=np.ones((1, 3), dtype=bool),
        variables=variables,
    )
    with pytest.raises(AssertionError):
        run_eval_main(
            logger=logger,
            model=model,
            dataset=dataset,
            vamp_prior_data=None,
            impute_config=impute_config,
            objective_config=objective_config,
            extra_eval=False,
            seed=0,
            user_id=1,
            metrics_logger=None,
            impute_train_data=True,
        )


def test_run_eval_main(model, variables):
    logger = logging.getLogger()
    impute_config = {"sample_count": 100, "batch_size": 100}
    objective_config = {"sample_count": 100, "use_vamp_prior": True, "imputation_method": None, "batch_size": 100}
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )

    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        logger=logger,
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        objective_config=objective_config,
        extra_eval=True,
        seed=0,
        user_id=1,
        metrics_logger=None,
        impute_train_data=True,
    )
    results = read_json_as(os.path.join(model.save_dir, "results.json"), dict)
    assert "RMSE" in results["train_data"]["cts_input"]
    assert "RMSE" in results["test_data"]["cts_input"]
    assert "RMSE" in results["val_data"]["cts_input"]
    target_results = read_json_as(os.path.join(model.save_dir, "target_results.json"), dict)
    assert "RMSE" in target_results["train_data"]["cts_target"]
    assert "RMSE" in target_results["test_data"]["cts_target"]
    assert "RMSE" in target_results["val_data"]["cts_target"]
    assert os.path.isfile(os.path.join(model.save_dir, "imputed_values_violin_plot_user1.png"))
    assert os.path.isfile(os.path.join(model.save_dir, "info_gain_vs_metric_drop.png"))


def test_run_eval_main_no_val_data(model, variables):
    logger = logging.getLogger()
    impute_config = {"sample_count": 100, "batch_size": 100}
    objective_config = {"sample_count": 100, "use_vamp_prior": True, "imputation_method": None, "batch_size": 100}
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=None,
        val_mask=None,
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )
    run_eval_main(
        logger=logger,
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        objective_config=objective_config,
        extra_eval=False,
        seed=0,
        user_id=1,
        metrics_logger=None,
        impute_train_data=True,
    )
    results = read_json_as(os.path.join(model.save_dir, "results.json"), dict)
    assert "RMSE" in results["train_data"]["cts_input"]
    assert "RMSE" in results["test_data"]["cts_input"]
    assert results["val_data"] == {}
    target_results = read_json_as(os.path.join(model.save_dir, "target_results.json"), dict)
    assert "RMSE" in target_results["train_data"]["cts_target"]
    assert "RMSE" in target_results["test_data"]["cts_target"]
    assert target_results["val_data"] == {}


def test_run_eval_main_impute_train_data_false(model, variables):
    logger = logging.getLogger()
    impute_config = {"sample_count": 100, "batch_size": 100}
    objective_config = {"sample_count": 100, "use_vamp_prior": True, "imputation_method": None, "batch_size": 100}
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )
    run_eval_main(
        logger=logger,
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        objective_config=objective_config,
        extra_eval=False,
        seed=0,
        user_id=1,
        metrics_logger=None,
        impute_train_data=False,
    )
    results = read_json_as(os.path.join(model.save_dir, "results.json"), dict)
    assert results["train_data"] == {}
    assert "RMSE" in results["test_data"]["cts_input"]
    assert "RMSE" in results["val_data"]["cts_input"]
    target_results = read_json_as(os.path.join(model.save_dir, "target_results.json"), dict)
    assert target_results["train_data"] == {}
    assert "RMSE" in target_results["test_data"]["cts_target"]
    assert "RMSE" in target_results["val_data"]["cts_target"]


def test_run_eval_main_no_target_var_idxs(tmpdir_factory):
    logger = logging.getLogger()
    impute_config = {"sample_count": 100, "batch_size": 100}
    objective_config = {"sample_count": 100, "use_vamp_prior": True, "imputation_method": None, "batch_size": 100}
    variables = Variables(
        [
            Variable("cts_input", True, "continuous", 0.0, 5.0),
            Variable("binary_input", True, "binary", 0, 1),
            Variable("cts_target", True, "continuous", 0.0, 5.0),
        ]
    )
    model_config = {
        "random_seed": 0,
        "input_dim": 3,
        "output_dim": 3,
        "embedding_dim": 1,
        "set_embedding_dim": 1,
        "set_embedding_multiply_weights": True,
        "encoding_function": "sum",
        "metadata_filepath": None,
        "encoder_layers": [1],
        "latent_dim": 1,
        "decoder_layers": [1],
        "decoder_variances": 1.0,
        "categorical_likelihood_coefficient": 1.0,
        "kl_coefficient": 1.0,
        "set_encoder_type": "default",
        "variance_autotune": False,
        "use_importance_sampling": False,
    }
    model = PartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )
    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712, "2": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        logger=logger,
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        objective_config=objective_config,
        extra_eval=False,
        seed=0,
        user_id=1,
        metrics_logger=None,
        impute_train_data=True,
    )
    assert os.path.isfile(os.path.join(model.save_dir, "difficulty.csv"))
    assert os.path.isfile(os.path.join(model.save_dir, "quality.csv"))


def test_run_eval_main_sparse_dataset(model, variables):
    logger = logging.getLogger()
    impute_config = {"sample_count": 100, "batch_size": 100}
    objective_config = {"sample_count": 100, "use_vamp_prior": True, "imputation_method": None, "batch_size": 100}
    dataset = SparseDataset(
        train_data=csr_matrix(np.ones((5, 3))),
        train_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        val_data=csr_matrix(np.ones((5, 3))),
        val_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        test_data=csr_matrix(np.ones((5, 3))),
        test_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        variables=variables,
    )

    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        logger=logger,
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        objective_config=objective_config,
        extra_eval=True,
        seed=0,
        user_id=1,
        metrics_logger=None,
        impute_train_data=True,
    )
    results = read_json_as(os.path.join(model.save_dir, "results.json"), dict)
    assert "RMSE" in results["train_data"]["cts_input"]
    assert "RMSE" in results["test_data"]["cts_input"]
    assert "RMSE" in results["val_data"]["cts_input"]
    target_results = read_json_as(os.path.join(model.save_dir, "target_results.json"), dict)
    assert "RMSE" in target_results["train_data"]["cts_target"]
    assert "RMSE" in target_results["test_data"]["cts_target"]
    assert "RMSE" in target_results["val_data"]["cts_target"]
    assert not os.path.isfile(os.path.join(model.save_dir, "imputed_values_violin_plot_user1.png"))
    assert os.path.isfile(os.path.join(model.save_dir, "info_gain_vs_metric_drop.png"))


def test_run_eval_main_sparse_dataset_no_target_variables(tmpdir_factory):
    logger = logging.getLogger()
    impute_config = {"sample_count": 100, "batch_size": 100}
    objective_config = {"sample_count": 100, "use_vamp_prior": True, "imputation_method": None, "batch_size": 100}
    variables = Variables(
        [
            Variable("cts_input", True, "continuous", 0.0, 5.0),
            Variable("binary_input", True, "binary", 0, 1),
            Variable("cts_target", True, "continuous", 0.0, 5.0),
        ]
    )
    model_config = {
        "random_seed": 0,
        "input_dim": 3,
        "output_dim": 3,
        "embedding_dim": 1,
        "set_embedding_dim": 1,
        "set_embedding_multiply_weights": True,
        "encoding_function": "sum",
        "metadata_filepath": None,
        "encoder_layers": [1],
        "latent_dim": 1,
        "decoder_layers": [1],
        "decoder_variances": 1.0,
        "categorical_likelihood_coefficient": 1.0,
        "kl_coefficient": 1.0,
        "set_encoder_type": "default",
        "variance_autotune": False,
        "use_importance_sampling": False,
    }
    model = PartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    dataset = SparseDataset(
        train_data=csr_matrix(np.ones((5, 3))),
        train_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        val_data=csr_matrix(np.ones((5, 3))),
        val_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        test_data=csr_matrix(np.ones((5, 3))),
        test_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        variables=variables,
    )
    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712, "2": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        logger=logger,
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        objective_config=objective_config,
        extra_eval=False,
        seed=0,
        user_id=1,
        metrics_logger=None,
        impute_train_data=True,
    )
    assert not os.path.isfile(os.path.join(model.save_dir, "difficulty.csv"))
    assert not os.path.isfile(os.path.join(model.save_dir, "quality.csv"))


def test_run_eval_main_sparse_dataset_no_targets_elements_split(tmpdir_factory):
    logger = logging.getLogger()
    impute_config = {"sample_count": 100, "batch_size": 100}
    objective_config = {"sample_count": 100, "use_vamp_prior": True, "imputation_method": None, "batch_size": 100}
    variables = Variables(
        [
            Variable("cts_input", True, "continuous", 0.0, 5.0),
            Variable("binary_input", True, "binary", 0, 1),
            Variable("cts_target", True, "continuous", 0.0, 5.0),
        ]
    )
    model_config = {
        "random_seed": 0,
        "input_dim": 3,
        "output_dim": 3,
        "embedding_dim": 1,
        "set_embedding_dim": 1,
        "set_embedding_multiply_weights": True,
        "encoding_function": "sum",
        "metadata_filepath": None,
        "encoder_layers": [1],
        "latent_dim": 1,
        "decoder_layers": [1],
        "decoder_variances": 1.0,
        "categorical_likelihood_coefficient": 1.0,
        "kl_coefficient": 1.0,
        "set_encoder_type": "default",
        "variance_autotune": False,
        "use_importance_sampling": False,
    }
    model = PartialVAE.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    dataset = SparseDataset(
        train_data=csr_matrix(1 - np.eye(5, 3)),
        train_mask=csr_matrix(~np.eye(5, 3, dtype=bool)),
        val_data=None,
        val_mask=None,
        test_data=csr_matrix(np.eye(5, 3)),
        test_mask=csr_matrix(np.eye(5, 3, dtype=bool)),
        variables=variables,
    )
    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712, "2": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        logger=logger,
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        objective_config=objective_config,
        extra_eval=False,
        split_type="elements",
        seed=0,
        user_id=1,
        metrics_logger=None,
        impute_train_data=True,
    )
    assert not os.path.isfile(os.path.join(model.save_dir, "difficulty.csv"))
    assert not os.path.isfile(os.path.join(model.save_dir, "quality.csv"))
