import os
from typing import Dict

import numpy as np
import torch

from ..datasets.variables import Variable, Variables
from .partial_vae import PartialVAE


# Auxilary class for creating dependency network in (Predictive)VAEM models:
# Managing parameters and variables conversion
class DependencyNetworkCreator:
    @staticmethod
    def create(
        variables: Variables,
        dependency_save_dir: str,
        device: torch.device,
        marginal_latent_dim: int,
        dep_network_config: Dict,
    ) -> PartialVAE:
        marginal_output_dim = variables.num_unprocessed_cols * marginal_latent_dim

        # We need to distinguish between loading/creating model, as variables.json
        # carries information on scale of z variables
        dep_network_vars_file = os.path.join(dependency_save_dir, "variables.json")
        if os.path.exists(dep_network_vars_file):
            dep_network_vars = Variables.create_from_json(dep_network_vars_file)
        else:
            dep_network_vars = DependencyNetworkCreator._create_dep_network_vars(variables, marginal_latent_dim)

        dep_network_config["input_dim"] = marginal_output_dim
        dep_network_config["output_dim"] = marginal_output_dim

        dependency_network = PartialVAE.create(
            model_id="dependency_network",
            save_dir=dependency_save_dir,
            variables=dep_network_vars,
            model_config_dict=dep_network_config,
            device=device,
        )
        return dependency_network

    @staticmethod
    def _create_dep_network_vars(variables: Variables, marginal_latent_dim: int) -> Variables:
        dep_network_variables = []
        for x_variable in variables:
            z_variable = Variable(
                query=x_variable.query,
                type="continuous",
                lower=np.nan,
                upper=np.nan,
                name="z_%s" % x_variable.name,
                overwrite_processed_dim=marginal_latent_dim,
            )
            dep_network_variables.append(z_variable)

        variables = Variables(dep_network_variables)

        return variables
