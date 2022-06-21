# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple, Type, Union

import torch
from torch.nn import Identity, ReLU, Sigmoid, Tanh

from ..datasets.variables import Variables
from ..utils.torch_utils import get_torch_device
from .decoder import Decoder
from .encoder import Encoder
from .torch_vae import TorchVAE


class VAE(TorchVAE):
    """
    Variational autoencoder.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        encoder_layers: List[int],
        latent_dim: int,  # Encoder options
        decoder_layers: List[int],
        decoder_variances: float,  # Decoder options
        categorical_likelihood_coefficient: float,
        kl_coefficient: float,
        variance_autotune: bool = False,
        metadata_filepath: Optional[str] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        non_linearity: Optional[Union[str, Type[ReLU], Type[Sigmoid], Type[Tanh], Type[Identity]]] = "ReLU",
        activation_for_continuous: Optional[
            Union[str, Type[ReLU], Type[Sigmoid], Type[Tanh], Type[Identity]]
        ] = "Sigmoid",
        init_method: Optional[str] = "default",
        **kwargs,
    ) -> None:
        """
        Args:
            model_id (str): Unique model ID for referencing this model instance.
            variables (Variables): Information about variables/features used
                by this model.
            save_dir (str): Location to save any information about this model, including training data.
                be created if it doesn't exist.
            device (`torch.device`): Device to load model to.

            input_dim (int): Dimension of input data.
            output_dim (int): Dimension of data output.

            non_linearity (str): Non linear activation function used between Linear layers. Defaults to ReLU.
            activation_for_continuous (str): activation function for continuous variable outputs. Defaults to Sigmoid.
            init_method (str): initialization method

            encoder_layers (list of int): Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            latent_dim (int): Dimension of output latent space.

            decoder_layers (list of int): Sizes of internal hidden layers. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
            decoder_variances (float): Output variance to use.

            categorical_likelihood_coefficient (float): coefficient for balancing likelihoods
            kl_coefficient (float): coefficient for KL terms
            variance_autotune (bool): automatically tune variances or not

        """

        if input_dim is None:
            input_dim = variables.num_processed_cols
        if output_dim is None:
            output_dim = variables.num_processed_non_aux_cols
        if metadata_filepath is not None:
            raise ValueError

        for k in kwargs:
            # TODO happens when we pass metadata to this constructor
            # Should get fixed when we replace the many keyword args with a single
            # config object
            warnings.warn(f"Ignoring keyword argument {k}")
        non_linearity = self.get_activation_function(non_linearity)

        encoder = Encoder(
            input_dim, encoder_layers, latent_dim, device, non_linearity=non_linearity, init_method=init_method
        )

        activation_for_continuous = self.get_activation_function(activation_for_continuous)

        decoder = Decoder(
            latent_dim,
            output_dim,
            variables,
            decoder_layers,
            decoder_variances,
            device,
            variance_autotune=variance_autotune,
            non_linearity=non_linearity,
            activation_for_continuous=activation_for_continuous,
            init_method=init_method,
        )
        super().__init__(
            model_id,
            variables,
            save_dir,
            device,
            encoder,
            decoder,
            categorical_likelihood_coefficient=categorical_likelihood_coefficient,
            kl_coefficient=kl_coefficient,
        )

    # CLASS METHODS #
    @classmethod
    def name(cls) -> str:
        return "vae"

    @classmethod
    def _load(
        cls,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: Union[str, int, torch.device],
        **kwargs,
    ) -> VAE:
        """
        Load an instance of `VAE` class.

        Args:
            model_id: Unique model ID for referencing this model instance.
            variables (Variables): Information about variables/features used
                by this model.
            save_dir: Location to save any information about this model, including training data.
                be created if it doesn't exist.
            device: Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a device ID
                (e.g. 0 or 1 on a two-GPU machine). Can also pass a torch.device directly.
            random_seed: Random seed to set before creating model. Defaults to 0.
            **kwargs: Any other arguments needed by the concrete class. Defaults can be specified in the concrete class.
                e.g. ..., embedding_dim, latent_dim=20, ...

        Returns:
            Instance of `VAE` class.
        """
        torch_device = get_torch_device(device)

        model_path = os.path.join(save_dir, cls._model_file)

        model = cls(
            model_id,
            variables,
            save_dir,
            torch_device,
            input_dim=variables.num_processed_cols,
            output_dim=variables.num_processed_cols,
            **kwargs,
        )
        model.load_state_dict(torch.load(model_path))

        return model

    def encode(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run encoding part of the VAE.

        Args:
            input_tensors: Input tensors. Each with shape (batch_size, input_dim).

        Returns:
            mean, logvar: Latent space samples of shape (batch_size, latent_dim).
        """
        return self._encoder(input_tensors[0])

    def decode(self, data: torch.Tensor, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoding part of the VAE.

        Args:
            data: Input tensor with shape (batch_size, latent_dim).
            input_tensors: Input tensors. Each with shape (batch_size, input_dim).

        Returns:
            mean, logvar: Output of shape (batch_size, output_dim)
        """
        return self._decoder(data)

    @staticmethod
    def get_activation_function(non_linearity_strings):
        if non_linearity_strings == "ReLU":
            return ReLU
        elif non_linearity_strings == "Sigmoid":
            return Sigmoid
        elif non_linearity_strings == "Tanh":
            return Tanh
        elif non_linearity_strings == "Identity":
            return Identity
        else:
            return ReLU
