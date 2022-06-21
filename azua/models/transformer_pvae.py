from typing import Optional, Tuple

import torch
from torch.nn import Sigmoid

from ..datasets.variables import Variables
from ..models.feature_embedder import FeatureEmbedder
from ..models.torch_training import train_model
from ..models.torch_training_types import LossConfig, VAELossResults
from ..models.transformer_set_encoder import ISAB, SAB, SetTransformer
from ..models.torch_model import ONNXNotImplemented
from ..utils.training_objectives import (
    get_input_and_scoring_processed_masks,
    kl_divergence,
    negative_log_likelihood,
)
from .decoder import FeaturewiseActivation
from .pvae_base_model import PVAEBaseModel
from .torch_vae import TorchVAE


class TransformerPVAE(PVAEBaseModel):
    """
    Partial VAE with a transformer as a an encoder and a decoder.

    The TransformerPVAE module takes care of feature embedding and output activations
    and uses a TransformerVAE internally.

    See the Set Transformer model https://arxiv.org/abs/1810.00825 for more details
    about the implementation of SAB and ISAB blocks.

    Args:
        model_id: Unique model ID for referencing this model instance.
        variables: Information about variables/features used by this model.
        save_dir: Location to save any information about this model, including training data.
            It is created if it doesn't exist.
        device: Device to load model to.

        categorical_likelihood_coefficient: Coefficient for balancing likelihoods.
        kl_coefficient: Coefficient for the KL term.

        embedding_dim: Dimension of embedding for each feature.
        transformer_embedding_dim: Embedding dimension to be used in the set transformer blocks.
        num_heads: Number of heads in each multi-head attention block.
        num_blocks: Number of SABs in the model, both the encoder and decoder.
        use_isab: Should ISAB blocks be used instead of SAB blocks.
        num_inducing_points: Number of inducing points.
        multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            Valid options are "xavier" and "kaiming".
        use_layer_norm: Whether layer normalisation should be used in MAB blocks.
        elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
            Valid options are "single" and "double".
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        categorical_likelihood_coefficient: float,
        kl_coefficient: float,
        embedding_dim: int,
        transformer_embedding_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 4,
        use_isab: bool = False,
        num_inducing_points: Optional[int] = 16,
        multihead_init_type: str = "kaiming",
        use_layer_norm: bool = True,
        elementwise_transform_type: str = "single",
    ) -> None:
        super().__init__(model_id, variables, save_dir, device)
        self._categorical_likelihood_coefficient = categorical_likelihood_coefficient
        self._kl_coefficient = kl_coefficient

        self.input_dim = variables.num_processed_cols
        self._feature_embedder = FeatureEmbedder(
            input_dim=self.input_dim,
            embedding_dim=embedding_dim,
            metadata=None,
            device=device,
            multiply_weights=False,
        )
        self._feature_embedding_dim = self._feature_embedder.output_dim
        self._transformer_embedding_dim = transformer_embedding_dim
        self._input_dimension_transform = torch.nn.Linear(self._feature_embedding_dim, self._transformer_embedding_dim)

        self.transformer_vae = TransformerVAE(
            model_id=model_id,
            variables=variables,
            save_dir=save_dir,
            device=device,
            categorical_likelihood_coefficient=self._categorical_likelihood_coefficient,
            kl_coefficient=self._kl_coefficient,
            transformer_embedding_dim=transformer_embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )

        self._featurewise_activation = FeaturewiseActivation(activation_for_continuous=Sigmoid, variables=variables)

        self.to(device)

    @classmethod
    def name(cls) -> str:
        return "transformer_pvae"

    def _train(self, dataset, train_output_dir, report_progress_callback, **train_config_dict):
        train_model(
            self,
            dataset=dataset,
            train_output_dir=train_output_dir,
            report_progress_callback=report_progress_callback,
            **train_config_dict,
        )

    def encode(self, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run encoding part of the PVAE.
        Args:
            data: Input tensor with shape (batch_size, input_dim).
            mask: Mask indicting observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.
        Returns:
            mean, logvar: Latent space samples of shape (batch_size, latent_dim).
        """
        _, _, (encoder_mean, encoder_logvar) = self.reconstruct(data=data, mask=mask, sample=False, count=1)
        return encoder_mean, encoder_logvar

    def loss(self, loss_config: LossConfig, input_tensors: Tuple[torch.Tensor, ...]) -> VAELossResults:
        """
        Calculate tensor loss on a single batch of data.
        Args:
            loss_config (LossConfig): Parameters that specify how the loss is calculated.
            input_tensors: Single batch of data and mask, given as (data, mask). The mask indicates
                which data is present. 1 is observed, 0 is unobserved.

        """
        data, mask = input_tensors
        assert loss_config.max_p_train_dropout is not None
        assert loss_config.score_imputation is not None
        assert loss_config.score_reconstruction is not None
        assert loss_config.score_reconstruction or loss_config.score_imputation

        input_mask, scoring_mask = get_input_and_scoring_processed_masks(
            self.data_processor,
            mask,
            max_p_train_dropout=loss_config.max_p_train_dropout,
            score_imputation=loss_config.score_imputation,
            score_reconstruction=loss_config.score_reconstruction,
        )

        # TODO: Deduplicate with PartialVAE and TorchVAE loss functions.
        (decoder_mean, decoder_logvar), _, encoder_outputs = self.reconstruct(data=data, mask=input_mask)
        kl = kl_divergence(encoder_outputs).sum()
        nll = negative_log_likelihood(
            data, decoder_mean, decoder_logvar, self.variables, self._categorical_likelihood_coefficient, scoring_mask
        )
        loss = self._kl_coefficient * kl + nll
        return VAELossResults(loss=loss, kl=kl, nll=nll, mask_sum=scoring_mask.sum())

    def reconstruct(
        self, data: torch.Tensor, mask: Optional[torch.Tensor], sample: bool = True, count: int = 1, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reconstruct data by passing them through the VAE.
        Args:
            data: Input data with shape (batch_size, input_dim).
            mask: If not None, mask indicating observed variables with shape (batch_size, input_dim). 1 is observed,
                  0 is un-observed. If None, everything is observed.
            sample: If True, samples the latent variables, otherwise uses the mean.
            count: Number of samples to reconstruct.
        Returns:
            (decoder_mean, decoder_logvar): Reconstructed variables, output from the decoder. Both are shape (count, batch_size, output_dim). Count dim is removed if 1.
            samples: Latent variable used to create reconstruction (input to the decoder). Shape (count, batch_size, latent_dim). Count dim is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)
        """
        # Check shape of data and mask
        assert data.shape[1] == self.input_dim
        if mask is None:
            mask = torch.ones_like(data)
        assert data.shape == mask.shape

        # Embed features
        batch_size, input_dim = data.shape
        data = data * mask
        data = self._feature_embedder(data)  # shape (batch_size * input_dim, feature_embedding_dim)
        data = data.reshape(
            batch_size, input_dim, self._feature_embedding_dim
        )  # shape (batch_size, input_dim, feature_embedding_dim)
        # This channel held the embedding bias, which we don't need, so we can use it to hold the mask.
        data[:, :, -1] = mask

        # Reshape data
        data = self._input_dimension_transform(data)  # shape (batch_size, input_dim, transformer_embedding_dim)
        data = data.reshape(
            batch_size, input_dim * self._transformer_embedding_dim
        )  # shape (batch_size, input_dim * transformer_embedding_dim)

        # Transformer VAE
        (decoder_mean, decoder_logvar), samples, (encoder_mean, encoder_logvar) = self.transformer_vae.reconstruct(
            data, sample=sample, count=count
        )

        # Feature-wise activation
        decoder_mean = self._featurewise_activation(decoder_mean)

        return (decoder_mean, decoder_logvar), samples, (encoder_mean, encoder_logvar)


class TransformerVAE(TorchVAE):
    """
    Transformer VAE

    A VAE that has a transformer as an encoder and a decoder.

    See the Set Transformer model https://arxiv.org/abs/1810.00825 for more details
    about the implementation of SAB and ISAB blocks.

    Args:
        model_id: Unique model ID for referencing this model instance.
        variables: Information about variables/features used by this model.
        save_dir: Location to save any information about this model, including training data.
            It is created if it doesn't exist.
        device: Device to load model to.

        categorical_likelihood_coefficient: Coefficient for balancing likelihoods.
        kl_coefficient: Coefficient for the KL term.

        transformer_embedding_dim: Embedding dimension to be used in the set transformer blocks.
        num_heads: Number of heads in each multi-head attention block.
        num_blocks: Number of SABs in the model, both the encoder and decoder.
        use_isab: Should ISAB blocks be used instead of SAB blocks.
        num_inducing_points: Number of inducing points.
        multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            Valid options are "xavier" and "kaiming".
        use_layer_norm: Whether layer normalisation should be used in MAB blocks.
        elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
            Valid options are "single" and "double".
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        categorical_likelihood_coefficient: float,
        kl_coefficient: float,
        transformer_embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        use_isab: bool,
        num_inducing_points: Optional[int],
        multihead_init_type: str,
        use_layer_norm: bool,
        elementwise_transform_type: str,
    ):
        encoder = TransformerEncoder(
            transformer_embedding_dim=transformer_embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )
        decoder = TransformerDecoder(
            transformer_embedding_dim=transformer_embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )
        super().__init__(
            model_id=model_id,
            variables=variables,
            save_dir=save_dir,
            device=device,
            encoder=encoder,
            decoder=decoder,
            categorical_likelihood_coefficient=categorical_likelihood_coefficient,
            kl_coefficient=kl_coefficient,
        )

    @classmethod
    def name(cls) -> str:
        return "transformer_vae"

    def encode(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run encoding part of the VAE.
        Args:
            input_tensors: Input tensors. Each with shape (batch_size, input_dim).
        Returns:
            mean, logvar: Latent space samples of shape (batch_size, latent_dim).
        """
        data = input_tensors[0]
        return self._encoder(data)

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


class TransformerEncoder(torch.nn.Module):
    """
    Transformer encoder

    The TransformerEncoder module passes features through a Transformer module
    and reads out latent distribution means and variances via a single linear layer.

    See the Set Transformer model https://arxiv.org/abs/1810.00825 for more details
    about the implementation of SAB and ISAB blocks.

    Args:
        transformer_embedding_dim: Embedding dimension to be used in the set transformer blocks.
        num_heads: Number of heads in each multi-head attention block.
        num_blocks: Number of SABs in the model.
        use_isab: Should ISAB blocks be used instead of SAB blocks.
        num_inducing_points: Number of inducing points.
        multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            Valid options are "xavier" and "kaiming".
        use_layer_norm: Whether layer normalisation should be used in MAB blocks.
        elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
            Valid options are "single" and "double".
    """

    def __init__(
        self,
        transformer_embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        use_isab: bool,
        num_inducing_points: Optional[int],
        multihead_init_type: str,
        use_layer_norm: bool,
        elementwise_transform_type: str,
    ):
        super().__init__()
        self._transformer_embedding_dim = transformer_embedding_dim
        self._transformer = Transformer(
            transformer_embedding_dim=self._transformer_embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )
        self._mean_logvar_readout = torch.nn.Linear(
            in_features=self._transformer_embedding_dim,
            out_features=2 * self._transformer_embedding_dim,
        )

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass data through the transformer encoder.
        Args:
            data: Data tensor with shape (batch_size, input_dim * transformer_embedding_dim).
        Returns:
            mean, logvar: Encoder output tensors each with shape (batch_size, input_dim * transformer_embedding_dim).
        """
        batch_size, _ = data.shape
        data = data.reshape(batch_size, -1, self._transformer_embedding_dim)

        _, input_dim, transformer_embedding_dim = data.shape
        assert transformer_embedding_dim == self._transformer_embedding_dim

        # Transformer
        data = self._transformer(data)  # shape (batch_size, input_dim, transformer_embedding_dim)

        # Mean and log-variance readout
        data = self._mean_logvar_readout(data)  # shape  (batch_size, input_dim, 2 * transformer_embedding_dim)
        mean, logvar = data.chunk(chunks=2, dim=-1)
        mean = mean.reshape(batch_size, input_dim * self._transformer_embedding_dim)
        logvar = logvar.reshape(batch_size, input_dim * self._transformer_embedding_dim)
        return mean, logvar

    def save_onnx(self, save_dir):
        raise ONNXNotImplemented


class TransformerDecoder(torch.nn.Module):
    """
    Transformer decoder

    The TransformerDecoder module that passes features through a Transformer module
    and reads out reconstructed means and variances through a single linear layer.

    See the Set Transformer model https://arxiv.org/abs/1810.00825 for more details
    about the implementation of SAB and ISAB blocks.

    Args:
        transformer_embedding_dim: Embedding dimension to be used in the set transformer blocks.
        num_heads: Number of heads in each multi-head attention block.
        num_blocks: Number of SABs in the model.
        use_isab: Should ISAB blocks be used instead of SAB blocks.
        num_inducing_points: Number of inducing points.
        multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            Valid options are "xavier" and "kaiming".
        use_layer_norm: Whether layer normalisation should be used in MAB blocks.
        elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
            Valid options are "single" and "double".
    """

    def __init__(
        self,
        transformer_embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        use_isab: bool,
        num_inducing_points: Optional[int],
        multihead_init_type: str,
        use_layer_norm: bool,
        elementwise_transform_type: str,
    ):
        super().__init__()
        self._transformer_embedding_dim = transformer_embedding_dim
        self._transformer = Transformer(
            transformer_embedding_dim=self._transformer_embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )
        self._output_readout = torch.nn.Linear(in_features=self._transformer_embedding_dim, out_features=2)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass data through the transformer decoder.
        Args:
            data: Data tensor with shape (batch_size, input_dim * transformer_embedding_dim).
        Returns:
            mean, logvar: Decoder output tensors each with shape (batch_size, input_dim).
        """
        batch_size, _ = data.shape

        # Reshape input
        data = data.reshape((batch_size, -1, self._transformer_embedding_dim))

        # Transformer
        data = self._transformer(data)  # shape (batch_size, input_dim, transformer_embedding_dim)

        # Output readout & feature-wise activation
        data = self._output_readout(data)  # shape (batch_size, input_dim, 2)
        mean, logvar = data.chunk(chunks=2, dim=-1)  # shape (batch_size, input_dim, 1) each
        mean, logvar = (
            mean.reshape(batch_size, -1),
            logvar.reshape(batch_size, -1),
        )  # shape (batch_size, input_dim) each
        return mean, logvar

    def save_onnx(self, save_dir):
        raise ONNXNotImplemented


class Transformer(torch.nn.Module):
    """
    A transformer module composed of several blocks of sabs or isabs.

    See the Set Transformer model https://arxiv.org/abs/1810.00825 for more details
    about the implementation of SAB and ISAB blocks.

    Args:
        transformer_embedding_dim: Embedding dimension to be used in the set transformer blocks.
        num_heads: Number of heads in each multi-head attention block.
        num_blocks: Number of SABs in the model.
        use_isab: Should ISAB blocks be used instead of SAB blocks.
        num_inducing_points: Number of inducing points.
        multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            Valid options are "xavier" and "kaiming".
        use_layer_norm: Whether layer normalisation should be used in MAB blocks.
        elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
            Valid options are "single" and "double".
    """

    def __init__(
        self,
        transformer_embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        use_isab: bool,
        num_inducing_points: Optional[int],
        multihead_init_type: str,
        use_layer_norm: bool,
        elementwise_transform_type: str,
    ):
        super().__init__()
        if use_isab:
            assert num_inducing_points is not None
            self._sabs = torch.nn.ModuleList(
                [
                    ISAB(
                        embedding_dim=transformer_embedding_dim,
                        num_heads=num_heads,
                        num_inducing_points=num_inducing_points,
                        multihead_init_type=SetTransformer.MultiheadInitType[multihead_init_type],
                        use_layer_norm=use_layer_norm,
                        elementwise_transform_type=SetTransformer.ElementwiseTransformType[elementwise_transform_type],
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self._sabs = torch.nn.ModuleList(
                [
                    SAB(
                        embedding_dim=transformer_embedding_dim,
                        num_heads=num_heads,
                        multihead_init_type=SetTransformer.MultiheadInitType[multihead_init_type],
                        use_layer_norm=use_layer_norm,
                        elementwise_transform_type=SetTransformer.ElementwiseTransformType[elementwise_transform_type],
                    )
                    for _ in range(num_blocks)
                ]
            )

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass data through self-attention blocks.
        Args:
            data: Data tensor with shape (batch_size, input_dim, embedding_dim) to be used as query, key and value.
        Returns:
            output: Attention output tensor with shape (batch_size, input_dim, embedding_dim).
        """
        for sab in self._sabs:
            data = sab(data, mask=None)  # shape (batch_size, input_dim, embedding_dim)
        return data
