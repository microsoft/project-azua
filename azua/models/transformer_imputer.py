from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import Sigmoid

from ..models.feature_embedder import FeatureEmbedder
from ..models.imodel import IBatchImputer
from ..models.torch_model import TorchModel
from ..models.torch_training import train_model
from ..models.torch_training_types import LossConfig, LossResults
from ..models.transformer_set_encoder import ISAB, SAB, SetTransformer
from ..utils.training_objectives import get_input_and_scoring_masks
from ..utils.data_mask_utils import restore_preserved_values
from ..utils.training_objectives import negative_log_likelihood
from .decoder import FeaturewiseActivation
from .torch_imputation import impute


class TransformerImputer(TorchModel, IBatchImputer):
    """
    Transformer model for imputation.

    This loosely imitates BERT, see https://arxiv.org/abs/1810.04805v2

    Some differences from BERT:
      - This model is implemented with continuous inputs in mind, not categorical
        ones.  Our readout layer maps the final-layer
        embedding of a feature to a mean and log-variance, whereas BERT's readout layer maps the final-layer
        embedding for a token position to logits.  This model will run with categorical variables, but might
        not handle them in the most sensible way.
      - BERT's 'positions' (of tokens in a sentence) correspond to our 'features' (columns of a table).
        BERT's sinusoidal position encoder is replaced by our FeatureEmbedder.


    """

    def __init__(
        self,
        model_id: str,
        variables,
        save_dir,
        device,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        metadata_filepath: Optional[str],
        num_inducing_points: Optional[int],
        multihead_init_type: str,
        use_layer_norm: bool,
        elementwise_transform_type: str,
        transformer_embedding_dim: int,
        use_isab: bool,
        share_readout_weights: bool,
    ):
        super().__init__(model_id, variables, save_dir, device)

        input_dim = variables.num_processed_cols

        # TODO BERT would add a many-dimensional embedding of x elementwise to the positional encoding, but this puts x in only a single channel.
        self._feature_embedder = FeatureEmbedder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            metadata=None,
            device=device,
            multiply_weights=False,
        )
        feature_embedding_dim = self._feature_embedder.output_dim

        self._input_dimension_transform = torch.nn.Linear(feature_embedding_dim, transformer_embedding_dim)

        if use_isab:
            assert num_inducing_points is not None
            # Inducing-point SAB, for efficient computation with large numbers of features.
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

        self._readout: torch.nn.Module
        if share_readout_weights:
            # Use same readout weights for all features. Treat each feature like a generic mid-sentence token in BERT,
            # so our imputation imitates cloze task/MLM in BERT.
            self._readout = torch.nn.Linear(in_features=transformer_embedding_dim, out_features=2)
        else:
            # Learn separate linear readout weights for each feature. Our imputation imitates BERT classification task.
            self._readout = ParallelLinear(
                num_parallel=input_dim, in_channels=transformer_embedding_dim, out_channels=2
            )

        self._featurewise_activation = FeaturewiseActivation(activation_for_continuous=Sigmoid, variables=variables)
        self.to(self._device)

    def validate_loss_config(self, loss_config: LossConfig) -> None:
        # TODO dedup with PVAE validate_loss_config, probably by putting loss outside model def.
        assert loss_config.score_imputation is not None and loss_config.score_reconstruction is not None
        assert loss_config.score_reconstruction or loss_config.score_imputation
        assert loss_config.max_p_train_dropout is not None

    @classmethod
    def name(cls):
        return "transformer_imputer"

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mean and logvar for each value given masked input.

        Args:
            x (torch.Tensor): processed data. Shape (batch_size, input_dim)
            mask (torch.Tensor): processed mask, where 1 is observed, 0 is unobserved. Shape (batch_size, input_dim)

        Returns:
            torch.Tensor of shape (batch_size, input_dim, 2). The [:, :, 0] is mean and [:, :, 1] is log-variance.
        """
        batch_size, input_dim = data.shape

        # Drop masked data now, so we don't have to bother with masking later.
        data = data * mask

        # Feature embedder is analogous to BERT position encoding.
        data = self._feature_embedder(data)  # shape (batch_size * input_dim, feature_embedder.output_dim)
        data = data.reshape(batch_size, input_dim, -1)  # shape (batch_size, input_dim, feature_embedder.output_dim)

        # This channel held the embedding bias, which we don't need, so we can use it to hold the mask.
        data[:, :, -1] = mask

        # Change number of channels
        data = self._input_dimension_transform(data)  # shape (batch_size, input_dim, transformer_embedding_dim)

        # Run through self-attention blocks
        for sab in self._sabs:
            data = sab(data, mask=None)  # shape (batch_size, input_dim, transformer_embedding_dim)

        readout = self._readout(data)  # shape (batch_size, input_dim, 2)

        # Output channels are mean and log-variance. The feature-wise activation applies to the mean only.
        readout[:, :, 0] = self._featurewise_activation(readout[:, :, 0])

        return readout

    def run_train(self, dataset, train_config_dict, report_progress_callback=None):
        # TODO put this in TorchModel
        train_output_dir = self._create_train_output_dir_and_save_config(train_config_dict)
        processed_dataset = self.data_processor.process_dataset(dataset)
        train_model(
            self,
            dataset=processed_dataset,
            train_output_dir=train_output_dir,
            report_progress_callback=report_progress_callback,
            **train_config_dict,
        )

    def impute(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        impute_config_dict: Optional[Dict[str, int]] = None,
        *,
        vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        average: bool = True,
    ) -> np.ndarray:
        return impute(self, data, mask, impute_config_dict=impute_config_dict, average=average)

    def impute_processed_batch(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        sample_count: int,
        vamp_prior_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        preserve_data: bool = True,
        sample: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """

         Fill in unobserved variables in a minibatch of data using a trained model. Optionally, use a vamp prior to
         impute empty rows, and optionally replace imputed values with input values for observed features.

         Assumes data is a torch.Tensor and in processed form (i.e. variables will be in their squashed ranges,
         and categorical variables will be in one-hot form).

         Args:
             data (shape (batch_size, input_dim)): Data to be used to train the model, in processed form.
             mask (shape (batch_size, input_dim)): Data observation mask, where observed values are 1 and unobserved
                 values are 0.
             sample_count: Number of imputation samples to generate.
             vamp_prior_data: ignored by this model
             preserve_data (bool): Whether or not to impute data already present. Defaults to True, which keeps data
                 present in input.

        Returns:
             imputations (torch.Tensor of shape (sample_count, batch_size, output_dim)): Input data with missing values
                 filled in.
        """
        if vamp_prior_data is not None:
            # Theoretically we could implement this to just impute from vamp prior samples, but that
            # would mean exposing training data at test time, which could pose privacy problems.
            raise NotImplementedError
        if sample:
            mean_and_logvar = self.forward(data=data, mask=mask)
            stddev = torch.exp(mean_and_logvar[:, :, 1] * 0.5)
            gaussian = torch.distributions.Normal(mean_and_logvar[:, :, 0], stddev)
            imputations = gaussian.rsample((sample_count,)).to(
                self._device
            )  # Shape (sample_count, batch_size, input_dim)
        else:
            imputations = self.forward(data=data, mask=mask)[:, :, 0].unsqueeze(0).expand(sample_count, -1, -1)
        if preserve_data:
            imputations = restore_preserved_values(self.variables, data, imputations, mask)

        return imputations

    def loss(self, loss_config: LossConfig, input_tensors) -> LossResults:
        # TODO Reduce code duplication with PVAE.
        x, mask = input_tensors
        # Asserts here are just for mypy; validate_loss_config has already checked loss config is valid.
        # We can't just do loss_config.max_p_train_dropout = cast(...) because loss_config fields
        # are read-only.
        assert loss_config.max_p_train_dropout is not None
        assert loss_config.score_imputation is not None
        assert loss_config.score_reconstruction is not None
        assert loss_config.score_reconstruction or loss_config.score_imputation
        input_mask, scoring_mask = get_input_and_scoring_masks(
            mask,
            max_p_train_dropout=loss_config.max_p_train_dropout,
            score_imputation=loss_config.score_imputation,
            score_reconstruction=loss_config.score_reconstruction,
        )
        prediction = self.forward(x, input_mask)
        mean = prediction[:, :, 0]
        logvar = prediction[:, :, 1]
        assert mean.shape == x.shape

        nll = negative_log_likelihood(
            data=x,
            decoder_mean=mean,
            decoder_logvar=logvar,
            variables=self.variables,
            alpha=0.1,
            mask=scoring_mask,
        )
        return LossResults(loss=nll, mask_sum=torch.sum(scoring_mask))


class ParallelLinear(torch.nn.Module):
    def __init__(self, num_parallel, in_channels, out_channels):
        super().__init__()
        self._conv1d = torch.nn.Conv1d(
            kernel_size=1,
            in_channels=num_parallel * in_channels,
            out_channels=num_parallel * out_channels,
            groups=num_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_parallel, in_channels = x.shape
        x = x.reshape(batch_size, num_parallel * in_channels, 1)
        predictions = self._conv1d(x)  # shape (batch_size, num_parallel * out_channels, 1)
        return predictions.reshape(batch_size, num_parallel, -1)  # Shape (batch_size, num_parallel, out_channels)
