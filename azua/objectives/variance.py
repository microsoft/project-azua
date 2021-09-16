from typing import List, Optional, Tuple, Dict, Union, cast

import numpy as np
import torch

from ..models.transformer_imputer import TransformerImputer
from ..objectives.eddi import EDDIObjective
from ..utils.data_mask_utils import add_to_data, add_to_mask


class VarianceObjective(EDDIObjective):
    """
    Variance objective: select the feature x_i that minimises expected variance in the target variable y after observing x_i.
    
    The variance of P(y|x_i, x_o) is, by definition, the expected squared error in y if you use E(y|x_i, x_o) as a 
    prediction of y. Thus, this objective directly targets small mean square error in y.

    Limitations:
        
    1) This objective uses some strong, and generally wrong assumptions: 
    - the conditional distribution of each feature given the others is Gaussian.
    - the model gives well-calibrated variance estimates.    
    
    2) This objective is not currently compatible with PVAE. To use it with PVAE, we would need to implement a method 
    that returns *marginal* variance of the target variable (not the same as decoder_variance returned by PVAE.reconstruct). For each sampled value of x_i,
    you would need to take multiple samples of z from q(z|x_i, x_o) and corresponding values of p(y|z). Then, if PVAE uses a fixed decoder variance,
    the marginal variance of y given x_i is something like the between-samples variance of E(y|z) plus the decoder variance.

    3) This implementation is useless for non-continuous target variables, for which the variance output from TransformerImputer is ignored.
    The analogous treatment for categorical targets would be to select the feature that maximises expected correct classification rate,
    i.e. pick feature i that maximises the expectation over x_i of max_n P(y=n| x_i, x_o), where y is the target variable and x_o is data already observed.
    """

    def __init__(self, model: TransformerImputer, sample_count: int, **kwargs):  # type: ignore[override]

        if not model.variables.target_var_idxs:
            raise NotImplementedError
        kwargs.update({"use_vamp_prior": False})
        super().__init__(model=model, sample_count=sample_count, **kwargs)  # type: ignore

    @classmethod
    def name(cls):
        return "variance"

    def _information_gain(
        self,
        data: torch.Tensor,
        data_mask: torch.Tensor,
        obs_mask: torch.Tensor,
        as_array: bool = False,
        vamp_prior_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[List[Dict[int, float]], np.ndarray]:
        """
        For each observable group of features and each row in a batch of data, estimate the reduction in variance of the target
        variable that would be obtained by observing the feature group.

        TODO deduplicate code with EDDIObjective._information_gain

        Args:
            data (shape (batch_size, proc_feature_count)): processed, observed data.
            data_mask (shape (batch_size, proc_feature_count)): processed mask where 1 is observed in the 
                underlying data, 0 is missing.
            obs_mask (shape (batch_size, proc_feature_count)): indicates which
                features have already been observed (i.e. which to condition the information gain calculations on).
            as_array (bool): When True will return info_gain values as an np.ndarray. When False (default) will return info
                gain values as a List of Dictionaries.
            vamp_prior_data: used to generate latent space samples when input data is completely unobserved
        Returns:
            rewards (List of Dictionaries or Numpy array): Length (batch_size) of dictionaries of the form {group_name : info_gain} where
                info_gain is np.nan if the group of variables is observed already (1 in obs_mask).
                If as_array is True, rewards is an array of shape (batch_size, group_count) instead.
        """
        assert obs_mask.shape == data.shape
        self._model.set_evaluation_mode()
        batch_size, feature_count = data.shape
        is_variable_to_observe = self._model.variables.get_variables_to_observe(data_mask)
        mask = data_mask * obs_mask

        # Repeat each row sample_count times to allow batch computation over all samples
        repeated_data = torch.repeat_interleave(data, self._sample_count, dim=0)
        repeated_mask = torch.repeat_interleave(mask, self._sample_count, dim=0)

        with torch.no_grad():  # Turn off gradient tracking for performance and to prevent numpy issues

            # Shape (sample_count, batch_size, feature_count)
            imputed = self._model.impute_processed_batch(
                data, mask, sample_count=self._sample_count, vamp_prior_data=None, preserve_data=True, sample=True
            )

            imputed = imputed.permute(1, 0, 2)  # Shape (batch_size, sample_count, feature_count)

            imputed = imputed.reshape(self._sample_count * batch_size, feature_count)

            # Var(y|x_o): variance in the target given already observed data
            current_var = self._calc_sum_target_variances(data, mask)

            rewards_list = []
            for group_idxs in self._model.variables.query_group_idxs:
                if all(not is_variable_to_observe[idx] for idx in group_idxs):
                    diff = torch.full((batch_size,), np.nan)
                else:
                    # Include x_i in the observed data
                    mask_i_o = add_to_mask(
                        self._model.variables, repeated_mask, group_idxs
                    )  # shape (sample count * batch_size, feature_count)

                    x_i_o = add_to_data(self._model.variables, repeated_data, imputed, group_idxs)

                    # Var(y|x_o, x_i): variance in the target after observing x_i.
                    var_after_observing_i = self._calc_sum_target_variances(
                        data=x_i_o, mask=mask_i_o
                    )  # shape (batch_size * sample_count)

                    # Average Var(y|x_o, x_i) over samples of x_i
                    var_after_observing_i = var_after_observing_i.reshape(batch_size, self._sample_count).mean(dim=1)

                    # Expected improvement in variance if we were to observe x_i
                    diff = current_var - var_after_observing_i  # Shape (batch_size,)

                rewards_list.append(diff.cpu().numpy())

            rewards = np.vstack(rewards_list).T  # Shape (batch_size, feature_count)

            rewards = self._remove_rewards_for_observed_groups(mask, rewards)

            if not as_array:
                rewards = [{idx: float(val) for idx, val in enumerate(row)} for row in rewards]
        return rewards

    def _calc_sum_target_variances(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate the sum of variances in target dimensions for given input.
    
        Args:
            data (torch.Tensor): shape (batch_size, input_dim)
            mask (torch.Tensor): shape (batch_size, input_dim)

        Returns:
            var: shape (batch_size,)

        """
        model = cast(TransformerImputer, self._model)
        mean_and_logvar = model(data=data, mask=mask)  # shape (batch_size, input_dim, 2)
        target_logvars = mean_and_logvar[:, self._model.variables.target_var_idxs, 1]  # shape (batch_size, num_targets)
        var = torch.sum(torch.exp(target_logvars), dim=1)  # shape (batch_size,)
        return var
