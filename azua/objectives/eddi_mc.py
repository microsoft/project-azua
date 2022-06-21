from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..models.imodel import IModelForObjective
from ..utils.data_mask_utils import add_to_data, add_to_mask, restore_preserved_values
from ..utils.training_objectives import negative_log_likelihood
from .eddi import EDDIObjective


class EDDIMCObjective(EDDIObjective):
    """ "
    EDDIMC objective.
    """

    def __init__(self, model: IModelForObjective, sample_count, use_vamp_prior, **kwargs):
        """
        Args:
            model (Model): Trained `Model` class to use.
            sample_count (int): Number of imputation samples to use.
            use_vamp_prior (bool): Whether or not to use vamp prior method.
        """
        super().__init__(model, sample_count, use_vamp_prior, **kwargs)

    @classmethod
    def name(cls):
        return "eddi_mc"

    def _log_ll_mc_estimate(
        self,
        imputed: torch.Tensor,
        dec_mean: torch.Tensor,
        dec_logvar: torch.Tensor,
        summed: bool = False,
    ):
        """
        A helper function for computing log p(x_i|x_o, ...) using Monte Carlo estimate.
        This is computed by averaging log p(x_i|x_o, ...) over samples of x_i, and for each log p(x_i|x_o, ...),
        it is approximated by log-mean-exp(log p(x_i|z_k)) where z_k ~ q(z|x_o, ...).

        Args:
             imputed: imputed results/samples from p(x_i|x_o, ...), with shape (sample_count, feature_count)
             dec_mean: mean of p(x_i|z_k) for each samples of x_i and z, has the same shape as imputed
             dec_logvar: logvar of p(x_i|z_k), has the same shape as imputed
             summed: indicator for summing over feature indices, default is False

        Returns:
             ll: a Monte Carlo approximation to log p(x_i|x_o, ...), with shape (sample_count, feature_count) if
                      summed=False or shape (sample_count,) if summed=True
        """
        sample_count, feature_count = imputed.shape
        # It needs evaluating each of the x_i samples with all the other (dec_mean_j, dec_logvar_j) pairs
        # and to debias this, need to make sure i != j
        # first repeat the imputed and dec dist. parameters to shape (sample_count*sample_count, feature_count)
        # but imputed_ is repeated, and dec_mean_, dec_logvar_ are "tiled", see numpy doc explanations
        imputed_ = imputed.unsqueeze(0).repeat(sample_count, 1, 1).reshape(-1, feature_count)
        dec_mean_ = dec_mean.unsqueeze(1).repeat(1, sample_count, 1).reshape(-1, feature_count)
        dec_logvar_ = dec_logvar.unsqueeze(1).repeat(1, sample_count, 1).reshape(-1, feature_count)

        # then compute the log p(x_i|z) where z is sampled from q(z|x_o)
        ll = -negative_log_likelihood(
            imputed_,
            dec_mean_,
            dec_logvar_,
            self._model.variables,
            alpha=1.0,
            sum_type=None,
        )
        ll = ll.reshape(sample_count, sample_count, feature_count)
        # now exclude the diagonals (i.e. set them to 0), later the diagonals will be exponentiated
        ll = ll * (1 - torch.eye(sample_count).unsqueeze(2))
        if summed:  # summing over features
            ll = ll.sum(dim=-1)
        # log-mean-exp to approximate log p(x_i|x_o) for every sample of x_i
        # need to subtract 1.0 to counter for the diagonal entries
        ll = torch.log(torch.clamp((torch.sum(ll.exp(), dim=0) - 1) / (sample_count - 1), 1e-10, 1e10))
        return ll

    def _entropy_mc_approx(
        self,
        imputed: torch.Tensor,
        dec_mean: torch.Tensor,
        dec_logvar: torch.Tensor,
        summed: bool = False,
    ):
        """
        A helper function for computing entropy of log p(x_i | x_o, ...) using Monte Carlo estimate.
        This is computed by averaging log p(x_i | x_o, ...) over samples of x_i, and for each log p(x_i| x_o, ...),
        it is approximated by log-mean-exp(log p(x_i|z_k)) where z_k ~ q(z | x_o, ...).

        Args:
             imputed: imputed results/samples from p(x_i|x_o, ...), with shape (sample_count, batch_size, feature_count)
             dec_mean: mean of p(x_i|z_k) for each samples of x_i and z, has the same shape as imputed
             dec_logvar: logvar of p(x_i|z_k), has the same shape as imputed
             summed: indicator for summing over feature indices, default is False

        Returns:
             entropy: a Monte Carlo approximation to the entropy, with shape (batch_size, feature_count) if
                      summed=False or shape (batch_size,) if summed=True
        """
        entropy = []
        for b in range(imputed.shape[1]):
            ll = self._log_ll_mc_estimate(imputed[:, b], dec_mean[:, b], dec_logvar[:, b], summed)
            entropy.append(-ll.mean(dim=0))  # shape (feature_count,) or (,)
        return torch.stack(entropy, dim=0)  # shape (batch_size, feature_count) or (batch_size,)

    def _information_gain(
        self,
        data: torch.Tensor,
        data_mask: torch.Tensor,
        obs_mask: torch.Tensor,
        as_array: bool = False,
        vamp_prior_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[List[Dict[int, float]], np.ndarray]:
        """
        Calculate information gain of adding each observable feature individually for a batch of data rows.
        The estimate is computed by the difference between two entropy terms, where each of the entropy term
        is approximated with Monte Carlo.

        Args:
            data (shape (batch_size, proc_feature_count)): Contains processed, observed data.
            data_mask (shape (batch_size, proc_feature_count)): Contains mask where 1 is observed in the
                underlying data, 0 is missing.
            obs_mask (shape (batch_size, proc_feature_count)): Processed mask indicating which
                features have already been observed (i.e. which to condition the information gain calculations on).
            as_array (bool): When True will return info_gain values as an np.ndarray. When False (default) will return info
                gain values as a List of Dictionaries.
        Returns:
            rewards (List of Dictionaries or Numpy array): Length (batch_size) of dictionaries of the form {idx : info_gain} where
                info_gain is np.nan if the variable is observed already (1 in obs_mask).
                If as_array is True, rewards is an array of shape (batch_size, feature_count) instead.

        """
        assert obs_mask.shape == data.shape
        # TODO why do we require self._sample_count > 1?
        assert self._sample_count > 1
        phi_idxs = self._model.variables.target_var_idxs
        # if no target variable is specified, revert back to the original eddi objective (KL in z space)
        if len(phi_idxs) == 0:
            return super()._information_gain(data, data_mask, obs_mask, as_array, vamp_prior_data)

        self._model.set_evaluation_mode()
        batch_size, feature_count = data.shape
        is_variable_to_observe = self._model.variables.get_variables_to_observe(data_mask)

        mask = data_mask * obs_mask

        # Repeat each row sample_count times to allow batch computation over all samples
        repeated_data = data.unsqueeze(0).repeat(self._sample_count, 1, 1).reshape(-1, feature_count)
        repeated_mask = mask.unsqueeze(0).repeat(self._sample_count, 1, 1).reshape(-1, feature_count)

        with torch.no_grad():  # Turn off gradient tracking for performance and to prevent numpy issues
            # impute with current observations, returning tensors of shape (sample_count, batch_size, dim)
            (dec_mean, dec_logvar), _, _ = self._model.reconstruct(
                data=data, mask=mask, sample=True, count=self._sample_count
            )
            imputed = restore_preserved_values(self._model.variables, data, dec_mean, mask)

            # compute the entropy of p(x_i|x_o) with MC approximation:
            entropy = self._entropy_mc_approx(imputed, dec_mean, dec_logvar, summed=False)

            # conditional entropy computation if there exist target variables
            # adding x_phi to data
            mask_o_phi = add_to_mask(self._model.variables, repeated_mask, phi_idxs)
            x_o_phi = add_to_data(
                self._model.variables,
                repeated_data,
                imputed.reshape(-1, feature_count),
                phi_idxs,
            )
            (dec_mean_phi, dec_logvar_phi), _, _ = self._model.reconstruct(
                data=x_o_phi, mask=mask_o_phi, sample=True, count=self._sample_count
            )
            imputed_phi = restore_preserved_values(self._model.variables, x_o_phi, dec_mean_phi, mask_o_phi)
            # compute entropy for repeated data, returns tensor of shape (sample_count*batch_size, feature_count)
            conditional_entropy = self._entropy_mc_approx(imputed_phi, dec_mean_phi, dec_logvar_phi, summed=False)
            conditional_entropy = conditional_entropy.reshape(self._sample_count, batch_size, feature_count).mean(dim=0)
            rewards = entropy - conditional_entropy  # acquisition based on increasing mutual info
            rewards = rewards.cpu().numpy()  # shape (batch_size, feature_count)
            rewards = rewards[:, is_variable_to_observe.cpu().numpy()]

            # Remove estimates for already observed features - to be optimised
            feature_mask = np.delete(
                self._model.data_processor.revert_mask(mask.cpu().numpy()),
                self._model.variables.target_var_idxs,
                axis=1,
            )
            rewards[feature_mask.astype(bool)] = np.nan

            if not as_array:
                rewards = [{idx: float(val) for idx, val in enumerate(row)} for row in rewards]
        return rewards
