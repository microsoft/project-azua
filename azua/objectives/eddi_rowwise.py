from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..utils.data_mask_utils import add_to_data, add_to_mask
from ..utils.training_objectives import kl_divergence
from .eddi import EDDIObjective


class EDDIRowwiseObjective(EDDIObjective):
    """
    EDDI objective batched across rows.
    #TODO: Combine with EDDIObjective to batch across all features for a batch of rows simultaneously.
    """

    @staticmethod
    def name():
        return "eddi_rowwise"

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
            rewards (List of Dictionaries or Numpy array): Length (batch_size) of dictionaries of the form {idx : info_gain} where
                info_gain is np.nan if the variable is observed already (1 in obs_mask).
                If as_array is True, rewards is an array of shape (batch_size, feature_count) instead.
        """
        assert obs_mask.shape == data.shape
        self._model.set_evaluation_mode()
        num_rows, feature_count = data.shape
        num_groups = len(self._model.variables.group_names)

        # Here we perform the EDDI calculation for each data row separately, and batch the computation over all
        # query groups that can be queried (i.e. they contain at least one feature that is unobserved so far during
        # active learning with an underlying value in the data). We perform this batching by adding a new row for each
        # queriable group, with the imputed values corresponding to this group revealed.

        # This batching method can be more efficient when the number of features is much greater than the number of rows,
        # for example in the Azure service if we have a single data row with many features to select from.

        # Mask that is 1 if data_mask == 1 (there is an underlying data value) and obs_mask == 1 (we have observed
        # this feature during active learning), so we can treat this feature as observed.
        mask = data_mask * obs_mask

        # Mask that is 1 if data_mask == 1 (there is an underlying data value) and obs_mask == 0 (we haven't observed
        # this feature yet), so we can query this feature.
        query_mask = data_mask * (1 - obs_mask)

        with torch.no_grad():  # Turn off gradient tracking for performance and to prevent numpy issues
            # EDDI rewards for features in each row
            rewards_list = []
            for row_idx in range(num_rows):
                data_x_o = data[row_idx : row_idx + 1, :]
                mask_x_o = mask[row_idx : row_idx + 1, :]
                # Keep additional mask indicating which features we can query, in order to determine which imputed
                # values to reveal.
                query_mask_x_o = query_mask[row_idx : row_idx + 1, :]
                # Shape (sample_count, 1, feature_count)
                imputations = self._model.impute_processed_batch(
                    data_x_o,
                    mask_x_o,
                    sample_count=self._sample_count,
                    vamp_prior_data=vamp_prior_data,
                    preserve_data=True,
                )
                # Shape (sample_count, feature_count)
                imputations = imputations.reshape(self._sample_count, feature_count)
                # Repeat the row by number of MC samples.
                # Shape (sample_count, feature_count)
                data_x_o = torch.repeat_interleave(data_x_o, self._sample_count, dim=0)
                mask_x_o = torch.repeat_interleave(mask_x_o, self._sample_count, dim=0)

                # Reveal imputed values for each observable query group on a separate row
                unobs_reward_idxs = []
                unobs_group_idxs = []
                # TODO maybe rename to group_idx and group_var_idxs? and group_idxs -> group_var_idxs
                for reward_idx, group_idxs in enumerate(self._model.variables.group_idxs):

                    def flatten(lists):
                        # Flatten proc_cols for continuous and binary unproc_cols, since they will be of form [[1], [2], ...]
                        return [i for sublist in lists for i in sublist]

                    # Get processed data/mask columns corresponding to all variables in the group
                    group_cols = flatten([self._model.variables.processed_cols[idx] for idx in group_idxs])
                    if max(query_mask_x_o[0, group_cols]) != 0:
                        unobs_reward_idxs.append(reward_idx)
                        unobs_group_idxs.append(group_idxs)

                num_unobs_groups = len(unobs_reward_idxs)

                # Add imputed value for each feature to a copy of the data tensor, and reveal in a corresponding copy of
                # the mask.
                data_x_i_o_list = []
                mask_x_i_o_list = []
                for group_idxs in unobs_group_idxs:
                    data_x_i_o_list.append(add_to_data(self._model.variables, data_x_o, imputations, group_idxs))
                    mask_x_i_o_list.append(add_to_mask(self._model.variables, mask_x_o, group_idxs))

                # Compute q(z|x_o)
                # Shape (sample_count, latent_dim)
                q_o = self._model.encode(data_x_o, mask_x_o)

                # Repeat by number of query variables, each contiguous block of rows corresponds to a single MC sample.
                # Shape (sample_count * num_groups, latent_dim)
                q_o = (
                    torch.repeat_interleave(q_o[0], num_unobs_groups, dim=0),
                    torch.repeat_interleave(q_o[1], num_unobs_groups, dim=0),
                )

                # Stack and reorder so that the samples are on the same dimension as the variables, and the samples for
                # each variable are grouped together
                # Shape (sample_count * num_groups, feature_count)
                if len(data_x_i_o_list) > 0:
                    data_x_i_o = torch.stack(data_x_i_o_list, dim=0).reshape(
                        self._sample_count * num_unobs_groups, feature_count
                    )
                    mask_x_i_o = torch.stack(mask_x_i_o_list, dim=0).reshape(
                        self._sample_count * num_unobs_groups, feature_count
                    )

                    # Shape (sample_count * num_groups, latent_dim)
                    q_i_o = self._model.encode(data_x_i_o, mask_x_i_o)

                    # Shape (sample_count * num_groups)
                    kl1 = kl_divergence(q_i_o, q_o)

                    # TODO double check this
                    phi_idxs = self._model.variables.target_var_idxs
                    if len(phi_idxs) > 0:
                        # Additionally reveal imputed values for target variables in order to calculate p(z|x_o, x_phi) and
                        # p(z|x_o, x_i, x_phi)
                        # Shape (sample_count * num_unobs_groups, feature_count)
                        data_x_o_phi = add_to_data(self._model.variables, data_x_o, imputations, phi_idxs)
                        mask_x_o_phi = add_to_mask(self._model.variables, mask_x_o, phi_idxs)

                        # Shape (sample_count * num_unobs_groups, feature_count)
                        data_x_i_o_phi = add_to_data(
                            self._model.variables, data_x_i_o, imputations.repeat(num_unobs_groups, 1), phi_idxs
                        )
                        mask_i_o_phi = add_to_mask(self._model.variables, mask_x_i_o, phi_idxs)

                        q_o_phi = self._model.encode(data_x_o_phi, mask_x_o_phi)
                        # Shape (sample_count * num_unobs_groups, latent_dim), samples grouped togethe
                        q_o_phi = (
                            q_o_phi[0].repeat(num_unobs_groups, 1),
                            q_o_phi[1].repeat(num_unobs_groups, 1),
                        )

                        # Shape (sample_count * num_unobs_groups, latent_dim)
                        q_i_o_phi = self._model.encode(data_x_i_o_phi, mask_i_o_phi)

                        # Shape (sample_count * num_unobs_groups), samples grouped together
                        kl2 = kl_divergence(q_i_o_phi, q_o_phi)

                    else:
                        # Shape (sample_count * num_unobs_groups), samples grouped together
                        kl2 = torch.zeros_like(kl1)

                    # Shape (sample_count * num_unobs_groups), samples grouped together
                    row_unobs_rewards = (kl1 - kl2).cpu().numpy()

                    # Shape (sample_count, num_unobs_groups), with the samples for each variable on a separate row
                    row_unobs_rewards = np.reshape(row_unobs_rewards, (num_unobs_groups, self._sample_count))
                    # Shape (num_unobs_groups, )
                    row_unobs_rewards = np.mean(row_unobs_rewards, axis=1)

                row_rewards = np.full(num_groups, np.nan)

                for i, j in enumerate(unobs_reward_idxs):
                    row_rewards[j] = row_unobs_rewards[i]

                # Add diffs for observed query groups to make shape correct
                rewards_list.append(row_rewards)

            # Shape (num_rows, num_groups)
            rewards = np.vstack(rewards_list)

            if not as_array:
                return [{idx: float(val) for idx, val in enumerate(row)} for row in rewards]
        return rewards
