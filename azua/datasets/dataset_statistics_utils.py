import numpy as np


class DatasetStatistics:
    @classmethod
    def get_statistics(cls, data, mask, variables, verbose=False):
        # get statistics on the marginal distributions
        # data: an np.array that has shape (N_data, N_feature)
        # mask: an np.array with {0, 1} entries, has shape (N_data, N_feature)
        stats = {}
        for var_idx, var in enumerate(variables):
            if var.type == "continuous":
                stats[var_idx] = cls._statistics_continuous(data[:, var_idx], var, mask[:, var_idx])
            elif var.type == "binary":
                stats[var_idx] = cls._statistics_binary(data[:, var_idx], var, mask[:, var_idx])
            elif var.type == "categorical":
                stats[var_idx] = cls._statistics_categorical(data[:, var_idx], var, mask[:, var_idx])
            else:
                raise ValueError("data statistics computation not supported for %s variable" % var.type)

        if verbose:  # print statistics

            print("Data statistics")
            for var_idx, var in enumerate(variables):
                output = "id: %d, " % var_idx
                for key in stats[var_idx].keys():
                    val = stats[var_idx][key]
                    if type(val) is str:
                        output += "%s: %s, " % (key, val)
                    elif type(val) is int:
                        output += "%s: %d, " % (key, val)
                    elif type(val) in [
                        float,
                        np.float32,
                        np.float64,
                    ]:  # assume a float number
                        output += "%s: %.3f, " % (key, val)
                    else:
                        output += "%s: %s, " % (key, str(val))
                output = output[:-2]
                print(output)

        return stats

    @staticmethod
    def _statistics_categorical(feature, variable, mask):
        # assume we collect the statistics of a categorical data
        # feature: an np.array has shape (N_data,)
        # variable: an object of variable class
        # mask: an np.array with {0, 1} entries, the missing value indicator

        assert variable.type == "categorical", "expecting a categorical variable, get %s" % variable.type

        # first mask out the missing part
        idx = np.where(mask == 1)[0]
        masked_feature = feature[idx]
        stats = {"type": variable.type}
        stats["missing_percentage"] = (1.0 - np.mean(mask)) * 100

        # convert to one-hot
        from sklearn.preprocessing import label_binarize

        value_range = range(int(variable.lower), int(variable.upper + 1))
        processed_masked_feature = label_binarize(masked_feature, value_range, neg_label=0, pos_label=1)

        # now compute statistics! assume processed_masked_feature has shape (N_observed, value_range)
        stats["n_class"] = len(value_range)
        stats["marginal_prob"] = np.mean(processed_masked_feature, 0)
        stats["majority_vote"] = np.argmax(stats["marginal_prob"])
        stats["majority_prob"] = stats["marginal_prob"][stats["majority_vote"]]
        stats["entropy"] = -np.sum(stats["marginal_prob"] * np.log(np.clip(stats["marginal_prob"], 1e-5, 1.0)))

        return stats

    @staticmethod
    def _statistics_binary(feature, variable, mask):
        # assume we collect the statistics of a binary data
        # feature: an np.array has shape (N_data,)
        # variable: an object of variable class
        # mask: an np.array with {0, 1} entries, the missing value indicator

        assert variable.type == "binary", "expecting a binary variable, get %s" % variable.type

        # first mask out the missing part
        idx = np.where(mask == 1)[0]
        masked_feature = feature[idx]
        stats = {"type": variable.type}
        stats["missing_percentage"] = (1.0 - np.mean(mask)) * 100

        # now compute statistics! assume masked_feature has shape (N_observed,) with entries in {0, 1}
        stats["n_class"] = 2
        prob = np.mean(masked_feature)  # probability of class 1
        stats["majority_vote"] = int(prob > 0.5)  # 1 or zero
        stats["majority_prob"] = stats["majority_vote"] * prob + (1 - stats["majority_vote"]) * (1 - prob)
        stats["entropy"] = -(
            prob * np.log(np.clip(prob, 1e-5, 1.0)) + (1 - prob) * np.log(np.clip(1 - prob, 1e-5, 1.0))
        )

        return stats

    @staticmethod
    def _statistics_continuous(feature, variable, mask):
        # assume we collect the statistics of a continuous data
        # feature: an np.array has shape (N_data,)
        # variable: an object of variable class
        # mask: an np.array with {0, 1} entries, the missing value indicator

        assert variable.type == "continuous", "expecting a continuous variable, get %s" % variable.type

        # first mask out the missing part
        idx = np.where(mask == 1)[0]
        masked_feature = feature[idx]
        stats = {"type": variable.type}
        stats["missing_percentage"] = (1.0 - np.mean(mask)) * 100

        # now compute statistics! assume processed_masked_feature has shape (N_observed,)
        stats["variable_lower"] = variable.lower
        stats["variable_upper"] = variable.upper  # defined by the variable
        # stats for a box plot
        stats["data_min_val"] = np.min(masked_feature)  # the actual min value in data
        stats["data_max_val"] = np.max(masked_feature)  # the actual max value in data
        stats["mean"] = np.mean(masked_feature)
        # now compute quartiles, 25%, 50% (median), 75%
        stats["quartile_1"], stats["median"], stats["quartile_3"] = np.quantile(masked_feature, [0.25, 0.5, 0.75])
        # now for potential outliers
        stats["iqr"] = stats["quartile_3"] - stats["quartile_1"]  # interquartile range
        stats["lower_fence"] = stats["quartile_1"] - 1.5 * stats["iqr"]
        stats["upper_fence"] = stats["quartile_3"] + 1.5 * stats["iqr"]
        # one can print out-lier values potentially by the following code:
        # outliers_upper = masked_feature[np.where(masked_feature > stats["upper_fence"])[0]]
        # outliers_lower = masked_feature[np.where(masked_feature < stats["lower_fence"])[0]]

        return stats
