import numpy as np


class ImputationStatistics:
    @classmethod
    def get_statistics(cls, data, variables, verbose=False):
        # get statistics on the marginal distributions
        # data: an np.array that has shape (sample_count, N_data, N_feature)
        stats = {}
        for var_idx, var in enumerate(variables):
            if var.type == "continuous":
                stats[var_idx] = cls._statistics_continuous(data[:, :, var_idx].astype(float), var)
            elif var.type == "binary":
                stats[var_idx] = cls._statistics_binary(data[:, :, var_idx].astype(float), var)
            elif var.type == "categorical":
                stats[var_idx] = cls._statistics_categorical(data[:, :, var_idx].astype(float), var)
            elif var.type == "text":
                stats[var_idx] = cls._statistics_text(data[:, :, var_idx].astype(str), var)
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
    def _statistics_categorical(feature, variable):
        # assume we collect the statistics of a categorical data
        # feature: an np.array has shape (sample_count, N_data)
        # variable: an object of variable class

        assert variable.type == "categorical", "expecting a categorical variable, get %s" % variable.type
        stats = {"type": variable.type}

        # convert to one-hot
        from sklearn.preprocessing import label_binarize

        value_range = range(int(variable.lower), int(variable.upper + 1))
        sample_count, N_data = feature.shape
        stats["n_class"] = len(value_range)
        # If a feature is categorical, some methods such as mean imputing treating this as continous. Thus convert it to int first.
        int_feature = np.rint(feature.reshape((-1)))
        processed_feature = label_binarize(int_feature, classes=value_range, neg_label=0, pos_label=1).reshape(
            (sample_count, N_data, stats["n_class"])
        )

        # now compute statistics! assume processed_feature has shape (sample_count, N_data, n_class)
        stats["marginal_prob"] = np.mean(processed_feature, axis=0)
        stats["majority_vote"] = np.argmax(stats["marginal_prob"], axis=-1)
        stats["majority_prob"] = stats["marginal_prob"][np.arange(N_data), stats["majority_vote"]]
        stats["majority_vote"] += variable.lower  # shift the category back if variable.lower != 0
        stats["entropy"] = -np.sum(
            stats["marginal_prob"] * np.log(np.clip(stats["marginal_prob"], 1e-5, 1.0)),
            axis=-1,
        )

        return stats

    @staticmethod
    def _statistics_binary(feature, variable):
        # assume we collect the statistics of a binary data
        # feature: an np.array has shape (sample_count, N_data)
        # variable: an object of variable class

        assert variable.type == "binary", "expecting a binary variable, get %s" % variable.type
        stats = {"type": variable.type}

        # now compute statistics! assume feature has shape (sample_count, N_data) with entries in {0, 1}
        stats["n_class"] = 2
        prob = np.mean(feature, axis=0)  # probability of class 1
        stats["majority_vote"] = np.asarray(prob > 0.5, dtype="f")  # 1 or zero
        stats["majority_prob"] = stats["majority_vote"] * prob + (1 - stats["majority_vote"]) * (1 - prob)
        stats["entropy"] = -(
            prob * np.log(np.clip(prob, 1e-5, 1.0)) + (1 - prob) * np.log(np.clip(1 - prob, 1e-5, 1.0))
        )

        return stats

    @staticmethod
    def _statistics_continuous(feature, variable):
        # assume we collect the statistics of a continuous data
        # feature: an np.array has shape (sample_count, N_data)
        # variable: an object of variable class

        assert variable.type == "continuous", "expecting a continuous variable, get %s" % variable.type
        stats = {"type": variable.type}

        # now compute statistics! assume processed_masked_feature has shape (N_observed,)
        stats["variable_lower"] = variable.lower
        stats["variable_upper"] = variable.upper  # defined by the variable
        # stats for a box plot
        stats["min_val"] = np.min(feature, axis=0)  # the actual min value in data
        stats["max_val"] = np.max(feature, axis=0)  # the actual max value in data
        stats["mean"] = np.mean(feature, axis=0)
        # now compute quartiles, 25%, 50% (median), 75%
        stats["quartile_1"], stats["median"], stats["quartile_3"] = np.quantile(feature, [0.25, 0.5, 0.75], axis=0)

        return stats

    @staticmethod
    def _statistics_text(feature, variable):
        assert variable.type == "text", "expecting a text variable, get %s" % variable.type
        stats = {"type": variable.type}

        # TODO #18598: Add metrics for text variable
        # To do so, we probably need to add decoding capability to text embedder first

        return stats
