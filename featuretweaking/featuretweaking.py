import pandas as pd
import xgboost as xgb
import featuretweaking.metrics
import copy
from typing import Union
from featuretweaking.models import XGBEnsemble


class FeatureTweaking:
    _metrics = {
        "l1": featuretweaking.metrics.L1,
        "l2": featuretweaking.metrics.L2,
        "relative l1": featuretweaking.metrics.L1Relative,
        "relative l2": featuretweaking.metrics.L2Relative,
    }

    _categorical_metrics = {
        "kronecker delta": featuretweaking.metrics.KroneckerDelta,
    }

    def __init__(
        self,
        model: Union[xgb.sklearn.XGBModel, xgb.Booster],
        continuous_features: list = [],
        categorical_features: list = [],
        continuous_metric: str = "l1",
        categorical_metric: str = "kronecker delta",
    ):
        """Implementation of the feature tweaking algorithm for XGBoost models.

        Interpretable Predictions of Tree-based Ensembles via Actionable Feature Tweaking, Tolomei et. al. 2017
        https://doi.org/10.48550/arXiv.1706.06691

        :param model: Ensemble decision tree model, currently only supports XGBoost (both sklearn and native interfaces).
        :param continuous_features: list containing column names of the continuous features to tweak
        :param categorical_features: list containing column names of the categorical features to tweak
        :param continuous_metric: metric to use for all of the continuous features
        :param categorical_metric: metric to use for all of the categorical features
        """
        self.ensemble = XGBEnsemble(model)
        self.num_trees = self.ensemble.get_num_trees()

        self.metric = self._metrics[continuous_metric]()
        self.categorical_metric = self._categorical_metrics[categorical_metric]()

        self.continuous_features = continuous_features
        self.categorical_features = categorical_features

        # TODO: Handle ordinal encoded features?
        # TODO: Also handle LGBM?

        return

    def create_eps_vec(
        self, x: Union[pd.Series, pd.DataFrame], epsilon_value: Union[int, float]
    ) -> pd.Series:
        """Creates a vector of epsilon values based of a single number.

        :param x: A single factual example
        :param epsilon_value:
        :return:
        """
        epsilon_series = copy.copy(x)
        epsilon_series.loc[:] = epsilon_value
        return epsilon_series

    def generate_counterfactuals(
        self,
        x: Union[pd.Series, pd.DataFrame],
        label_threshold: float = 0.5,
        epsilon: Union[int, float] = 0.1,
    ) -> pd.DataFrame:
        """Generates a set of counterfactual examples based on the single, factual example x.

        :param x: Factual example
        :param label_threshold: Threshold value used for true/false prediction.
        :param epsilon: epsilon value to use
        :return: Pandas dataframe containing a set of counterfactual examples
        """
        if type(epsilon) is float:
            epsilon = self.create_eps_vec(x, epsilon)

        # Store the examples in a list of pandas dataframes (joined later to optimize memory allocations)
        counterfactuals = []

        ensemble_prediction = self.ensemble(x)
        ensemble_label = ensemble_prediction > label_threshold
        aim_label = int(not ensemble_label)

        num_counterfactuals = 0

        for i, tree in enumerate(self.ensemble):
            tree_prediction = tree(x)
            tree_label = tree_prediction > label_threshold
            # NOTE: The second check seems rather redundant, as we are
            # only ever interested in flipping the majority vote and thus
            # each of the two checks implies the other.
            # However, this was the form in CARLA, so I may be missing something.
            if tree_label == ensemble_label and tree_label != aim_label:
                es_instance = tree.generate_epsilon_satisfactory_examples(
                    x=x,
                    aim_label=aim_label,
                    label_threshold=label_threshold,
                    epsilon=epsilon,
                    continuous_features=self.continuous_features,
                    categorical_features=self.categorical_features,
                )
                if len(es_instance) > 0:
                    is_counterfactual = (self.ensemble(es_instance) > label_threshold) == aim_label
                    if is_counterfactual.sum() > 0:
                        counterfactuals.append(es_instance.iloc[is_counterfactual])
                        num_counterfactuals += is_counterfactual.sum()

        if num_counterfactuals == 0:
            return None

        # Place the generated counterfactuals into a single dataframe
        if isinstance(x, pd.Series):
            counterfactuals_df = pd.DataFrame(columns=x.index, index=range(num_counterfactuals))
        if isinstance(x, pd.DataFrame):
            counterfactuals_df = pd.DataFrame(columns=x.columns, index=range(num_counterfactuals))

        # Fetch counterfactuals from the list of counterfactuals
        i = 0
        for cf in counterfactuals:
            j = len(cf)
            counterfactuals_df.iloc[i:i + j] = cf.iloc[:]
            i += j

        # Compute the error and place it as a new column in the outgoing dataframe
        counterfactuals_df.loc[:, "error"] = None
        for i in counterfactuals_df.index:
            counterfactuals_df.loc[i, "error"] = self.metric(
                a=counterfactuals_df.loc[i, self.continuous_features], b=x[self.continuous_features]
            )
            counterfactuals_df.loc[i, "error"] += self.categorical_metric(
                a=counterfactuals_df.loc[i, self.categorical_features], b=x[self.categorical_features]
            )

        counterfactuals_df.sort_values(by="error", inplace=True)

        return counterfactuals_df

    def __call__(
        self,
        x: Union[pd.Series, pd.DataFrame],
        label_threshold: float = 0.5,
        epsilon: Union[int, float] = 0.1,
    ) -> pd.DataFrame:
        """Generates a set of counterfactual examples based on the single, factual example x.

        :param x: Factual example
        :param label_threshold: Threshold value used for true/false prediction.
        :param epsilon: epsilon value to use
        :return: Pandas dataframe containing a set of counterfactual examples
        """
        return self.generate_counterfactuals(x, label_threshold, epsilon)
