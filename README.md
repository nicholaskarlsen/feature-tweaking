# feature-tweaking
Implementation of the Feature Tweaking algorithm for XGBoost models


Interpretable Predictions of Tree-based Ensembles via Actionable Feature Tweaking, Tolomei et. al.
https://arxiv.org/abs/1706.06691


# Setup
The package may be installed by
```
pip installl git+https://github.com/nicholaskarlsen/feature-tweaking
```


# Usage 

```python
# the feature-tweaking class may be instantiated in the following way
ft = FeatureTweaking(model, continuous_features, categorical_features, continuous_metric="l1", categorical_metric="kronecker delta")

# which may then be used to generate counterfactual examples
counterfactuals = ft.generate_counterfactuals(X.iloc[i], epsilon=0.1)
# the factual class is inferred from the passed model and a pandas dataframe contiaining the set of epsilon-satisfactory examples
# is returned along with their distance from the factual example measured using the selected metric
```


| Parameter | Description |
|---|---|
| `model (xgboost.Booster | xgboost.sklearn.XGBModel)` | Ensemble decision tree model, currently onsly supports XGBoost (both sklearn and native interfaces)|
| `continuous_features (list<str>)` | list containing column names of the continuous features to tweak |
| `categorical_features(list<str>)` | list containing column names of the categorical features to tweak |
| `continuous_metric (str)`  | metric to use for all of the continuous features. see FeatureTweaking._metrics for a list of currently implemented metrics |
| `categorical_metric (str)`  | metric to use for all of the categorical features. see FeatureTweaking._categorical_metrics for a list of currently implemented metrics |

