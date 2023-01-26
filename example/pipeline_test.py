import os
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from featuretweaking import FeatureTweaking


def generate_random_dataset():
    np.random.seed(42)
    samples = 10_000
    feats = 1000
    df = pd.DataFrame(
        np.random.randn(samples, feats + 1),
        columns=["target"] + [f"feat_{i}" for i in range(feats)],
    )
    df["target"] = df["target"] >= 0
    y = df["target"]
    X = df.drop("target", axis=1)
    return X, y


def train_random_model():
    X, y = generate_random_dataset()
    dmat = xgb.DMatrix(data=X, label=y, enable_categorical=False)
    params = {
        "n_estimators": 100,
        "objective": "binary:logistic",
        "eval_metric": ["error", "logloss", "auc"],
        "eta": 0.02,
        "min_child_weight": 1,
        "max_depth": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7,
        "lambda": 1,
        "alpha": 0,
        "gamma": 15,
        "seed": 41,
    }
    if os.path.isfile("model.json"):
        bst = xgb.Booster()
        bst.load_model("model.json")
        print("loaded model!")
    else:
        bst = xgb.train(params, dmat, num_boost_round=10)
        bst.save_model("model.json")
        print("saved mode!")
    return X, y, bst


def train_cancer():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data=data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"])
    scaler = MinMaxScaler()
    X.loc[:, X.columns] = scaler.fit_transform(X)

    dmat = xgb.DMatrix(data=X, label=y, enable_categorical=True)

    if os.path.isfile("model_cancer.json"):
        bst = xgb.Booster()
        bst.load_model("model_cancer.json")
        print("loaded model!")
    else:
        params = {
            "objective": "binary:logistic",
            "seed": 41,
        }
        bst = xgb.train(params, dmat, num_boost_round=10)
        bst.save_model("model_cancer.json")
        print("saved mode!")

    return X, y, bst, scaler


if __name__ == "__main__":
    # X, y, bst = train_random_model()
    X, y, bst, scaler = train_cancer()
    ft = FeatureTweaking(bst, continuous_features=X.columns)
    print(y.iloc[0], "->", not y.iloc[0])
    ft.generate_counterfactuals(X.iloc[0], epsilon=0.1)
