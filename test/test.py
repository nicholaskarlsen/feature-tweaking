import re
import pytest

import xgboost as xgb
import numpy as np
import pandas as pd


import featuretweaking.models


@pytest.mark.skip(reason="not relevant to functionality")
def generate_random_dataset():
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


@pytest.mark.skip(reason="not relevant to functionality")
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
    bst = xgb.train(params, dmat, num_boost_round=10)
    return X, y, bst


def test_xgbtree_traverse_tree():
    X, y, bst = train_random_model()

    for booster in bst:
        inst = featuretweaking.models.XGBTree(booster, X.columns)
        json_dump = booster.get_dump(dump_format="json", with_stats=False)
        json_dump = " ".join(json_dump)  # convert to a single string
        match = re.findall(pattern="leaf", string=json_dump)
        num_matches = len(match)
        assert num_matches == len(inst.paths)
        assert num_matches == len(inst.scores)

    return


def test_xgb_eval():
    X, y, bst = train_random_model()
    inst = featuretweaking.models.XGBEnsemble(bst)
