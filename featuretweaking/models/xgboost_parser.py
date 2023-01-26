from __future__ import annotations

import numpy as np
import xgboost as xgb
import pandas as pd
import json


class XGBTree:
    def __init__(self, tree: xgb.Booster, feature_names):
        """Wrapper around XGBoost for a single decision tree, parsing all
        relevant information pertaining to the feature-tweaking algorithm.

        :param tree: Single XGBoost decision tree
        :param feature_names: Feature names from the XGBoost object
        """
        self.feature_names = feature_names
        self.split = {}  # indexed by node_id
        self.split_condition = {}  # indexed by node_id
        self.children_left = {}  # indexed by node_id
        self.children_right = {}  # indexed by node_id
        self.scores = {}  # indexed by leaf node_id
        self.paths = {}  # indexed by leaf node_id
        self.inequality = {}  # indexed by leaf node_id

        self.tree = tree
        self.tree_dump = tree.get_dump(dump_format="json", with_stats=False)
        self.tree_dump = self.tree_dump[0]
        self.tree_dump = json.loads(self.tree_dump)
        self._traverse_tree(self.tree_dump, [], [])

        self._feature_index_to_name()
        return

    def _feature_index_to_name(self):
        # JSON dump will sometimes return indices in tree.feature_names rather than the feature name
        for s in self.split:
            if not self.split[s] in self.feature_names:
                self.split[s] = self.feature_names[int(self.split[s])]
        return

    @staticmethod
    def logistic_function(x):
        """NOTE: if overflow ever becomes an issue, use e^x / (e^x + 1) for x << 0
        :param x:
        :return:
        """
        # return np.where(x < 0, np.exp(x) / (np.exp(x) + 1), 1 / (1 + np.exp(-x)))
        return 1 / (1 + np.exp(-x))

    def tweak_continuous_feature(self, x, split_condition, inequality, epsilon):
        """

        :param x:
        :param split_condition:
        :param inequality:
        :param epsilon:
        :return:
        """
        if inequality == 0:  # feature < threhsold
            x = split_condition + epsilon
        else:
            x = split_condition - epsilon
        return x

    def tweak_categorical_feature(self, x, split_condition, inequality, epsilon):
        """

        :param x:
        :param split_condition:
        :param inequality:
        :param epsilon:
        :return:
        """

        if inequality == 0:  # feature < threshold
            x = np.floor(split_condition + epsilon)
        else:
            x = np.roof(split_condition - epsilon)
        return x

    def generate_epsilon_satisfactory_examples(
        self, x, aim_label, continuous_features, categorical_features, epsilon, label_threshold=0.5
    ):
        """

        :param x:
        :param aim_label:
        :param continuous_features:
        :param categorical_features:
        :param epsilon:
        :param label_threshold:
        :return:
        """
        aim_leafs = [leaf for leaf in self.scores if (label_threshold < self.scores[leaf]) == aim_label]
        es_instance = pd.DataFrame(data=np.tile(x.values, [len(aim_leafs), 1]), columns=x.index)

        assert len(es_instance) == len(aim_leafs)

        for leaf, i in zip(aim_leafs, es_instance.index):
            for (node, inequality) in zip(self.paths[leaf], self.inequality[leaf]):
                feature = self.split[node]

                # Does the feature already satisfy the criterion?
                if (x[feature] < self.split_condition[node]) != bool(inequality):

                    if feature in continuous_features:
                        es_instance.loc[i, feature] = self.tweak_continuous_feature(
                            x[feature], self.split_condition[node], inequality, epsilon[feature]
                        )

                    elif feature in categorical_features:
                        es_instance.loc[i, feature] = self.tweak_categorical_feature(
                            x[feature], self.split_condition[node], inequality, epsilon[feature]
                        )
        return es_instance

    def _traverse_tree(self, dump, path, greater_than):
        """Recursive function that traverses the decision tree and records
        the path to each leaf in a dictionary
        TODO: Handle missing numbers (default/missing paths)
        :param dump: json dump of XGBoost decision tree
        :param path: current path of decision tree represented as a sequence of node ids
        :param greater_than: next step in the current path, i.e left or right child next
        :return:
        """
        node_id = dump["nodeid"]

        # Record information about the nodes
        self.split[node_id] = dump["split"]
        self.split_condition[node_id] = dump["split_condition"]
        self.children_left[node_id] = dump["yes"]
        self.children_right[node_id] = dump["no"]

        # Make copies to prevent mutation of existing lists
        path = path.copy()
        left_inequality = greater_than.copy()
        right_inequality = greater_than.copy()

        # Append path information
        path.append(node_id)
        left_inequality.append(1)  # feature is LESS LESS threhsold
        right_inequality.append(0)  # feature is GREATHER THAN threshold

        for child in dump["children"]:
            if "leaf" in child:
                leaf_id = child["nodeid"]
                self.paths[leaf_id] = path
                self.scores[leaf_id] = XGBTree.logistic_function(child["leaf"])
                self.inequality[leaf_id] = right_inequality if leaf_id == dump["no"] else left_inequality
                # self.scores[leaf_id] = child["leaf"]
            elif child["nodeid"] == dump["yes"]:
                self._traverse_tree(child, path, left_inequality)
            elif child["nodeid"] == dump["no"]:
                self._traverse_tree(child, path, right_inequality)
        return

    def __call__(self, x):
        """
        :param x: Data point(s) to evaluate with the XGBoost decision tree
        :return: Decision tree prediction (probability)
        """
        if type(x) == pd.Series:
            x = x.values
        if len(x.shape) == 1:
            x = x.reshape(-1, 1).transpose()
        return self.tree.inplace_predict(data=x, validate_features=True, predict_type="value")


class XGBEnsemble:
    def __init__(self, model):
        """Wrapper around XGBoost for an ensemble of decision trees, parsing all
        relevant information pertaining to the feature-tweaking algorithm.

        :param model: XGBoost model composed of an ensemble of decision trees.
        """
        self.model = model
        self.booster = None
        self.ensemble = []
        self._get_booster()
        self._parse_model()
        self.num_trees = len(self.ensemble)
        return

    def _get_booster(self):
        """
        Obtain the Booster object from the xgboost instance.
        :return: xgb.core.Booster
        """
        if type(self.model) is xgb.core.Booster:
            self.booster = self.model
        elif type(self.model) is xgb.sklearn.XGBModel:
            self.booster = self.model.get_booster()
        return

    def _parse_model(self):
        """

        :return:
        """
        # dump = self.booster.get_dump(dump_format="json", with_stats=False)
        for i, tree in enumerate(self.booster):
            tree_dump = tree.get_dump(dump_format="json", with_stats=False)
            tree_dump = tree_dump[0]
            tree_dump = json.loads(tree_dump)
            if not ("leaf" in tree_dump):
                self.ensemble.append(XGBTree(tree, feature_names=self.booster.feature_names))

        return

    def get_num_trees(self):
        """

        :return:
        """
        return self.num_trees

    def __call__(self, x):
        """
        :param x: Data point(s) to evaluate with the XGBoost decision tree
        :return: Decision tree prediction (probability)
        """
        if type(x) == pd.Series:
            x = x.values
        if len(x.shape) == 1:
            x = x.reshape(-1, 1).transpose()
        return self.booster.inplace_predict(data=x, validate_features=True, predict_type="value")

    def __len__(self) -> int:
        """

        :return:
        """
        return self.num_trees

    def __iter__(self):
        """

        :return:
        """
        self._iter_index = 0
        self._max_iter = len(self.ensemble) - 1
        return self

    def __next__(self):
        """

        :return:
        """
        if self._iter_index > self._max_iter:
            raise StopIteration
        out = self.ensemble[self._iter_index]
        self._iter_index += 1
        return out
