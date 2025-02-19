import pandas as pd
from data_prep import new_data
from selecting_features import select_dataframe
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from data_prep import target
import numpy as np


def train(clf: RandomForestClassifier, X_train: np.ndarray, y_train: np.ndarray, to_pred: np.ndarray) -> RandomForestClassifier:
    clf.fit(X_train, y_train)
    pred = clf.predict(to_pred)
    return pred


def score(preds: np.ndarray, y: np.ndarray) -> float:
    return sum(preds == y) / len(y)


def perm_importance(model: RandomForestClassifier, x: np.ndarray, y: np.ndarray, data_frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    r_dict = {}
    r = permutation_importance(model, x, y,
                        n_repeats=30,
                        random_state=42)
    for i in r.importances_mean.argsort()[::-1]:
        feature_name = data_frame.columns[i]
        importance_mean = r.importances_mean[i]
        importance_std = r.importances_std[i]
        r_dict[feature_name] = (float(importance_mean), float(importance_std))
    return r_dict



dataset_qubo_filtered = select_dataframe(new_data,
                                         '/Users/francescoaldoventurelli/qml/FeatureSelectionQubo/solution.json',
                                         visualize=False)

scaler = MinMaxScaler()
dataset_qubo_filtered_arr = scaler.fit_transform(dataset_qubo_filtered)

X_train, X_test_dummy, y_train, y_test_dummy = train_test_split(dataset_qubo_filtered_arr, target, test_size=0.5,
                                                                random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_dummy, y_test_dummy, test_size=0.33, random_state=42)

clf = RandomForestClassifier(max_depth=2, random_state=42)

preds = train(clf, X_train, y_train, X_val)
#print("Validation score:", score(preds, y_val))

r = perm_importance(clf, X_val, y_val, dataset_qubo_filtered)