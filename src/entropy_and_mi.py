import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")


def entropy(X: np.ndarray, bins: int) -> float:
    """
    Compute the Shannon entropy of a variable X.
    :param X: (np.ndarray) The input variable;
    :return: (float) The Shannon entropy of the input variable.
    """
    binned_dist = np.histogram(X, bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    entropy_ = - np.sum(probs * np.log(probs))
    return entropy_


def joint_entropy(X: np.ndarray, Y: np.ndarray, bins: int) -> float:
    """
    Compute the joint Shannon entropy between two variables.
    Each variable could be two features expressed as arrays or a feature and a label.
    :param X: (np.ndarray) The first input variable (usually feature);
    :param Y: (np.ndarray) The second input variable (feature or label);
    :return: (float) The joint Shannon entropy of the two variables.
    """
    binned_dist = np.histogram2d(X, Y, bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    joint_e = - np.sum(probs * np.log(probs))
    return joint_e


def mutual_info(X: np.ndarray, Y: np.ndarray, bins: int) -> float:
    """
    Compute the mutual information between two input variables.
    :param X: (np.ndarray) The first input variable (usually feature);
    :param Y: (np.ndarray) The second input variable (feature or label);
    :return: The mutual information between the two variables.
    """
    HX = entropy(X, bins)
    HY = entropy(Y, bins)
    HXHY = joint_entropy(X, Y, bins)
    H = HX + HY - HXHY
    return H


def fill_redundancy(data: pd.DataFrame, bins: int) -> np.ndarray:
    num_features = len(data.columns)
    R_mtx = np.zeros((num_features, num_features), dtype=np.float32)
    for i in range(num_features):
        for j in range(i, num_features):
            if i == j:
                R_mtx[i, j] = 0
            else:
                R_mtx[i, j] = mutual_info(data.iloc[:, i], data.iloc[:, j], bins)
                R_mtx[j, i] = R_mtx[i, j]
    return R_mtx


def fill_importance(data: pd.DataFrame, target: np.ndarray, bins: int) -> np.ndarray:
    num_features = len(data.columns)
    Importance_vector = [mutual_info(data.iloc[:, i], target, bins) for i in range(num_features)]
    return np.array(Importance_vector)


def plot(matrix: np.ndarray) -> None:
    sns.heatmap(matrix, annot=False, cmap="coolwarm", square=True)


def get_matrix(matrix: str, data: pd.DataFrame, target: np.ndarray, bins: int):
    if matrix == "r":
        return fill_redundancy(data, bins)
    elif matrix == "i":
        return fill_importance(data, target, bins)
    else:
        raise ValueError("Use only r or i to get useful matrices.")
