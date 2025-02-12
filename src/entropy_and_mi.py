import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
from numba.core.target_extension import target_registry

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


def conditional_joint_entropy(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, bins: int) -> float:
    """
    Compute the joint Shannon entropy between three variables.
    :param X: (np.ndarray) The first input variable;
    :param Y: (np.ndarray) The second input variable;
    :param Z: (np.ndarray) The third input variable;
    :param bins: (int) The number of bins for discretization.
    :return: (float) The joint Shannon entropy of the three variables.
    """
    binned_dist = np.histogramdd((X, Y, Z), bins=bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    joint_e = - np.sum(probs * np.log(probs))
    return joint_e


def conditional_mutual_inf(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, bins: int) -> float:
    """
    Compute the conditional mutual information I(X;Y | Z).
    :param X: (np.ndarray) The first input variable;
    :param Y: (np.ndarray) The second input variable;
    :param Z: (np.ndarray) The conditioning variable;
    :param bins: (int) The number of bins for discretization.
    :return: (float) The conditional mutual information I(X;Y | Z).
    """
    HXZ = joint_entropy(X, Z, bins)
    HYZ = joint_entropy(Y, Z, bins)
    HXYZ = conditional_joint_entropy(X, Y, Z, bins)

    return HXZ + HYZ - HXYZ


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
    """
    Compute the redundancy matrix out of the input data.
    :param data: (pd.DataFrame) The input data as pandas dataframe;
    :param bins: (int) The number of bins to use to discretize the data to compute the entropy;
    :return: R_mtx: (np.ndarray) The redundancy matrix.
    """
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
    """
    Compute the importance vector out of the input data.
    :param data: (pd.DataFrame) The input data as pandas dataframe;
    :param target: (np.ndarray) The target variable | y labels;
    :param bins: (int) The number of bins to use to discretize the data to compute the entropy;
    :return: importance: (np.ndarray) The importance vector.
    """
    num_features = len(data.columns)
    Importance_vector = [mutual_info(data.iloc[:, i], target, bins) for i in range(num_features)]
    importance = np.array(Importance_vector)
    return importance


def plot(matrix: np.ndarray) -> None:
    """
    Plot the input matrix.
    :param matrix: (np.ndarray) The input matrix to plot;
    :return: None
    """
    sns.heatmap(matrix, annot=False, cmap="coolwarm", square=True)


def get_matrix(matrix: str, data: pd.DataFrame, target: np.ndarray, bins: int):
    """
    Compute a specific matrix based on the input provided by the user.
    :param matrix: (str) The matrix to compute. Can be r or i, otherwise raise an error;
    :param data: (pd.DataFrame) The input data as pandas dataframe;
    :param target: (np.ndarray) The target variable | y labels;
    :param bins: (int) The number of bins to use to discretize the data to compute the entropy;
    :return:
    """
    if matrix == "r":
        return fill_redundancy(data, bins)
    elif matrix == "i":
        return fill_importance(data, target, bins)
    else:
        raise ValueError("Use only r or i to get useful matrices.")


## I reconstruct this from MIQUBO
def Q(data: pd.DataFrame, target: pd.Series, k: int, lambda_ : float, bins: int = 20):
    N = len(data.columns)
    qubo = np.zeros((N, N))

    ### off-diagonal terms
    for i in range(N):
        for j in range(i, N):
            qubo[i, j] = -conditional_mutual_inf(data.iloc[:, i], target, data.iloc[:, j], bins)
            qubo[j, i] = qubo[i, j]

    # diag terms
    for i in range(N):
        qubo[i, i] = -mutual_info(data.iloc[:, i], target, bins) + lambda_ * (1 - 2 * k)

    ## penalty term
    # k features
    for i in range(N):
        for j in range(i + 1, N):
            qubo[i, j] += lambda_ * 2
            qubo[j, i] = qubo[i, j]

    return qubo


if __name__ == "__main__":
    from data_prep import new_data, target
    lambda_ = 2.8
    qubo_matrix = Q(new_data, target, 8, lambda_)
    path_to_save = "/Users/francescoaldoventurelli/qml/FeatureSelectionQubo/qubo_matrices"
    np.savetxt(path_to_save + f"/qubo_matrix_12febbraio_{lambda_}.txt", qubo_matrix, delimiter=",")