import numpy as np
from entropy_and_mi import get_matrix
from data_prep import new_data as data, target
import argparse


def make_qubo_matrix(alpha: float, R: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Create a Qubo matrix using the formula describes here:
    Q_ij(alpha) = R_ij - alpha * (R_ij + delta_ij * I_ij).
    :param alpha: (float) the alpha tuning parameter;
    :param R: (np.ndarray) the Redundancy matrix;
    :param I: (np.ndarray) the Importance vector;
    :return: qubo_matrix (np.ndarray) the Qubo matrix.
    """
    length = len(data.columns)
    qubo_matrix = (1 - alpha) * R
    for i in range(length):
        qubo_matrix[i, i] = - alpha * I[i] # since the value of R_ii = 0
    return qubo_matrix


def decode_sol(solution: dict) -> list:
    """
    Decode qubo bitstring solution.
    :param solution: (dict) A dictionary containing qubo extracted solution;
    :return: features: (list) Selected features out of qubo bitstrings.
    """
    features = [key for key in solution.keys() if solution[key] == 1]
    return features


if __name__ == "__main__":
    path = "/Users/francescoaldoventurelli/qml/FeatureSelectionQubo/r_i_matrices" ## my path
    parser = argparse.ArgumentParser(description="Feature selection using QUBO and Mutual Information")
    parser.add_argument("bins", type=int, help="Number of bins to discretize the dataset")
    args = parser.parse_args()
    R = get_matrix("r", data, target, args.bins)
    I = get_matrix("i", data, target, args.bins)
    np.savetxt(path + "/r_matrix_breastcancer.txt", R, delimiter=",")
    np.savetxt(path + "/i_matrix_breastcancer.txt", I, delimiter=",")