import dimod
from entropy_and_mi import mutual_info, conditional_mutual_inf
from data_prep import new_data, target
import pandas as pd
import numpy as np
import itertools
import pyqubo
import neal
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json



### some column names : "mean radius", "mean texture", "mean perimeter", "mean area"
### Importance vector
def importance(data: pd.DataFrame, target: np.ndarray, bins: int, bqm: dimod.BinaryQuadraticModel) -> None:
    """
    Compute the importance vector out of the input data.
    :param data: (pd.DataFrame) The input data as pandas dataframe;
    :param target: (np.ndarray) The target variable | y labels;
    :param bins: (int) The number of bins to use to discretize the data to compute the entropy;
    :return: importance: (np.ndarray) The importance vector.
    """
    num_features = len(data.columns)
    for i in range(num_features):
        mi_dummy = mutual_info(data.iloc[:, i].values, target, bins)
        bqm.add_variable('x' + str(i), -mi_dummy)


### to check
def check_bqm(bqm: dimod.BinaryQuadraticModel) -> None:
    for item in bqm.linear.items():
        print("{}: {:.3f}".format(item[0], item[1]))


def redundancy(data: pd.DataFrame, target: np.ndarray, bins: int, bqm: dimod.BinaryQuadraticModel) -> None:
    for i, j in itertools.combinations(range(len(data.columns)), 2):
        cmi_dummy = conditional_mutual_inf(data.iloc[:, i].values, new_data.iloc[:, j].values, target, bins)
        bqm.add_interaction('x' + str(i), 'x' + str(j), -cmi_dummy)


def fill_bqm(data: pd.DataFrame, target: np.ndarray, bins: int) -> dimod.BinaryQuadraticModel:
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    importance(data, target, bins, bqm)
    redundancy(data, target, bins, bqm)
    #for item in bqm.quadratic.items():
    #    print("{}: {:.3f}".format(item[0], item[1]))
    return bqm



def check_bqm_total(bqm: dimod.BinaryQuadraticModel) -> None:
    print("Linear Terms (Feature Importance):")
    for var, bias in bqm.linear.items():
        print(f"{var}: {bias:.3f}")
    print("\nQuadratic Terms (Feature Redundancy):")
    for (var1, var2), bias in bqm.quadratic.items():
        print(f"({var1}, {var2}): {bias:.3f}")


#print(fill_bqm(new_data, target, 10))

# Introduce the penalty term
# select a k number of features:


def importance_pyqubo(data: pd.DataFrame, target: np.ndarray, bins: int, Q_matrix: np.ndarray) -> None:
    for i in range(len(new_data.columns)):
        mi_dummy = mutual_info(data.iloc[:, i].values, target, bins)
        Q_matrix[i, i] = -mi_dummy


def redundancy_pyqubo(data: pd.DataFrame, target: np.ndarray, bins: int, Q_matrix: np.ndarray) -> None:
    for i, j in itertools.combinations(range(len(data.columns)), 2):
        cmi_dummy = conditional_mutual_inf(data.iloc[:, i].values, new_data.iloc[:, j].values, target, bins)
        Q_matrix[i, j] = cmi_dummy
        Q_matrix[j, i] = cmi_dummy


def create_qubo_matrix(data: pd.DataFrame, target: np.ndarray, bins: int, l: float) -> tuple:
    Q_matrix = np.zeros(shape=(len(new_data.columns), len(new_data.columns)))
    x = pyqubo.Array.create('x', shape=Q_matrix.shape[0], vartype='BINARY')
    importance_pyqubo(data, target, bins, Q_matrix)
    redundancy_pyqubo(data, target, bins, Q_matrix)

    ### penalty term
    k = 8 ## features I want
    for i in range(Q_matrix.shape[0]):
        Q_matrix[i, i] = Q_matrix[i, i] + l * (1 - 2 * k)
    for i in range(Q_matrix.shape[0]):
        for j in range(Q_matrix.shape[1]):
            if i != j:
                Q_matrix[i, j] = 2 * l * Q_matrix[i, j]
    Q = sum(Q_matrix[i, j] * x[i] * x[j] for i in range(Q_matrix.shape[0]) for j in range(Q_matrix.shape[1]))

    return (Q, Q_matrix)


def count_solution(best_sample: dict[str]) -> int:
    count = 0
    for key in best_sample.keys():
        if best_sample[key] == 1:
            count += 1
    return count


def plot_qmatrix(Q: np.array) -> plt.show:
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(Q, annot=False, cmap="coolwarm", square=True, alpha=0.85, cbar=True, linecolor='white', linewidths=0.5)
    plt.xlabel(r"$f_i$", fontdict={"fontsize": 16})
    plt.ylabel(r"$f_i$", fontdict={"fontsize": 16})
    return fig


if __name__ == "__main__":
    qubo_model, q_arr = create_qubo_matrix(new_data, target, 20, 0.02)
    #print(qubo_model)
    qubo = qubo_model.compile()
    bqm = qubo.to_bqm()
    sa = neal.SimulatedAnnealingSampler()
    samplest = sa.sample(bqm, num_reads=1000)
    decoded_samples = qubo.decode_sampleset(samplest)
    best_sample = min(decoded_samples, key=lambda s: s.energy)
    sol = best_sample.sample
    with open('/Users/francescoaldoventurelli/qml/FeatureSelectionQubo/solution.json', 'w') as fp:
        json.dump(sol, fp)
    print("Selected n =", count_solution(best_sample.sample), "features.")
    #plot_qmatrix(q_arr)
    #plt.savefig("/Users/francescoaldoventurelli/qml/FeatureSelectionQubo/qubo_matrix.png")
    #plt.show()

