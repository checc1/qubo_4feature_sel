import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import argparse
import neal
from pyqubo import Array


def load_qubo(file: str):
    return np.array(np.loadtxt(file, delimiter=","))


class QuboProblem:
    def __init__(self, qubo_matrix: np.ndarray) -> None:
        self.qubo_matrix = qubo_matrix
        self.n_features = qubo_matrix.shape[0]

    def solve(self) -> dict[str]:
        Q_matrix = self.qubo_matrix
        x = Array.create('x', shape=Q_matrix.shape[0], vartype='BINARY')

        # Formulate the QUBO model
        qubo_model = sum(Q_matrix[i, j] * x[i] * x[j] for i in range(Q_matrix.shape[0]) for j in range(Q_matrix.shape[1]))
        qubo_model = qubo_model.compile()
        qubo = qubo_model.to_bqm()
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(qubo, num_reads=1_000)
        samples = qubo_model.decode_sampleset(sampleset)
        best_sample = min(samples, key=lambda s: s.energy)
        return best_sample.sample

    def decode_sol(self) -> list[str]:
        """
        Decode qubo bitstring solution.
        :param solution: (dict) A dictionary containing qubo extracted solution;
        :return: features: (list) Selected features out of qubo bitstrings.
        """
        solution = self.solve()
        features = [key for key in solution.keys() if solution[key] == 1]
        return features


if __name__ == "__main__":

    path = "/Users/francescoaldoventurelli/qml/FeatureSelectionQubo/qubo_matrices" ## my path
    file = "/qubo_matrix_12febbraio_0.644.txt"
    parser = argparse.ArgumentParser(description="Loading QUBO matrix from a text file")
    args = parser.parse_args()

    qubo_matrix = load_qubo(path + file)

    qubo_solv = QuboProblem(qubo_matrix)
    sol = qubo_solv.decode_sol()
    print(sol)
    print("Selected n =", len(sol), "features.")
    #print(qubo_matrix)
