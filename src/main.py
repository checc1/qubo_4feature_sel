import numpy as np
from pyqubo import Array
import neal
from qubo_formulation import decode_sol, make_qubo_matrix
import argparse


def run(Q: np.ndarray) -> list:
    x = Array.create("x", shape=Q.shape[0], vartype="BINARY")
    qubo_model = sum(Q[i, j] * x[i] * x[j] for i in range(Q.shape[0]) for j in range(Q.shape[1]))
    qubo_model = qubo_model.compile()
    bqm = qubo_model.to_bqm()
    sa = neal.SimulatedAnnealingSampler()
    sampleset = sa.sample(bqm, num_reads=1_000)
    samples = qubo_model.decode_sampleset(sampleset)
    best_sample = min(samples, key=lambda s: s.energy)
    return decode_sol(best_sample)


if __name__ == "__main__":

    path = "/Users/francescoaldoventurelli/qml/FeatureSelectionQubo/r_i_matrices"
    parser = argparse.ArgumentParser(description="Running feature selection experiment as a QUBO problem"
                                                 "using Entropy and Mutual Information to estimate the "
                                                 "Redundancy (R) and Importance (I) matrices."
                                                 "The general formula the Qubo finds the solution of it"
                                                 "is Q_ij(alpha) = R_ij - alpha * (R_ij + delta_ij * I_ij)")
    parser.add_argument("alpha", type=float, help="Tuning parameter which help selecting "
                                                  "features maximizing the redundancy or the importance"
                                                  "based on its value.")
    args = parser.parse_args()
    r_matrix, i_matrix = (np.loadtxt(path + "/r_matrix_breastcancer.txt", delimiter=","),
                          np.loadtxt(path + "/i_matrix_breastcancer.txt", delimiter=","))
    Q = make_qubo_matrix(args.alpha, r_matrix, i_matrix)
    np.savetxt(path + "/Q.txt", Q, delimiter=",")