import pyqubo
import neal
import numpy as np
from entropy_and_mi import fill_importance, fill_redundancy


def qubo_formulation(redundancy, importance, alpha=0.5, importance_threshold=1e-8, threshold_penalty=None) -> np.ndarray:
    """
    Compute the QUBO formulation.
    :param redundancy: (np.ndarray) The redundancy matrix;
    :param importance: (np.ndarray) The importance vector;
    :param alpha: (float) The tuning alpha parameter;
    :param importance_threshold: (float) The importance threshold;
    :param threshold_penalty: (float) The threshold penalty;
    :return: Q: (np.ndarray) The QUBO matrix.
    """
    Q = (1-alpha)*redundancy - alpha*importance
    """pen = np.linalg.norm(Q, np.inf) if threshold_penalty is None else threshold_penalty
    diag = np.diag(Q).copy()
    diag[diag>-importance_threshold] = pen
    np.fill_diagonal(Q, diag)"""
    return Q


### TODO: another version!!!
def qfs_new(redundancy, importance, k: int, importance_threshold=1e-8, threshold_penalty=None, tol=1e-4) -> tuple[float, dict[str, int]]:
    """
    Perform Quadratic Feature Selection (QFS) algorithm.
    :param redundancy: (np.ndarray) The redundancy matrix;
    :param importance: (np.ndarray) The importance vector;
    :param k: (int) The number of features to select;
    :param importance_threshold: (float) The importance threshold;
    :param threshold_penalty: (float) The threshold penalty;
    :param tol: (float) The tolerance;
    :return: (tuple) The optimal alpha and the selected features.
    """

    def qubo_sol(alpha: float) -> dict[str, int]:
        q = qubo_formulation(redundancy, importance, alpha, importance_threshold, threshold_penalty)
        x = pyqubo.Array.create('x', shape=q.shape[0], vartype='BINARY')
        q = sum(q[i, j] * x[i] * x[j] for i in range(q.shape[0]) for j in range(q.shape[1]))
        model = q.compile()
        bqm = model.to_bqm()
        sa = neal.SimulatedAnnealingSampler()
        sampleset = sa.sample(bqm, num_reads=100)
        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda s: s.energy)
        return best_sample.sample

    # perform QFS algorithm
    a, b = 0.0, 1.0
    while (b - a) > tol:
        alpha_mid = (b + a) / 2
        sol_x = qubo_sol(alpha_mid)
        k_ = sum(sol_x.values())
        if k_ == k:
            return alpha_mid, sol_x
        elif k_ < k:
            a = alpha_mid
        else:
            b = alpha_mid
    alpha_mid = (b + a) / 2
    sol_x = qubo_sol(alpha_mid)
    return alpha_mid, sol_x



if __name__ == "__main__":
    from data_prep import new_data, target
    import argparse

    parser = argparse.ArgumentParser(description="QFS Algorithm")
    parser.add_argument("bins", type=int, default=20, help="Number of bins to discretize the data.")
    parser.add_argument("k", type=int, help="Number of features to select.")
    args = parser.parse_args()

    BINS, k = args.bins, args.k

    R = fill_redundancy(new_data, BINS)
    I = fill_importance(new_data, target, BINS)

    print(f"Starting QFS...searching for {k} features.")
    alpha_star, selected_features = qfs_new(R, I, k)
    print(f"Optimal Alpha: {alpha_star}")
    print("Selected Features:", selected_features)
    print(f"Founded {sum(selected_features.values())} features.")