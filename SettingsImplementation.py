
import numpy as np
from RRAlgorithm import fit_privatized_mlr, privatize_labels, make_rr_k_matrix
from NeuralNet import learn_transition_matrix


def fit_np(X, Y, k):
    """
    Setting 1: Non-private fit.
    """
    P_identity = np.eye(k)
    return fit_privatized_mlr(X, Y, P_identity)


def fit_rr_kdr(X, Y, epsilon, k, seed=None):
    """
    Setting 2: Standard k-dimensional randomized response.
    """
    P_rr = make_rr_k_matrix(k, epsilon)
    Y_star = privatize_labels(Y, P_rr, seed=seed)
    return fit_privatized_mlr(X, Y_star, P_rr), P_rr, Y_star


def project_rows_to_simplex(A):
    """
    Force rows to be positive and sum to 1.
    """
    A = np.clip(A, 1e-8, None)
    return A / A.sum(axis=1, keepdims=True)




def fit_orr_kdr(X, Y, epsilon, k, gamma=0.5, seed=None):
    """
    Setting 3: ORR-k-D-R with learned transition matrix.
    """
    P_orr = learn_transition_matrix(X, Y, k, gamma=gamma)

    Y_star = privatize_labels(Y, P_orr, seed=seed)

    return fit_privatized_mlr(X, Y_star, P_orr), P_orr, Y_star