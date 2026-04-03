import numpy as np
import pandas as pd

from RRAlgorithm import make_rr_k_matrix, privatize_labels, fit_privatized_mlr
from NeuralNet import learn_transition_matrix


def collapse_severity(x):
    if x in [0, 1]:
        return 0
    elif x in [2, 3]:
        return 1
    elif x == 4:
        return 2
    else:
        return np.nan


def load_person_data(filepath="person.csv"):
    return pd.read_csv(filepath, low_memory=False)


def prepare_real_data(filepath="person.csv", sample_size=100000, random_state=42):
    df = load_person_data(filepath)

    df["Y"] = df["INJ_SEV"].apply(collapse_severity)

    cols = ["AGE", "SEX", "PER_ALCH", "PER_DRUG"]

    df = df[cols + ["Y"]].dropna()
    df = df[(df["AGE"] >= 0) & (df["AGE"] <= 120)]

    df = df.sample(n=sample_size, random_state=random_state)

    df["AGE"] = (df["AGE"] - df["AGE"].mean()) / df["AGE"].std()

    X = pd.get_dummies(df[cols], drop_first=True)
    X = X.astype(float).values
    Y = df["Y"].astype(int).values

    return X, Y, df


def print_class_counts(Y):
    print("Class counts:")
    print(pd.Series(Y).value_counts().sort_index())


def fit_nonprivate_model(X, Y):
    k = 3
    P_identity = np.eye(k)
    B_hat, cov, result = fit_privatized_mlr(X, Y, P_identity)
    return B_hat, cov, result, P_identity, Y


def fit_private_rr_model(X, Y, epsilon=0.5, seed=42):
    k = 3
    P_rr = make_rr_k_matrix(k, epsilon)
    Y_star = privatize_labels(Y, P_rr, seed=seed)
    B_hat, cov, result = fit_privatized_mlr(X, Y_star, P_rr)
    return B_hat, cov, result, P_rr, Y_star


def fit_private_orr_model(X, Y, gamma=0.5, seed=42):
    k = 3

    # learn transition matrix from neural network
    P_orr = learn_transition_matrix(X, Y, k, gamma=gamma)

    # privatize labels using learned matrix
    Y_star_orr = privatize_labels(Y, P_orr, seed=seed)

    # fit privatized model
    B_hat, cov, result = fit_privatized_mlr(X, Y_star_orr, P_orr)

    return B_hat, cov, result, P_orr, Y_star_orr


def run_real_data_analysis(filepath="person.csv", sample_size=100000, epsilon=0.5, gamma=0.5, random_state=42):
    X, Y, df = prepare_real_data(
        filepath=filepath,
        sample_size=sample_size,
        random_state=random_state
    )

    print_class_counts(Y)

    # 1. Non-private
    B_np, cov_np, result_np, P_np, Y_np = fit_nonprivate_model(X, Y)

    # 2. Standard RR
    B_rr, cov_rr, result_rr, P_rr, Y_star_rr = fit_private_rr_model(
        X, Y, epsilon=epsilon, seed=random_state
    )

    # 3. ORR / learned mechanism
    B_orr, cov_orr, result_orr, P_orr, Y_star_orr = fit_private_orr_model(
    X, Y, gamma=gamma, seed=random_state
    )

    print("\nNP fit success:", result_np.success)
    print("NP message:", result_np.message)

    print("\nRR fit success:", result_rr.success)
    print("RR message:", result_rr.message)

    print("\nORR fit success:", result_orr.success)
    print("ORR message:", result_orr.message)

    print("\nRR transition matrix:")
    print(P_rr)

    print("\nORR learned transition matrix:")
    print(P_orr)

    return {
        "X": X,
        "Y": Y,
        "df": df,

        "np": {
            "B_hat": B_np,
            "cov": cov_np,
            "result": result_np,
            "P": P_np,
            "Y_used": Y_np
        },

        "rr": {
            "B_hat": B_rr,
            "cov": cov_rr,
            "result": result_rr,
            "P": P_rr,
            "Y_star": Y_star_rr
        },

        "orr": {
            "B_hat": B_orr,
            "cov": cov_orr,
            "result": result_orr,
            "P": P_orr,
            "Y_star": Y_star_orr
        }
    }