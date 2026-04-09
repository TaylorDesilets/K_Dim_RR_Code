import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ReadFromCSV import load_data
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
# -------------------- Linear SVM model --------------------
class LinearSVM(nn.Module): # this is taken from documentation
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
    
# SVM loss with L2 penalty
def svm_loss(model, X, y, lambda_reg=0.01):
    '''
    Computes the soft-margin SVM loss with L2 regularization.

    Loss = mean(max(0, 1 - y * f(x))) + lambda * ||w||^2

    - The hinge loss penalizes points inside the margin or misclassified.
    - The L2 term controls model complexity to prevent overfitting.
    '''
    scores = model(X) # shape(n, 1)
    hinge = torch.clamp(1 - y * scores, min=0)
    loss = torch.mean(hinge)               

    # L2 regularization on weights
    w = model.linear.weight
    loss += lambda_reg * torch.sum(w ** 2)

    return loss


def load_dataset():
    '''
    Reads data from ReadFromCSV.py
    '''
    data = load_data()
    data["target"] = np.where(data["Political Affiliation"] == "Liberal", 1, -1) # thing we want to predict

    # features and response
    X = data.drop(columns=["Political Affiliation", "target", "riding", "Constituency"]) # X is all the features except these
    X = X.select_dtypes(include=[np.number]).values
    y = data["target"].values
    return X, y
def preprocess_linear(X_train, X_val):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    return X_train, X_val, scaler

def preprocess_poly_kernel(X_train, X_val, degree, gamma, coef0):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    K_train = polynomial_kernel(X_train, X_train,
                                degree=degree,
                                gamma=gamma,
                                coef0=coef0)

    K_val = polynomial_kernel(X_val, X_train,
                              degree=degree,
                              gamma=gamma,
                              coef0=coef0)

    # Convert to tensors
    K_train = torch.tensor(K_train, dtype=torch.float32)
    K_val = torch.tensor(K_val, dtype=torch.float32)

    return K_train, K_val
def preprocess_rbf_kernel(X_train, X_val, gamma):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    L_train = rbf_kernel(X_train, X_train, gamma=gamma)
    L_val = rbf_kernel(X_val, X_train, gamma=gamma)

    L_train = torch.tensor(L_train, dtype = torch.float32)
    L_val = torch.tensor(L_val, dtype = torch.float32)

    return L_train, L_val
def grid_search_kernel(X, y, kernel_type="poly", param_grid=None, k_folds=5, epochs=500):
    """
    Performs grid search with k-fold CV to find best kernel parameters.
    Returns best params and corresponding accuracy.
    """
    from itertools import product
    best_params = None
    best_acc = 0

    # Create all combinations of parameters
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        print(f"Testing params: {params}")
        avg_acc = cross_validate_kernel(X, y, kernel_type=kernel_type, k_folds=k_folds, **params, epochs=epochs)
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_params = params
    
    print(f"\nBest params for {kernel_type} kernel: {best_params}, Accuracy: {best_acc:.4f}")
    return best_params, best_acc
def cross_validate_kernel(X, y, kernel_type="poly", k_folds=5, epochs=200, **kernel_params):
    """
    k-fold CV for kernel SVM.
    Returns average accuracy.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracies = []
    
    y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y_torch[train_idx], y_torch[val_idx]
        
        # Kernel preprocessing
        if kernel_type == "poly":
            X_train, X_val = preprocess_poly_kernel(X_train_raw, X_val_raw, **kernel_params)
        elif kernel_type == "rbf":
            X_train, X_val = preprocess_rbf_kernel(X_train_raw, X_val_raw, **kernel_params)
        elif kernel_type == "linear":
            X_train, X_val, _ = preprocess_linear(X_train_raw, X_val_raw)
        
        # Train model
        model = LinearSVM(X_train.shape[1])
        train_svm(model, X_train, y_train, lr=0.01, lambda_reg=0.01, epochs=epochs)
        
        # Evaluate
        with torch.no_grad():
            outputs = model(X_val)
            predictions = torch.sign(outputs).detach().numpy()
            y_true = y_val.numpy()
            acc = accuracy_score(y_true, predictions)
            accuracies.append(acc)
    
    avg_acc = np.mean(accuracies)
    return avg_acc

def split_data(X, y, test_size=0.25, random_state=42):
    '''
    training: 75% of the data, testing: 25% of the data. Stratifying on y
    '''
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_svm(model, X_train, y_train, lr=0.01, lambda_reg=0.01, epochs=1000):
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = svm_loss(model, X_train, y_train, lambda_reg)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

def evaluate_model(model, X_val, y_val, kernel_name="Kernel"):
    with torch.no_grad():
        outputs = model(X_val)
        predictions = torch.sign(outputs)
        y_pred = predictions.detach().numpy()
        y_true = y_val.numpy()

    print(f"\n=== Results for {kernel_name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not Liberal", "Liberal"]
    )
    plt.title(f"Confusion Matrix ({kernel_name})")
    plt.show()
poly_grid = {
    "degree": [2, 3, 4],
    "gamma": [0.01, 0.05, 0.1, 0.5, 1, 2],
    "coef0": [0, 1, 2]
}
rbf_grid = {
    "gamma": [0.001, 0.01, 0.05, 0.1, 0.5]
}
linear_grid = {
    "dummy": [None]  # placeholder so grid_search works
}
# Load and split data
X, y = load_dataset()
best_rbf_params, best_rbf_acc = grid_search_kernel(X, y, kernel_type="rbf", param_grid=rbf_grid, k_folds=5, epochs=500)
best_poly_params, best_poly_acc = grid_search_kernel(X, y, kernel_type="poly", param_grid=poly_grid, k_folds=5, epochs=500)
best_linear_params, best_linear_acc = grid_search_kernel(
    X, y, kernel_type="linear", param_grid=linear_grid, k_folds=5, epochs=500
)
X_train_raw, X_val_raw, y_train_np, y_val_np = split_data(X, y)
y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

# --- Polynomial kernel ---
X_train_poly, X_val_poly = preprocess_poly_kernel(X_train_raw, X_val_raw, **best_poly_params)
model_poly = LinearSVM(X_train_poly.shape[1])
losses_poly = train_svm(model_poly, X_train_poly, y_train, lr=0.01, lambda_reg=0.01, epochs=1000)
evaluate_model(model_poly, X_val_poly, y_val, kernel_name="Polynomial")

# --- RBF kernel ---
X_train_rbf, X_val_rbf = preprocess_rbf_kernel(X_train_raw, X_val_raw, **best_rbf_params)
model_rbf = LinearSVM(X_train_rbf.shape[1])
losses_rbf = train_svm(model_rbf, X_train_rbf, y_train, lr=0.01, lambda_reg=0.01, epochs=1000)
evaluate_model(model_rbf, X_val_rbf, y_val, kernel_name="RBF")

# --- Linear kernel ---
X_train_lin, X_val_lin, _ = preprocess_linear(X_train_raw, X_val_raw)

model_lin = LinearSVM(X_train_lin.shape[1])
losses_lin = train_svm(model_lin, X_train_lin, y_train, lr=0.01, lambda_reg=0.01, epochs=1000)

evaluate_model(model_lin, X_val_lin, y_val, kernel_name="Linear")

plt.figure()

plt.plot(losses_poly, label="Polynomial")
plt.plot(losses_rbf, label="RBF")
plt.plot(losses_lin, label="Linear")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence of SVM Training (Loss vs Epoch)")
plt.legend()

plt.show()