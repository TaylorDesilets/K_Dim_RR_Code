
This repository contains the code used in the paper *“A Continuous Time Random Response Algorithm for Categorical Data”*, by Ce Zhang, Forough Fazeliasl, Linglong Kong, and Taylor Desilets.

---

### Running the Code

To run the code, execute the following command in your terminal:

```bash
python main.py
```
I will go over Each File in this Repository and each of their purposes.
---
### 1. RRAlgorithm.py: Implementation of Privatisation Mechanism
This module simulates, privatizes, and estimates a multinomial logistic regression (MLR) model under a \(k\)-class randomized response mechanism.

### Main Functions

- **`make_rr_k_matrix(k, epsilon)`**  
  Constructs the \(k \times k\) randomized response matrix \(P\).  
  - Diagonal: probability of reporting the true label  
  - Off-diagonal: probability of misreporting  
  - Smaller \( \epsilon \) → stronger privacy

- **`multinomial_probs(X, B)`**  
  Computes class probabilities from a multinomial logistic regression model (last class as baseline).

- **`generate_data(n, d, B, cov_type, seed)`**  
  Simulates covariates \(X\), true probabilities, and labels \(Y\) from the MLR model.

- **`privatize_labels(Y, P)`**  
  Applies the randomized response mechanism to produce privatized labels \(Y^*\).

- **`observed_probs(X, B, P)`**  
  Computes observed probabilities  
  \[
  q_{ij} = \sum_k \Pr(Y^* = j \mid Y = k)\Pr(Y = k \mid X_i)
  \]

- **`neg_loglik(beta_vec, X, Y_star, P, k, lambda_reg)`**  
  Negative log-likelihood for privatized data with L2 regularization.

- **`fit_privatized_mlr(X, Y_star, P)`**  
  Estimates model parameters using BFGS optimization. Returns \( \hat{B} \), covariance (if available), and optimizer output.

### Procedure

1. Generate data with `generate_data`  
2. Create privacy mechanism with `make_rr_k_matrix`  
3. Privatize labels using `privatize_labels`  
4. Estimate parameters with `fit_privatized_mlr`

---

### 2. SettingsImplementation.py: Implementation of Benchmarking Method to test our algorithm. 

We compare three settings:
- Non-private estimation  (NP)
- Standard \(k\)-class randomized response (RR)  
- Our Optimal Method for \(k\)-class randomized response (ORR-k-D-R)  

### Main Functions

- **`fit_np(X, Y, k)`**  
  Setting 1: Non-private model.  
  Uses the identity matrix (no noise) and fits the model directly.

- **`fit_rr_kdr(X, Y, epsilon, k, seed=None)`**  
  Setting 2: Standard randomized response.  
  - Constructs \(P\) using `make_rr_k_matrix`  
  - Privatizes labels \(Y \rightarrow Y^*\)  
  - Fits the model using privatized data  

- **`fit_orr_kdr(X, Y, epsilon, k, gamma=0.5, seed=None)`**  
  Setting 3: Placeholder ORR-k-D-R method.  
  - Uses a perturbed RR matrix  
  - Projects rows onto the probability simplex  
  - Simulates a learned privatization mechanism  

### Helper Functions

- **`project_rows_to_simplex(A)`**  
  Ensures each row is positive and sums to 1.

- **`make_mock_orr_k_matrix(k, epsilon, gamma=0.5)`**  
  Creates a modified RR matrix by perturbing the standard RR mechanism and re-normalizing.





